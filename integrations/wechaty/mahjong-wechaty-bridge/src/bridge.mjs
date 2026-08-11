/**
 * WeChaty 通道桥接进程。
 *
 * 本文件只负责微信协议适配、消息可靠转发和受控外发，不参与麻将业务决策。
 * 收到的原始消息统一交给 Python Runtime；所有外发再次经过开关、幂等和敏感词兜底。
 */
import { WechatyBuilder } from 'wechaty'
import qrcodeTerminal from 'qrcode-terminal'
import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const DEFAULT_ENDPOINT = 'http://127.0.0.1:8790/api/channels/wechaty/raw'
const DEFAULT_REFERENCE_ENDPOINT = 'http://127.0.0.1:8790/api/message-references/link'
const BRIDGE_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const REPO_ROOT = path.resolve(BRIDGE_DIR, '../../..')
const LOCAL_DATA_DIR = process.env.MAHJONG_WECHATY_LOCAL_DATA_DIR || path.join(REPO_ROOT, 'data')
const INGRESS_SPOOL_DIR = process.env.MAHJONG_WECHATY_INGRESS_SPOOL_DIR || path.join(LOCAL_DATA_DIR, 'wechaty_ingress_spool')
const DELIVERY_LEDGER_PATH =
  process.env.MAHJONG_WECHATY_DELIVERY_LEDGER_PATH || path.join(LOCAL_DATA_DIR, 'wechaty_delivery_ledger.jsonl')
const CONTACT_ALIASES_PATH =
  process.env.MAHJONG_WECHATY_CONTACT_ALIASES_PATH || path.join(LOCAL_DATA_DIR, 'wechaty_contact_aliases.local.json')

const endpoint = process.env.MAHJONG_WECHATY_RAW_ENDPOINT || DEFAULT_ENDPOINT
const referenceEndpoint = process.env.MAHJONG_WECHATY_REFERENCE_ENDPOINT || DEFAULT_REFERENCE_ENDPOINT
const botName = process.env.MAHJONG_WECHATY_BOT_NAME || 'mahjong-wechaty-bridge'
const agentApiToken = loadAgentApiToken()
const outboundEnabled = process.env.MAHJONG_WECHATY_OUTBOUND_ENABLED
  ? truthy(process.env.MAHJONG_WECHATY_OUTBOUND_ENABLED)
  : true
const outboundPort = Number(process.env.MAHJONG_WECHATY_OUTBOUND_PORT || '8791')
let sendChannelEnabled = process.env.MAHJONG_WECHATY_SEND_ENABLED
  ? truthy(process.env.MAHJONG_WECHATY_SEND_ENABLED)
  : false
let sendChannelUpdatedAt = nowText()
let autoSendReplyEnabled = truthy(process.env.MAHJONG_WECHATY_AUTO_SEND_REPLY)
let autoSendReplyUpdatedAt = nowText()
const contactAliases = parseContactAliases(process.env.MAHJONG_WECHATY_CONTACT_ALIASES || '')
const forwardSelfMessages = process.env.MAHJONG_WECHATY_FORWARD_SELF
  ? truthy(process.env.MAHJONG_WECHATY_FORWARD_SELF)
  : true
const knownContacts = new Map()
const knownRooms = new Map()
const recentOutboundSignatures = new Map()
const deliveredReplyMessageIds = loadDeliveryLedger()
const blockedCustomerVisibleTerms = [
  'agent',
  'ai',
  'llm',
  'prompt',
  'trace',
  'idempotency',
  'tool',
  'runtime',
  'debug',
  'wechaty',
  'bridge',
  '智能助手',
  '大模型',
  '模型',
  '机器人',
  '系统',
  '系统账号',
  '后台',
  '工具',
  '提示词',
  '审批',
  '草稿',
  '待审批',
  '日志',
  '数据库',
  '幂等',
  '测试通道',
  '测试账号',
  '个人微信测试',
]

/** 将环境变量中的常见真值写法归一化为布尔值。 */
function truthy(value) {
  return ['1', 'true', 'yes', 'on'].includes(String(value || '').trim().toLowerCase())
}

/** 生成供本地日志使用的秒级时间文本。 */
function nowText() {
  const date = new Date()
  const pad = (item) => String(item).padStart(2, '0')
  return [
    date.getFullYear(),
    '-',
    pad(date.getMonth() + 1),
    '-',
    pad(date.getDate()),
    ' ',
    pad(date.getHours()),
    ':',
    pad(date.getMinutes()),
    ':',
    pad(date.getSeconds()),
  ].join('')
}

/** 确保可靠投递账本和待重放目录存在。 */
function ensureLocalDataPaths() {
  fs.mkdirSync(LOCAL_DATA_DIR, { recursive: true })
  fs.mkdirSync(INGRESS_SPOOL_DIR, { recursive: true })
}

/** 从环境变量或本地密钥文件读取 Runtime 接口令牌。 */
function loadAgentApiToken() {
  const configured = cleanText(process.env.MAHJONG_AGENT_API_TOKEN || process.env.MAHJONG_WECHATY_AGENT_API_TOKEN || '')
  if (configured) {
    return configured
  }
  const tokenPath = process.env.MAHJONG_AGENT_API_TOKEN_PATH || path.join(LOCAL_DATA_DIR, 'runtime_api_token')
  try {
    return cleanText(fs.readFileSync(tokenPath, 'utf8'))
  } catch {
    return ''
  }
}

/** 恢复已成功外发的业务消息标识，避免进程重启后重复发送。 */
function loadDeliveryLedger() {
  const delivered = new Set()
  try {
    const lines = fs.readFileSync(DELIVERY_LEDGER_PATH, 'utf8').split(/\r?\n/)
    for (const line of lines) {
      if (!line.trim()) continue
      const record = JSON.parse(line)
      const sourceMessageId = cleanText(record.source_message_id)
      if (sourceMessageId) delivered.add(sourceMessageId)
    }
  } catch {
    // 首次启动没有投递账本属于正常状态。
  }
  return delivered
}

/** 以追加日志方式持久化已投递标识，再同步更新内存去重集合。 */
function recordDeliveredReply(sourceMessageId, traceId) {
  const key = cleanText(sourceMessageId)
  if (!key || deliveredReplyMessageIds.has(key)) {
    return
  }
  ensureLocalDataPaths()
  fs.appendFileSync(
    DELIVERY_LEDGER_PATH,
    `${JSON.stringify({ source_message_id: key, trace_id: cleanText(traceId), delivered_at: nowText() })}\n`,
    'utf8',
  )
  deliveredReplyMessageIds.add(key)
}

/** 将 WeChaty 对象中的标量值转换为可序列化类型。 */
function primitive(value) {
  if (value === null || value === undefined) {
    return value
  }
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value
  }
  return String(value)
}

/** 移除控制字符并统一文本首尾空白。 */
function cleanText(value) {
  const text = String(value || '').replace(/[\u0000-\u001f\u007f]/g, '').trim()
  return text
}

/** 返回客户可见文本中命中的内部实现词，作为模型审查后的确定性兜底。 */
function customerVisibleTextViolations(text) {
  const clean = cleanText(text)
  const lower = clean.toLowerCase()
  const hits = []
  for (const term of blockedCustomerVisibleTerms) {
    const needle = term.toLowerCase()
    if (lower.includes(needle)) {
      hits.push(term)
    }
  }
  return [...new Set(hits)]
}

/** 合并本地联系人别名和环境变量配置，源代码中不保存真实备注。 */
function parseContactAliases(raw) {
  const aliases = new Map()
  try {
    const localAliases = JSON.parse(fs.readFileSync(CONTACT_ALIASES_PATH, 'utf8'))
    if (localAliases && typeof localAliases === 'object' && !Array.isArray(localAliases)) {
      for (const [alias, target] of Object.entries(localAliases)) {
        const key = contactKey(alias)
        const normalizedTarget = cleanText(target)
        if (key && normalizedTarget) aliases.set(key, normalizedTarget)
      }
    }
  } catch {
    // 本地别名为可选配置，并被有意排除在版本控制之外。
  }
  for (const part of String(raw || '').split(/[,\n]/)) {
    const item = part.trim()
    if (!item || !item.includes('=')) {
      continue
    }
    const [alias, ...targetParts] = item.split('=')
    const key = contactKey(alias)
    const target = cleanText(targetParts.join('='))
    if (key && target) {
      aliases.set(key, target)
    }
  }
  return aliases
}

/** 生成联系人和群名称的大小写无关查找键。 */
function contactKey(value) {
  return cleanText(value).toLowerCase()
}

/** 缓存已观察到的联系人，并为 ID、昵称、备注和微信号建立别名索引。 */
function rememberContact(contact) {
  if (!contact || !contact.id) {
    return
  }
  const snapshot = {
    id: cleanText(contact.id),
    name: cleanText(contact.name),
    alias: cleanText(contact.alias),
    weixin: cleanText(contact.payload?.weixin || ''),
  }
  knownContacts.set(snapshot.id, snapshot)
  for (const value of [snapshot.name, snapshot.alias, snapshot.weixin]) {
    const key = contactKey(value)
    if (key) {
      knownContacts.set(key, snapshot)
    }
  }
}

/** 返回按联系人 ID 去重后的可观测联系人快照。 */
function publicKnownContacts() {
  const seen = new Set()
  const contacts = []
  for (const item of knownContacts.values()) {
    if (!item?.id || seen.has(item.id)) {
      continue
    }
    seen.add(item.id)
    contacts.push(item)
  }
  return contacts
}

/** 缓存已观察到的群，并同时支持群 ID 和群名称查找。 */
function rememberRoom(room) {
  if (!room || !room.id) {
    return
  }
  const snapshot = {
    id: cleanText(room.id),
    topic: cleanText(room.topic),
  }
  knownRooms.set(snapshot.id, snapshot)
  const topicKey = contactKey(snapshot.topic)
  if (topicKey) {
    knownRooms.set(topicKey, snapshot)
  }
}

/** 返回按群 ID 去重后的可观测群列表。 */
function publicKnownRooms() {
  const seen = new Set()
  const rooms = []
  for (const item of knownRooms.values()) {
    if (!item?.id || seen.has(item.id)) {
      continue
    }
    seen.add(item.id)
    rooms.push(item)
  }
  return rooms
}

/** 将会话与文本组合成短期回声检测签名。 */
function outboundSignature(conversationId, text) {
  return `${conversationId || '-'}\n${cleanText(text)}`
}

/** 清理已经超过回声检测窗口的外发签名。 */
function pruneOutboundSignatures() {
  const now = Date.now()
  for (const [key, record] of recentOutboundSignatures.entries()) {
    const expiresAt = typeof record === 'number' ? record : record?.expiresAt
    if (expiresAt <= now) {
      recentOutboundSignatures.delete(key)
    }
  }
}

/** 在真正外发前登记签名，防止微信回流消息再次进入 Agent。 */
function markOutboundSignature(conversationId, text, reference = {}) {
  const clean = cleanText(text)
  if (!clean) {
    return
  }
  pruneOutboundSignatures()
  recentOutboundSignatures.set(outboundSignature(conversationId, clean), {
    expiresAt: Date.now() + 60_000,
    reference: jsonable(reference || {}),
  })
}

/** 识别并消费本进程刚发送后由 WeChaty 回流的自发消息记录。 */
function recentOutboundEchoRecord(payload) {
  if (!payload?.self_message) {
    return null
  }
  pruneOutboundSignatures()
  const key = outboundSignature(payload.conversation_id, payload.text || payload.raw_text || '')
  const record = recentOutboundSignatures.get(key) || null
  if (record) {
    recentOutboundSignatures.delete(key)
  }
  return record
}

/** 判断消息是否为近期外发回声；调用时同时消费对应签名。 */
function isRecentOutboundEcho(payload) {
  return Boolean(recentOutboundEchoRecord(payload))
}

/** 有界递归地把 Puppet 原始对象转换为 JSON 安全结构。 */
function jsonable(value, depth = 0) {
  if (depth > 4) {
    return String(value)
  }
  if (value === null || value === undefined) {
    return value
  }
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value
  }
  if (Array.isArray(value)) {
    return value.map((item) => jsonable(item, depth + 1))
  }
  if (typeof value === 'object') {
    const data = {}
    for (const [key, item] of Object.entries(value)) {
      data[key] = jsonable(item, depth + 1)
    }
    return data
  }
  return String(value)
}

/** 拼接原始载荷字段路径，供引用和媒体候选观测使用。 */
function candidatePath(parent, key) {
  return parent ? `${parent}.${key}` : key
}

/** 在受限深度和数量内收集满足条件的原始字段，避免日志无限膨胀。 */
function collectRawCandidates(value, matcher, path = '', depth = 0, results = []) {
  if (depth > 4 || results.length >= 20 || value === null || value === undefined) {
    return results
  }
  if (Array.isArray(value)) {
    value.slice(0, 20).forEach((item, index) => collectRawCandidates(item, matcher, `${path}[${index}]`, depth + 1, results))
    return results
  }
  if (typeof value !== 'object') {
    return results
  }
  for (const [key, item] of Object.entries(value)) {
    const currentPath = candidatePath(path, key)
    if (matcher(key, currentPath, item)) {
      results.push({ path: currentPath, value: jsonable(item) })
      if (results.length >= 20) {
        return results
      }
    }
    collectRawCandidates(item, matcher, currentPath, depth + 1, results)
    if (results.length >= 20) {
      return results
    }
  }
  return results
}

/** 保存不同 Puppet 实现中的引用、媒体和可用方法线索，便于后续兼容分析。 */
function rawObservation(message, rawPayload, text, type) {
  const quoteMatcher = (key, path, item) =>
    /(^|[._-])(quote|quoted|refer|reference|reply)([._-]|$)/i.test(path) ||
    /quote|quoted|refer|reference|reply/i.test(key) ||
    (typeof item === 'string' && /<\s*refermsg\b/i.test(item))
  const mediaMatcher = (key, path) =>
    /(^|[._-])(file|filename|media|image|voice|audio|video|thumb|url|cdn|mime|size|duration)([._-]|$)/i.test(path) ||
    /file|filename|media|image|voice|audio|video|thumb|url|cdn|mime|size|duration/i.test(key)
  return {
    message_constructor: message?.constructor?.name || '',
    message_methods: [
      'text',
      'type',
      'self',
      'room',
      'talker',
      'listener',
      'toFileBox',
      'mentionList',
      'mentionSelf',
      'date',
    ].filter((name) => typeof message?.[name] === 'function'),
    raw_payload_keys: Object.keys(rawPayload || {}).sort(),
    text_empty: !cleanText(text),
    message_type: primitive(type),
    quote_candidates: collectRawCandidates(rawPayload || {}, quoteMatcher),
    media_candidates: collectRawCandidates(rawPayload || {}, mediaMatcher),
  }
}

/**
 * 隔离不同 Puppet 实现的能力差异。
 *
 * 微信适配器并不保证每个方法都存在或返回相同类型，因此采集原始消息时不能让某个
 * 非关键字段读取失败中断整条入站链路。这里把异常转为可审计数据，交由上层决定是否使用。
 */
async function safeCall(label, fn) {
  try {
    const value = await fn()
    return primitive(value)
  } catch (error) {
    return { error: `${label}: ${error?.message || String(error)}` }
  }
}

/** 安全读取 Contact、Room 等复杂对象，同时保留错误原因供原始日志排查。 */
async function safeObject(label, fn) {
  try {
    return await fn()
  } catch (error) {
    return { error: `${label}: ${error?.message || String(error)}` }
  }
}

/**
 * 将 WeChaty/Puppet 消息归一化为 Agent 通道契约。
 *
 * 归一化结果同时保留稳定字段和原始载荷：稳定字段供路由与会话隔离使用，原始载荷供
 * 新消息类型、引用消息和不同 Puppet 兼容性分析使用。私聊会话按联系人隔离，群聊按群隔离。
 */
async function buildPayload(message) {
  const rawPayload = message.payload || {}
  const room = await safeObject('room', () => message.room())
  const talker = await safeObject('talker', () => message.talker())
  const listener = await safeObject('listener', () => message.listener())
  const text = await safeCall('text', () => message.text())
  const type = await safeCall('type', () => message.type())
  const id = primitive(message.id || rawPayload.id || rawPayload.filename || '')

  let roomPayload = null
  if (room && typeof room === 'object' && !room.error) {
    await safeCall('room.ready', () => room.ready?.())
    roomPayload = {
      id: primitive(room.id),
      topic: cleanText(await safeCall('room.topic', () => room.topic())),
      payload: jsonable(room.payload || {}),
    }
    rememberRoom(roomPayload)
  }

  let talkerPayload = null
  if (talker && typeof talker === 'object' && !talker.error) {
    await safeCall('talker.ready', () => talker.ready?.())
    talkerPayload = {
      id: primitive(talker.id),
      name: cleanText(await safeCall('talker.name', () => talker.name())),
      alias: cleanText(await safeCall('talker.alias', () => talker.alias())),
      payload: jsonable(talker.payload || {}),
    }
  }

  let listenerPayload = null
  if (listener && typeof listener === 'object' && !listener.error) {
    await safeCall('listener.ready', () => listener.ready?.())
    listenerPayload = {
      id: primitive(listener.id),
      name: cleanText(await safeCall('listener.name', () => listener.name())),
      payload: jsonable(listener.payload || {}),
    }
  }

  const roomId = roomPayload?.id || primitive(rawPayload.roomId || rawPayload.room?.id || '')
  const senderId = talkerPayload?.id || primitive(rawPayload.talkerId || rawPayload.fromId || '')
  const senderName =
    talkerPayload?.name ||
    cleanText(rawPayload.talkerName || rawPayload.fromName || rawPayload.senderName || '')
  if (!talkerPayload && senderId) {
    talkerPayload = {
      id: senderId,
      name: senderName,
      alias: '',
      payload: {},
    }
  }
  if (!listenerPayload && rawPayload.listenerId) {
    listenerPayload = {
      id: primitive(rawPayload.listenerId),
      name: '',
      payload: {},
    }
  }
  const conversationId = roomId ? `wechaty:room:${roomId}` : `wechaty:contact:${senderId}`

  const payload = {
    captured_at: nowText(),
    channel: 'wechaty',
    platform_name: 'wechaty',
    puppet: process.env.WECHATY_PUPPET || '',
    conversation_id: conversationId,
    message_id: id,
    source_message_id: id,
    message_type: primitive(type),
    is_room: Boolean(roomId),
    room: roomPayload,
    sender_id: senderId,
    sender_name: senderName,
    talker: talkerPayload,
    listener: listenerPayload,
    text: typeof text === 'string' ? text : '',
    raw_text: text,
    self_message: await safeCall('self', () => message.self()),
    payload: rawPayload,
    raw_observation: rawObservation(message, rawPayload, text, type),
  }
  rememberContact(payload.talker)
  rememberContact(payload.listener)
  return payload
}

/** 使用统一鉴权头向 Agent API 发送 JSON，并把非 2xx 响应转换为带状态码的异常。 */
async function postJson(url, payload) {
  const headers = { 'Content-Type': 'application/json; charset=utf-8' }
  if (agentApiToken) {
    headers['X-Mahjong-Agent-Token'] = agentApiToken
  }
  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  })
  const body = await response.text()
  let parsed = null
  try {
    parsed = JSON.parse(body)
  } catch {
    parsed = { raw_response: body }
  }
  if (!response.ok) {
    const error = new Error(`HTTP ${response.status}: ${body}`)
    error.status = response.status
    throw error
  }
  return parsed
}

/** 仅把网络错误、限流和临时服务端错误判定为可重试，业务拒绝不会被盲目重放。 */
function retriableForwardError(error) {
  const status = Number(error?.status || 0)
  return !status || [408, 409, 425, 429, 500, 502, 503, 504].includes(status)
}

/** 采用指数退避和随机抖动重试入站转发，降低 Agent 服务短暂抖动造成的消息丢失。 */
async function postJsonWithRetry(url, payload, attempts = 4) {
  let lastError = null
  for (let attempt = 1; attempt <= Math.max(1, attempts); attempt += 1) {
    try {
      return await postJson(url, payload)
    } catch (error) {
      lastError = error
      if (attempt >= attempts || !retriableForwardError(error)) {
        throw error
      }
      const delayMs = Math.min(5000, 250 * 2 ** (attempt - 1)) + Math.floor(Math.random() * 150)
      await new Promise((resolve) => setTimeout(resolve, delayMs))
    }
  }
  throw lastError || new Error('forward failed')
}

/** 根据源消息 ID 生成安全且稳定的重放文件名，避免同一消息产生无界临时文件。 */
function spoolPathForPayload(payload) {
  const sourceId = cleanText(payload?.source_message_id || payload?.message_id || `${Date.now()}`)
  const safeId = sourceId.replace(/[^a-zA-Z0-9_.-]/g, '_').slice(0, 180) || `${Date.now()}`
  return path.join(INGRESS_SPOOL_DIR, `${safeId}.json`)
}

/**
 * 原子持久化转发失败的原始消息。
 *
 * 先写临时文件再重命名，避免进程崩溃留下半截 JSON；文件只有在 Agent API 成功接收后才删除。
 */
function persistIngressForReplay(payload, error) {
  ensureLocalDataPaths()
  const target = spoolPathForPayload(payload)
  const temporary = `${target}.${process.pid}.tmp`
  fs.writeFileSync(
    temporary,
    JSON.stringify({ payload, attempts: 0, last_error: error?.message || String(error), queued_at: nowText() }),
    'utf8',
  )
  fs.renameSync(temporary, target)
  return target
}

/** 按文件顺序重放失败入站消息；遇到持续故障立即停止，避免形成重试风暴和消息乱序。 */
async function replayIngressSpool() {
  ensureLocalDataPaths()
  const files = fs.readdirSync(INGRESS_SPOOL_DIR).filter((item) => item.endsWith('.json')).sort()
  for (const filename of files) {
    const fullPath = path.join(INGRESS_SPOOL_DIR, filename)
    try {
      const record = JSON.parse(fs.readFileSync(fullPath, 'utf8'))
      const result = await postJsonWithRetry(endpoint, record.payload, 3)
      await maybeAutoSendReply(record.payload, result)
      fs.unlinkSync(fullPath)
      console.log(`[${nowText()}] replayed ingress message_id=${record.payload?.message_id || '-'} trace_id=${result.trace_id || '-'}`)
    } catch (error) {
      console.error(`[${nowText()}] ingress replay deferred file=${filename}: ${error?.message || String(error)}`)
      break
    }
  }
}

/**
 * 在通道开关允许时外发 Agent 的最终回复。
 *
 * 外发前同时执行源消息幂等检查和客户可见关键词兜底；只有成功发送后才记录交付状态，
 * 并登记回声签名，防止微信回流消息再次进入 Agent 形成自我对话。
 */
async function maybeAutoSendReply(payload, result, message = null) {
  const finalReply = cleanText(result?.route_result?.agent_result?.final_reply)
  if (!finalReply || !sendChannelEnabled || !autoSendReplyEnabled) {
    return false
  }
  const sourceMessageId = cleanText(payload?.source_message_id || payload?.message_id)
  if (sourceMessageId && deliveredReplyMessageIds.has(sourceMessageId)) {
    console.log(`[${nowText()}] skipped duplicate auto-reply source_message_id=${sourceMessageId}`)
    return false
  }
  const violations = customerVisibleTextViolations(finalReply)
  if (violations.length) {
    console.log(`[${nowText()}] skipped auto-send because reply contains internal implementation terms trace_id=${result.trace_id || '-'}`)
    return false
  }
  if (message) {
    await message.say(finalReply)
  } else {
    await sendContactText(payload.sender_id, finalReply, {
      source: 'wechaty_ingress_replay',
      source_trace_id: result.trace_id || '',
      source_message_id: sourceMessageId,
    })
  }
  recordDeliveredReply(sourceMessageId, result.trace_id || '')
  markOutboundSignature(payload.conversation_id, finalReply, {
    source: 'wechaty_auto_reply',
    source_trace_id: result.trace_id || '',
  })
  console.log(`[${nowText()}] auto-sent reply trace_id=${result.trace_id || '-'} text=${finalReply}`)
  return true
}

/** 判断一次外发是否携带足以回连业务对象或草稿的稳定引用锚点。 */
function hasReferenceAnchor(reference) {
  return Boolean(
    cleanText(reference?.source_message_id || reference?.sourceMessageId || reference?.draft_id || reference?.draftId || '') ||
      (cleanText(reference?.business_ref_type || reference?.businessRefType || '') &&
        cleanText(reference?.business_ref_id || reference?.businessRefId || ''))
  )
}

/** 将微信平台返回的消息 ID 回写 Agent，建立“业务草稿 -> 实际外发消息”的可追溯关系。 */
async function postDeliveredMessageReference(payload, reference) {
  if (!payload?.message_id || !hasReferenceAnchor(reference)) {
    return { ok: false, skipped: 'missing_reference_anchor' }
  }
  const body = {
    conversation_id: payload.conversation_id,
    platform_message_id: payload.message_id,
    source_message_id:
      reference.source_message_id || reference.sourceMessageId || reference.draft_id || reference.draftId || '',
    business_ref_type: reference.business_ref_type || reference.businessRefType || '',
    business_ref_id: reference.business_ref_id || reference.businessRefId || '',
    channel: 'wechaty',
    text: payload.text || payload.raw_text || '',
    metadata: {
      source: 'wechaty_outbound_echo',
      outbound_source: reference.source || reference.outbound_source || '',
      recipient_id: reference.recipient_id || reference.recipientId || '',
      recipient_name: reference.recipient_name || reference.recipientName || '',
    },
  }
  try {
    return await postJson(referenceEndpoint, body)
  } catch (error) {
    console.error(`[${nowText()}] message reference link failed: ${error?.message || String(error)}`)
    return { ok: false, error: error?.message || String(error) }
  }
}

const bot = WechatyBuilder.build({ name: botName })

/**
 * 将外部目标解析为可发送的微信联系人。
 *
 * 依次尝试微信内部 ID、本地缓存、别名/名称/微信号和全量联系人扫描；缓存既减少查询开销，
 * 也弥补不同 Puppet 对 Contact.find 查询字段支持不一致的问题。
 */
async function resolveContact(target) {
  const rawRequested = cleanText(target)
  const aliasTarget = contactAliases.get(contactKey(rawRequested))
  const requested = cleanText(aliasTarget || rawRequested)
  const requestedKey = contactKey(requested)
  if (!requested) {
    throw new Error('missing target contact')
  }
  if (requested.startsWith('@')) {
    const contact = bot.Contact.load(requested)
    await safeCall('contact.ready', () => contact.ready?.())
    return contact
  }
  const known = knownContacts.get(requested) || knownContacts.get(contactKey(requested))
  if (known?.id) {
    const contact = bot.Contact.load(known.id)
    await safeCall('contact.ready', () => contact.ready?.())
    return contact
  }
  for (const query of [{ alias: requested }, { name: requested }, { weixin: requested }]) {
    try {
      const contact = await bot.Contact.find(query)
      if (contact) {
        await safeCall('contact.ready', () => contact.ready?.())
        rememberContact({
          id: primitive(contact.id),
          name: cleanText(await safeCall('contact.name', () => contact.name())),
          alias: cleanText(await safeCall('contact.alias', () => contact.alias())),
          payload: jsonable(contact.payload || {}),
        })
        return contact
      }
    } catch {
      // 不同 Puppet 支持的 Contact.find 查询字段不同，失败后继续尝试下一种稳定标识。
    }
  }
  try {
    const contacts = await bot.Contact.findAll()
    for (const contact of contacts || []) {
      await safeCall('contact.ready', () => contact.ready?.())
      const snapshot = {
        id: primitive(contact.id),
        name: cleanText(await safeCall('contact.name', () => contact.name())),
        alias: cleanText(await safeCall('contact.alias', () => contact.alias())),
        payload: jsonable(contact.payload || {}),
      }
      rememberContact(snapshot)
      const values = [
        snapshot.id,
        snapshot.name,
        snapshot.alias,
        snapshot.payload?.weixin,
        snapshot.payload?.name,
        snapshot.payload?.alias,
      ]
      if (values.some((item) => contactKey(item) === requestedKey)) {
        return contact
      }
    }
  } catch {
    // 部分 Puppet 不开放完整联系人列表，统一保留下方更清晰的“联系人不存在”错误。
  }
  throw new Error(`contact not found: ${requested}`)
}

/** 按内部 ID、缓存、群名和全量群列表解析群对象，屏蔽不同 Puppet 的查询能力差异。 */
async function resolveRoom(target) {
  const requested = cleanText(target)
  const requestedKey = contactKey(requested)
  if (!requested) {
    throw new Error('missing target room')
  }
  if (requested.startsWith('@@')) {
    const room = bot.Room.load(requested)
    await safeCall('room.ready', () => room.ready?.())
    return room
  }
  const known = knownRooms.get(requested) || knownRooms.get(requestedKey)
  if (known?.id) {
    const room = bot.Room.load(known.id)
    await safeCall('room.ready', () => room.ready?.())
    return room
  }
  for (const query of [{ id: requested }, { topic: requested }]) {
    try {
      const room = await bot.Room.find(query)
      if (room) {
        await safeCall('room.ready', () => room.ready?.())
        rememberRoom({
          id: primitive(room.id),
          topic: cleanText(await safeCall('room.topic', () => room.topic())),
        })
        return room
      }
    } catch {
      // 不同 Puppet 支持的 Room.find 查询字段不同，失败后继续尝试其他定位方式。
    }
  }
  try {
    const rooms = await bot.Room.findAll()
    for (const room of rooms || []) {
      await safeCall('room.ready', () => room.ready?.())
      const snapshot = {
        id: primitive(room.id),
        topic: cleanText(await safeCall('room.topic', () => room.topic())),
      }
      rememberRoom(snapshot)
      if ([snapshot.id, snapshot.topic].some((item) => contactKey(item) === requestedKey)) {
        return room
      }
    }
  } catch {
    // 部分 Puppet 不开放完整群列表，此时返回统一的“群不存在”错误。
  }
  throw new Error(`room not found: ${requested}`)
}

/** 从草稿或源消息提取后端权威交付键，用于阻止同一业务消息重复发送。 */
function outboundDeliveryKey(options = {}) {
  return cleanText(
    options.draft_id || options.draftId || options.source_message_id || options.sourceMessageId || '',
  )
}

/** 组装外发引用元数据，供微信回声到达后回连草稿、局或其他业务对象。 */
function outboundReference(options, recipientId, recipientName, source) {
  return {
    source,
    source_message_id:
      options.source_message_id || options.sourceMessageId || options.draft_id || options.draftId || '',
    business_ref_type: options.business_ref_type || options.businessRefType || '',
    business_ref_id: options.business_ref_id || options.businessRefId || '',
    recipient_id: recipientId,
    recipient_name: recipientName,
  }
}

/**
 * 执行外发通道的最后一道确定性校验。
 *
 * 模型审查负责语义风险，此处只处理必须稳定执行的开关、空文本和内部实现关键词禁发规则。
 */
function validateOutboundText(text) {
  if (!sendChannelEnabled) {
    throw new Error('wechat send channel is paused')
  }
  const finalText = cleanText(text)
  if (!finalText) {
    throw new Error('missing text')
  }
  if (customerVisibleTextViolations(finalText).length) {
    throw new Error('customer visible text contains internal implementation terms')
  }
  return finalText
}

/** 兼容不同 Puppet 的发送结果结构，提取可用于审计和引用回连的平台消息 ID。 */
function sentMessageId(sent) {
  return cleanText(sent?.id || sent?.payload?.id || sent?.payload?.msgId || '')
}

/** 向联系人发送文本，并在发送前后维护业务幂等、回声抑制和交付审计状态。 */
async function sendContactText(target, text, options = {}) {
  const finalText = validateOutboundText(text)
  const deliveryKey = outboundDeliveryKey(options)
  if (deliveryKey && deliveredReplyMessageIds.has(deliveryKey)) {
    return { ok: true, deduplicated: true, to: target, text: finalText }
  }
  const contact = await resolveContact(target)
  const contactId = primitive(contact.id)
  const conversationId = `wechaty:contact:${contactId}`
  markOutboundSignature(
    conversationId,
    finalText,
    outboundReference(options, contactId, target, 'wechaty_send'),
  )
  let sent = null
  try {
    sent = await contact.say(finalText)
  } catch (error) {
    recentOutboundSignatures.delete(outboundSignature(conversationId, finalText))
    throw error
  }
  recordDeliveredReply(deliveryKey, options.source_trace_id || options.sourceTraceId || '')
  return {
    ok: true,
    to: target,
    contact_id: contactId,
    message_id: sentMessageId(sent),
    text: finalText,
    reference_pending: hasReferenceAnchor(options),
  }
}

/** 向群聊发送文本；其交付和回声抑制语义与私聊外发保持一致。 */
async function sendRoomText(target, text, options = {}) {
  const finalText = validateOutboundText(text)
  const deliveryKey = outboundDeliveryKey(options)
  if (deliveryKey && deliveredReplyMessageIds.has(deliveryKey)) {
    return { ok: true, deduplicated: true, to: target, target_type: 'room', text: finalText }
  }
  const room = await resolveRoom(target)
  const roomId = primitive(room.id)
  const topic = cleanText(await safeCall('room.topic', () => room.topic()))
  rememberRoom({ id: roomId, topic })
  const conversationId = `wechaty:room:${roomId}`
  markOutboundSignature(
    conversationId,
    finalText,
    outboundReference(options, roomId, topic || target, 'wechaty_room_send'),
  )
  let sent = null
  try {
    sent = await room.say(finalText)
  } catch (error) {
    recentOutboundSignatures.delete(outboundSignature(conversationId, finalText))
    throw error
  }
  recordDeliveredReply(deliveryKey, options.source_trace_id || options.sourceTraceId || '')
  return {
    ok: true,
    to: target,
    target_type: 'room',
    room_id: roomId,
    room_topic: topic,
    message_id: sentMessageId(sent),
    text: finalText,
    reference_pending: hasReferenceAnchor(options),
  }
}

/** 统一输出本地桥接控制接口的 JSON 与跨域响应头。 */
function sendJson(response, statusCode, payload) {
  const body = JSON.stringify(payload, null, 2)
  response.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': 'http://127.0.0.1:8790',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  })
  response.end(body)
}

/** 完整读取请求体后再解析 JSON，避免分块传输导致只读取到部分参数。 */
async function readJsonRequest(request) {
  const chunks = []
  for await (const chunk of request) {
    chunks.push(chunk)
  }
  if (!chunks.length) {
    return {}
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'))
}

/**
 * 启动仅监听本机的桥接控制面。
 *
 * 控制面提供健康检查、联系人/群发现、外发开关和统一发送入口；绑定 127.0.0.1，
 * 不直接暴露到局域网或公网，业务权限与客户可见审查仍由 Agent 后端负责。
 */
function startOutboundServer() {
  const server = http.createServer(async (request, response) => {
    try {
      if (request.method === 'OPTIONS') {
        sendJson(response, 200, { ok: true })
        return
      }
      if (request.method === 'GET' && request.url === '/health') {
        sendJson(response, 200, {
          ok: true,
          bot_name: botName,
          outbound_enabled: outboundEnabled,
          send_channel_enabled: sendChannelEnabled,
          send_channel_updated_at: sendChannelUpdatedAt,
          auto_send_reply: autoSendReplyEnabled,
          auto_send_reply_updated_at: autoSendReplyUpdatedAt,
          known_contact_count: publicKnownContacts().length,
          known_room_count: publicKnownRooms().length,
          contact_alias_count: contactAliases.size,
        })
        return
      }
      if (request.method === 'GET' && request.url === '/send-channel') {
        sendJson(response, 200, {
          ok: true,
          send_channel_enabled: sendChannelEnabled,
          updated_at: sendChannelUpdatedAt,
        })
        return
      }
      if (request.method === 'POST' && request.url === '/send-channel') {
        const payload = await readJsonRequest(request)
        if (typeof payload.enabled !== 'boolean') {
          sendJson(response, 400, { ok: false, error: 'enabled must be boolean' })
          return
        }
        sendChannelEnabled = payload.enabled
        sendChannelUpdatedAt = nowText()
        console.log(`[${nowText()}] send_channel_enabled=${sendChannelEnabled}`)
        sendJson(response, 200, {
          ok: true,
          send_channel_enabled: sendChannelEnabled,
          updated_at: sendChannelUpdatedAt,
        })
        return
      }
      if (request.method === 'GET' && request.url === '/auto-send') {
        sendJson(response, 200, {
          ok: true,
          auto_send_reply: autoSendReplyEnabled,
          updated_at: autoSendReplyUpdatedAt,
        })
        return
      }
      if (request.method === 'POST' && request.url === '/auto-send') {
        const payload = await readJsonRequest(request)
        if (typeof payload.enabled !== 'boolean') {
          sendJson(response, 400, { ok: false, error: 'enabled must be boolean' })
          return
        }
        autoSendReplyEnabled = payload.enabled
        autoSendReplyUpdatedAt = nowText()
        console.log(`[${nowText()}] auto_send_reply=${autoSendReplyEnabled}`)
        sendJson(response, 200, {
          ok: true,
          auto_send_reply: autoSendReplyEnabled,
          updated_at: autoSendReplyUpdatedAt,
        })
        return
      }
      if (request.method === 'GET' && request.url === '/contacts') {
        sendJson(response, 200, { ok: true, contacts: publicKnownContacts() })
        return
      }
      if (request.method === 'GET' && request.url === '/rooms') {
        sendJson(response, 200, { ok: true, rooms: publicKnownRooms() })
        return
      }
      if (request.method === 'POST' && request.url === '/send') {
        const payload = await readJsonRequest(request)
        const targetType = cleanText(payload.target_type || payload.targetType || '').toLowerCase()
        const result = targetType === 'room' || payload.room_id || payload.roomId
          ? await sendRoomText(payload.to || payload.room_id || payload.roomId, payload.text, payload)
          : await sendContactText(payload.to || payload.contact_id || payload.weixin, payload.text, payload)
        sendJson(response, 200, result)
        return
      }
      sendJson(response, 404, { ok: false, error: 'not found' })
    } catch (error) {
      sendJson(response, 500, {
        ok: false,
        error: error?.message || String(error),
        known_contacts: publicKnownContacts().slice(0, 20),
        known_rooms: publicKnownRooms().slice(0, 20),
      })
    }
  })
  server.listen(outboundPort, '127.0.0.1', () => {
    console.log(`[${nowText()}] outbound server=http://127.0.0.1:${outboundPort}`)
  })
}

bot.on('scan', (qrcode, status) => {
  console.log(`[${nowText()}] scan status=${status}`)
  console.log(`https://wechaty.js.org/qrcode/${encodeURIComponent(qrcode)}`)
  qrcodeTerminal.generate(qrcode, { small: true })
})

bot.on('login', (user) => {
  console.log(`[${nowText()}] login: ${user}`)
  void replayIngressSpool()
})

bot.on('logout', (user) => {
  console.log(`[${nowText()}] logout: ${user}`)
})

bot.on('error', (error) => {
  console.error(`[${nowText()}] error:`, error)
})

bot.on('message', async (message) => {
  if (!forwardSelfMessages && message.self()) {
    console.log(`[${nowText()}] skipped self message_id=${message.id || message.payload?.id || '-'}`)
    return
  }
  const payload = await buildPayload(message)
  // 外发消息也会作为微信入站事件回流；命中签名后仅补齐平台消息引用，不再进入 Agent。
  const outboundEcho = recentOutboundEchoRecord(payload)
  if (outboundEcho) {
    const linkResult = await postDeliveredMessageReference(payload, outboundEcho.reference || {})
    console.log(
      `[${nowText()}] skipped outbound echo message_id=${payload.message_id || '-'} ` +
        `conversation_id=${payload.conversation_id} reference_linked=${linkResult?.ok ? 'yes' : 'no'}`
    )
    return
  }
  try {
    const result = await postJsonWithRetry(endpoint, payload)
    console.log(
      `[${nowText()}] forwarded message_id=${payload.message_id || '-'} ` +
        `conversation_id=${payload.conversation_id} trace_id=${result.trace_id || '-'}`
    )
    await maybeAutoSendReply(payload, result, message)
    const finalReply = cleanText(result?.route_result?.agent_result?.final_reply)
    if (!sendChannelEnabled && autoSendReplyEnabled && finalReply) {
      console.log(`[${nowText()}] skipped auto-send because send channel is paused trace_id=${result.trace_id || '-'}`)
    }
  } catch (error) {
    // API 暂时不可用时先落本地重放队列，保证监听进程不因单条消息失败而丢失后续消息。
    const spoolPath = persistIngressForReplay(payload, error)
    console.error(`[${nowText()}] forward failed and queued for replay path=${spoolPath}: ${error?.message || String(error)}`)
  }
})

process.once('SIGINT', async () => {
  console.log(`[${nowText()}] stopping...`)
  await bot.stop()
  process.exit(0)
})

console.log(`[${nowText()}] starting ${botName}`)
console.log(`[${nowText()}] endpoint=${endpoint}`)
console.log(`[${nowText()}] agent_api_token=${agentApiToken ? 'configured' : 'missing'}`)
console.log(`[${nowText()}] WECHATY_PUPPET=${process.env.WECHATY_PUPPET || '(default)'}`)
console.log(`[${nowText()}] send_channel_enabled=${sendChannelEnabled}`)
console.log(`[${nowText()}] auto_send_reply=${autoSendReplyEnabled}`)
console.log(`[${nowText()}] contact_alias_count=${contactAliases.size}`)

if (outboundEnabled) {
  startOutboundServer()
}

ensureLocalDataPaths()
// 周期重放只负责恢复可靠投递；具体消息幂等仍由 Agent 后端和交付键共同保证。
setInterval(() => void replayIngressSpool(), 15_000).unref()
await bot.start()
