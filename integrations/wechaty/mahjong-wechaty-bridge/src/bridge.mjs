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

function truthy(value) {
  return ['1', 'true', 'yes', 'on'].includes(String(value || '').trim().toLowerCase())
}

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

function ensureLocalDataPaths() {
  fs.mkdirSync(LOCAL_DATA_DIR, { recursive: true })
  fs.mkdirSync(INGRESS_SPOOL_DIR, { recursive: true })
}

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
    // A missing ledger is the normal first-run state.
  }
  return delivered
}

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

function primitive(value) {
  if (value === null || value === undefined) {
    return value
  }
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value
  }
  return String(value)
}

function cleanText(value) {
  const text = String(value || '').replace(/[\u0000-\u001f\u007f]/g, '').trim()
  return text
}

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
    // Local aliases are optional and deliberately excluded from source control.
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

function contactKey(value) {
  return cleanText(value).toLowerCase()
}

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

function outboundSignature(conversationId, text) {
  return `${conversationId || '-'}\n${cleanText(text)}`
}

function pruneOutboundSignatures() {
  const now = Date.now()
  for (const [key, record] of recentOutboundSignatures.entries()) {
    const expiresAt = typeof record === 'number' ? record : record?.expiresAt
    if (expiresAt <= now) {
      recentOutboundSignatures.delete(key)
    }
  }
}

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

function recentOutboundEchoRecord(payload) {
  if (!payload?.self_message) {
    return null
  }
  pruneOutboundSignatures()
  return recentOutboundSignatures.get(outboundSignature(payload.conversation_id, payload.text || payload.raw_text || '')) || null
}

function isRecentOutboundEcho(payload) {
  return Boolean(recentOutboundEchoRecord(payload))
}

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

function candidatePath(parent, key) {
  return parent ? `${parent}.${key}` : key
}

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

async function safeCall(label, fn) {
  try {
    const value = await fn()
    return primitive(value)
  } catch (error) {
    return { error: `${label}: ${error?.message || String(error)}` }
  }
}

async function safeObject(label, fn) {
  try {
    return await fn()
  } catch (error) {
    return { error: `${label}: ${error?.message || String(error)}` }
  }
}

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

function retriableForwardError(error) {
  const status = Number(error?.status || 0)
  return !status || [408, 409, 425, 429, 500, 502, 503, 504].includes(status)
}

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

function spoolPathForPayload(payload) {
  const sourceId = cleanText(payload?.source_message_id || payload?.message_id || `${Date.now()}`)
  const safeId = sourceId.replace(/[^a-zA-Z0-9_.-]/g, '_').slice(0, 180) || `${Date.now()}`
  return path.join(INGRESS_SPOOL_DIR, `${safeId}.json`)
}

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

function hasReferenceAnchor(reference) {
  return Boolean(
    cleanText(reference?.source_message_id || reference?.sourceMessageId || reference?.draft_id || reference?.draftId || '') ||
      (cleanText(reference?.business_ref_type || reference?.businessRefType || '') &&
        cleanText(reference?.business_ref_id || reference?.businessRefId || ''))
  )
}

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
      // Different puppets support different Contact.find query fields.
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
    // Some puppets do not expose a full contact list; keep the clearer not-found error below.
  }
  throw new Error(`contact not found: ${requested}`)
}

async function sendContactText(target, text, options = {}) {
  if (!sendChannelEnabled) {
    throw new Error('wechat send channel is paused')
  }
  const finalText = cleanText(text)
  if (!finalText) {
    throw new Error('missing text')
  }
  const deliveryKey = cleanText(
    options.draft_id || options.draftId || options.source_message_id || options.sourceMessageId || '',
  )
  if (deliveryKey && deliveredReplyMessageIds.has(deliveryKey)) {
    return { ok: true, deduplicated: true, to: target, text: finalText }
  }
  const violations = customerVisibleTextViolations(finalText)
  if (violations.length) {
    throw new Error('customer visible text contains internal implementation terms')
  }
  const contact = await resolveContact(target)
  await contact.say(finalText)
  recordDeliveredReply(deliveryKey, options.source_trace_id || options.sourceTraceId || '')
  const contactId = primitive(contact.id)
  markOutboundSignature(`wechaty:contact:${contactId}`, finalText, {
    source: 'wechaty_send',
    source_message_id:
      options.source_message_id || options.sourceMessageId || options.draft_id || options.draftId || '',
    business_ref_type: options.business_ref_type || options.businessRefType || '',
    business_ref_id: options.business_ref_id || options.businessRefId || '',
    recipient_id: contactId,
    recipient_name: target,
  })
  return {
    ok: true,
    to: target,
    contact_id: contactId,
    text: finalText,
    reference_pending: hasReferenceAnchor(options),
  }
}

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
      if (request.method === 'POST' && request.url === '/send') {
        const payload = await readJsonRequest(request)
        const result = await sendContactText(payload.to || payload.contact_id || payload.weixin, payload.text, payload)
        sendJson(response, 200, result)
        return
      }
      sendJson(response, 404, { ok: false, error: 'not found' })
    } catch (error) {
      sendJson(response, 500, {
        ok: false,
        error: error?.message || String(error),
        known_contacts: publicKnownContacts().slice(0, 20),
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
setInterval(() => void replayIngressSpool(), 15_000).unref()
await bot.start()
