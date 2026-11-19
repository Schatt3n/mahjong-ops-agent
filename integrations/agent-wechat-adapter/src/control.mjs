/**
 * 本地控制面：兼容 wechaty 桥接的外发合同，并提供联系人消歧、定向刷新与开关查询。
 *
 * 只绑定回环地址。`/health` 永远可读；配置了控制 token 时其余路由必须带
 * `X-Mahjong-Agent-Token` 或 `Authorization: Bearer`。Python Runtime 通过
 * `MAHJONG_WECHATY_OUTBOUND_TOKEN` 发送同一个 token。定向联系人刷新和后台回复
 * 必须经过鉴权，未配置控制 token 时使用共用的 Runtime token。
 */
import http from 'node:http'
import { timingSafeEqual } from 'node:crypto'
import { UpstreamError } from './client.mjs'
import { readObservedMedia } from './media.mjs'
import { displayNameOf, isChatroomId, isPersonalContact, sendIdempotencyKey } from './normalize.mjs'
import { filterListedGroups } from './group-directory.mjs'
import { directAutomation, groupAutomation, groupPermissionMode, hasManualRecipientScope, isDirectContactId,
  isGroupPermissionId, permitsRecipientSend, validAutomation } from './contact-policy.mjs'
import { consoleConversation, consoleConversations, consoleConversationScope } from './conversation-console.mjs'
import { recordConversationSend } from './conversation-sends.mjs'

const MAX_TEXT_LENGTH = 10_000

class HttpError extends Error {
  constructor(status, message, details = {}) {
    super(message)
    this.name = 'HttpError'
    this.status = status
    this.details = details
  }
}

/** 常量时间比较 token，避免长度差异泄漏。 */
function tokenMatches(provided, expected) {
  const left = Buffer.from(String(provided || ''))
  const right = Buffer.from(String(expected || ''))
  if (!left.length || left.length !== right.length) return false
  return timingSafeEqual(left, right)
}

/** 从请求头提取控制面 token，兼容专用头和 Bearer。 */
function requestToken(request) {
  const header = request.headers['x-mahjong-agent-token']
  if (typeof header === 'string' && header) return header
  const authorization = String(request.headers.authorization || '')
  return authorization.toLowerCase().startsWith('bearer ') ? authorization.slice(7).trim() : ''
}

/** 以 no-store JSON 响应控制面请求，避免缓存敏感状态。 */
function sendJson(response, status, payload) {
  const body = JSON.stringify(payload, null, 2)
  response.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'Content-Length': Buffer.byteLength(body),
  })
  response.end(body)
}

/** 私有响应只返回已核验的媒体质量，并禁止浏览器缓存或猜测内容类型。 */
function sendMedia(response, media) {
  const headers = {
    'Content-Type': media.contentType,
    'Content-Length': media.bytes.length,
    'Cache-Control': 'private, no-store',
    'X-Content-Type-Options': 'nosniff',
    'X-Mahjong-Media-Variant': media.variant,
    'X-Mahjong-Media-Representation': media.representation,
  }
  if (media.contentType === 'image/x-wxgf') headers['Content-Disposition'] = 'attachment; filename="sticker.wxgf"'
  response.writeHead(200, headers)
  response.end(media.bytes)
}

/** 限长读取 JSON 请求体；超限直接断开，避免把内存打满。 */
async function readJsonBody(request, limitBytes) {
  const chunks = []
  let total = 0
  for await (const chunk of request) {
    total += chunk.length
    if (total > limitBytes) throw new HttpError(413, `request body exceeds ${limitBytes} bytes`)
    chunks.push(chunk)
  }
  if (!chunks.length) return {}
  try {
    const parsed = JSON.parse(Buffer.concat(chunks).toString('utf8'))
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('body must be a JSON object')
    }
    return parsed
  } catch (error) {
    throw new HttpError(400, `invalid JSON body: ${error?.message || error}`)
  }
}

/** 运行时开关；只在内存中保存，进程重启回到配置里的安全默认。 */
export class ControlSwitches {
  constructor({ sendChannelEnabled = false, autoSendReply = false } = {}) {
    this.sendChannelEnabled = Boolean(sendChannelEnabled)
    this.autoSendReply = Boolean(autoSendReply)
    this.sendChannelUpdatedAt = new Date().toISOString()
    this.autoSendReplyUpdatedAt = new Date().toISOString()
  }
}

/** 在联系人/群缓存里按精确 id 查找，优先于名字匹配。 */
function exactTarget(directory, kind, target) {
  if (kind === 'group') {
    const group = directory.groups.find((item) => String(item.id || item.roomId || '') === target)
    return group ? { threadId: target, kind: 'group', displayName: displayNameOf(group, target) } : null
  }
  const contact = directory.contacts.find((item) => String(item.username || item.id || '') === target)
  return isPersonalContact(contact) ? { threadId: target, kind: 'direct', displayName: displayNameOf(contact, target) } : null
}

/**
 * 把用户输入解析成唯一会话。
 *
 * 允许“精确 wxid/群 id”或“唯一完全匹配的名字”；重名、仅前缀命中和模糊命中一律
 * 返回 409 + 候选，绝不替业务系统挑人。已选会话模式只接受当前精确标识，避免旧选择被同名联系人替代。
 */
export async function resolveTarget(directory, rawTarget, rawKind = '', {
  exactOnly = false, allowedRecipients = null, allowUnlistedGroups = false, allowUnlistedDirect = false, excludedIds = null,
} = {}) {
  const target = String(rawTarget || '').trim()
  if (!target) throw new HttpError(400, 'target is required')
  const kindHint = String(rawKind || '').toLowerCase()
  const kind = ['room', 'group'].includes(kindHint) ? 'group' : ['contact', 'direct'].includes(kindHint) ? 'direct' : ''

  if (exactOnly) {
    if (kind === 'direct' && !(await directory.hasExactContact(target))) {
      throw new HttpError(409, 'selected contact is no longer available; choose an exact target again')
    }
    return { threadId: target, kind }
  }

  await directory.refresh()
  if (target.endsWith('@chatroom')) {
    return { threadId: target, kind: 'group', displayName: exactTarget(directory, 'group', target)?.displayName || target }
  }
  const byId = exactTarget(directory, kind || 'direct', target) || (kind ? null : exactTarget(directory, 'group', target))
  if (byId && !excludedIds?.has(byId.threadId) && (!allowedRecipients || byId.kind === 'group' || allowUnlistedDirect ||
      allowedRecipients.some(item => item.thread_id === byId.threadId))) return byId

  const resolved = await directory.resolve(target, { kind, allowedRecipients, allowUnlistedGroups, allowUnlistedDirect, excludedIds })
  if (resolved.recommendedId && !resolved.ambiguous) {
    const candidate = resolved.candidates.find((item) => item.id === resolved.recommendedId)
    return {
      threadId: resolved.recommendedId,
      kind: candidate?.kind === 'group' ? 'group' : 'direct',
      displayName: candidate?.displayName || resolved.recommendedId,
    }
  }
  throw new HttpError(409, 'target is ambiguous or not found; confirm an exact candidate', {
    query: target,
    kind: resolved.kind,
    candidates: resolved.candidates,
    exact_count: resolved.exactCount,
    ambiguous: resolved.ambiguous,
  })
}

/** 从实时外发名单选目标；已选择的会话只按精确标识匹配，名称只用于原有调用。 */
function resolveAllowedTarget(recipients, rawTarget, kind, { exactOnly = false } = {}) {
  const target = String(rawTarget || '').trim()
  if (!target) throw new HttpError(400, 'target is required')
  const eligible = recipients.filter(item =>
    !kind || (isChatroomId(item.thread_id) ? 'group' : 'direct') === kind)
  let recipient = eligible.find(item => item.thread_id === target)
  if (!recipient && !exactOnly) {
    const named = eligible.filter(item => [item.label, ...(item.aliases || [])].includes(target))
    if (named.length > 1) throw new HttpError(409, 'multiple allowlisted recipients have this name; use an exact identifier')
    recipient = named[0]
  }
  if (!recipient) throw new HttpError(403, 'recipient is outside the outbound allowlist')
  return { threadId: recipient.thread_id, kind: isChatroomId(recipient.thread_id) ? 'group' : 'direct' }
}

/** 手动候选可以匹配群覆盖，自动路径继续只读取原有授权；群观察不进入候选。 */
function permissionLookup(policy, manual) {
  const observed = new Set()
  const labels = new Map(policy.recipients.map(item => [item.thread_id, item]))
  for (const entry of policy.groupPermissions || []) {
    if (entry.mode === 'observe') observed.add(entry.thread_id)
    else if (manual && entry.mode === 'allowlist' && !labels.has(entry.thread_id)) labels.set(entry.thread_id, entry)
  }
  for (const id of observed) labels.delete(id)
  return {
    allowedRecipients: policy.direct || policy.group || labels.size ? [...labels.values()] : null,
    allowUnlistedGroups: !policy.group,
    allowUnlistedDirect: !policy.direct,
    excludedIds: observed,
  }
}

/** 联系人权限页只暴露操作者选择需要的姓名、备注和精确标识。 */
function contactPolicyPayload(policy, candidates = []) {
  if (!policy || policy.invalid) throw new HttpError(409, '联系人权限文件不可读或不属于当前微信账号。')
  const entries = [
    ...policy.recipients.filter(item => isDirectContactId(item.thread_id)).map(item =>
      ({ id: item.thread_id, displayName: item.label, mode: 'allowlist' })),
    ...(policy.managedOnly || []).map(item => ({ id: item.thread_id, displayName: item.label, mode: 'managed_only' })),
    ...policy.observed.map(item => ({ id: item.thread_id, displayName: item.label, mode: 'observe' })),
  ].map(row => ({ ...row, automation: directAutomation(policy, row.id) }))
  const modes = new Map(entries.map(item => [item.id, item.mode]))
  return {
    ok: true,
    managed: policy.managed,
    entries,
    candidates: candidates.filter(item => isDirectContactId(item.id)).map(item => ({
      id: item.id,
      kind: 'direct',
      displayName: item.displayName,
      remark: item.remark ?? null,
      nickname: item.nickname ?? null,
      alias: item.alias ?? null,
      mode: modes.get(item.id) || 'unassigned',
      automation: directAutomation(policy, item.id),
    })),
  }
}

/** 只接收授权布尔值，群授权开始时间由适配器写入。 */
function requestedAutomation(body) {
  if (!validAutomation(body) || (body.mode === 'observe' && (body.managed === true || body.auto_reply === true))) {
    throw new HttpError(400, '仅观察时不能开启托管或自动回复。请核对权限选项。')
  }
  const flags = {}
  if (body.managed !== undefined) flags.managed = body.managed
  if (body.auto_reply !== undefined) flags.auto_reply = body.auto_reply
  return flags
}

/** 授权需核实当前通讯录；名单中已知的精确联系人可在通讯录不可用时撤销。 */
async function updateContactPermission(body, client, directory) {
  const { recipient_id: id, mode } = body
  if (!isDirectContactId(id) || !['allowlist', 'managed_only', 'observe'].includes(mode)) {
    throw new HttpError(400, '请选择精确的微信联系人和有效权限。')
  }
  const automation = requestedAutomation(body)
  if (mode === 'managed_only' && (automation.managed === false || automation.auto_reply === true)) {
    throw new HttpError(400, '仅托管允许处理私聊消息，自动回复必须关闭。')
  }
  const previous = await client.contactPolicy()
  if (previous.invalid) {
    throw new HttpError(409, '联系人权限文件不可读或不属于当前微信账号。')
  }
  const known = mode === 'observe' && [...previous.recipients, ...previous.observed, ...(previous.managedOnly || [])]
    .find(item => item.thread_id === id)
  const person = known ? { username: id, displayName: known.label } : await directory.findExactContact(id).catch(() => {
    throw new HttpError(502, '暂时无法核对当前微信通讯录，请稍后重试。')
  })
  if (!person) throw new HttpError(404, '当前通讯录中找不到这个联系人，请重新选择。')
  const { name, ...contact } = publicContact(person)
  const policy = await client.saveContactPermission(id, name, mode, automation).catch(() => {
    throw new HttpError(500, '保存联系人权限失败，请检查本机权限文件是否可写。')
  })
  if (policy?.conflict) throw new HttpError(400, '私聊自动回复需要开启托管。')
  const { candidates: [recipient], ...result } = contactPolicyPayload(policy, [{ ...contact, displayName: name }])
  return { ...result, recipient }
}

/** 只暴露联系人展示与寻址字段，不返回内部 payload。 */
function publicContact(contact) {
  const id = String(contact.username || contact.id || '')
  return {
    id,
    name: displayNameOf(contact, id),
    remark: contact.remark ?? null,
    nickname: contact.nickname ?? null,
    alias: contact.alias ?? null,
  }
}

/** 刷新已选的精确私聊联系人；权限页可读回仅托管或观察资料，无需授予外发。 */
async function refreshSelectedContact(body, client, directory, { policyRecord = false } = {}) {
  const id = body.recipient_id
  // 排除 URI 路径特例和群标识，避免请求被解释成另一条路由或群对象。
  if (!isDirectContactId(id)) {
    throw new HttpError(400, '请选择一位微信联系人后刷新资料。')
  }
  const policy = await client.contactPolicy()
  if (policyRecord) {
    if (policy.invalid) throw new HttpError(409, '联系人权限文件不可读或不属于当前微信账号。')
    const entries = [...policy.recipients, ...policy.observed, ...(policy.managedOnly || [])]
    if (!entries.some(item => item.thread_id === id)) throw new HttpError(404, '权限名单中没有这个联系人。')
  } else if (policy.direct) {
    resolveAllowedTarget(policy.recipients, id, 'direct', { exactOnly: true })
  }
  let result
  try {
    result = await directory.refreshContact(id)
  } catch (error) {
    if (error.code === 'refresh_contact_not_found') {
      throw new HttpError(404, '未找到可刷新的微信联系人。')
    }
    if (error.code === 'refresh_contact_failed') {
      throw new HttpError(502, '微信未完成这次联系人刷新，请稍后重试。')
    }
    throw new HttpError(502, '暂时无法核对微信联系人刷新结果，请稍后重试。')
  }
  if (result.pending) return { ok: false, pending: true }
  const { name, ...contact } = publicContact(result.contact)
  return { ok: true, recipient: { ...contact, kind: 'direct', displayName: name }, projection: result.projection }
}

/** 只暴露群名称和群 id，避免泄漏多余的上游字段。 */
function publicRoom(group) {
  const id = String(group.id || group.roomId || '')
  return {
    id,
    name: displayNameOf(group, id),
    topic: String(group.displayName || ''),
    remark: group.remark ?? null,
  }
}

/** 输出可编辑群权限和当前生效值；原名单的群在继承时仍有白名单资格。 */
function groupPermissionRow(policy, id, displayName) {
  const override = groupPermissionMode(policy, id)
  const inherited = policy.group && policy.recipients.some(entry => entry.thread_id === id)
  return {
    id, kind: 'group', displayName, remark: null, nickname: null, alias: null,
    mode: override || (inherited ? 'allowlist' : 'unassigned'), override, automation: groupAutomation(policy, id),
  }
}

/** 精确匹配已保存的通讯录资料，缺失的联系人仍以权限标签显示和搜索。 */
function permissionEntries(policy, savedContacts, needle) {
  const byId = new Map(savedContacts.filter(item => item && item.deleted !== true)
    .map(item => [String(item.username || item.id || ''), item]))
  const matches = (...fields) => !needle || fields.some(value =>
    typeof value === 'string' && value.toLocaleLowerCase().includes(needle))
  const contacts = []
  for (const row of contactPolicyPayload(policy).entries) {
    const saved = byId.get(row.id)
    const { name, remark = null, nickname = null, alias = null } = saved ? publicContact(saved) : {}
    const displayName = name && name !== row.id ? name : row.displayName
    if (matches(row.id, row.displayName, displayName, remark, nickname, alias)) {
      contacts.push({ ...row, kind: 'direct', displayName, remark, nickname, alias })
    }
  }
  const groups = new Map()
  for (const entry of [...policy.recipients, ...policy.groupPermissions]) {
    if (isGroupPermissionId(entry.thread_id)) groups.set(entry.thread_id, entry.label)
  }
  const groupRows = [...groups].filter(([id, label]) => matches(id, label))
    .map(([id, label]) => groupPermissionRow(policy, id, label))
  return [...contacts, ...groupRows]
}

/** 权限和通讯录列表只读本机已保存资料，不触发微信同步或放行外发。 */
async function readPermissionRoster(client, directory, params) {
  const view = params.get('view') || 'policy'
  const query = (params.get('query') || '').trim()
  const cursor = params.get('cursor') || ''
  const limitText = params.get('limit') || '40'
  const limit = Number(limitText)
  if (!['policy', 'contacts', 'groups'].includes(view) || Buffer.byteLength(query) > 256 ||
      Buffer.byteLength(cursor) > 8192 || !/^\d+$/u.test(limitText) || !Number.isInteger(limit) || limit < 1 || limit > 100) {
    throw new HttpError(400, '联系人和群的分页参数无效。')
  }
  const policy = await client.contactPolicy()
  if (policy.invalid) throw new HttpError(409, '联系人权限文件不可读或不属于当前微信账号。')
  const info = { ok: true, account_id: client.accountId, managed: policy.managed }
  if (view === 'policy') {
    const offset = Number(cursor || '0')
    if ((cursor && !/^\d+$/u.test(cursor)) || !Number.isSafeInteger(offset)) {
      throw new HttpError(400, '授权名单分页参数无效。')
    }
    const needle = query.toLocaleLowerCase()
    if (policy.recipients.some(item => isDirectContactId(item.thread_id)) || policy.observed.length || policy.managedOnly?.length) {
      await directory.refreshSavedContacts()
    }
    const entries = permissionEntries(policy, directory.contacts, needle)
    return { ...info, items: entries.slice(offset, offset + limit),
      nextCursor: offset + limit < entries.length ? String(offset + limit) : null }
  }

  // 群目录和详情共用读取预算，给 Runtime 的控制面代理留出响应时间。
  const groupReadSignal = view === 'groups' ? AbortSignal.timeout(8_000) : undefined
  let upstream
  try {
    const options = { search: query, cursor: cursor || null, limit }
    upstream = view === 'groups'
      ? await client.listGroups(options, { signal: groupReadSignal })
      : await client.listContacts(options)
  } catch (error) {
    if (error.status === 409) throw new HttpError(409, '通讯录分页已过期，请重新加载。')
    throw new HttpError(502, '暂时无法读取当前微信通讯录，请稍后重试。')
  }
  const nextCursor = upstream?.nextCursor ?? null
  if (!Array.isArray(upstream?.items) || (nextCursor !== null &&
      (typeof nextCursor !== 'string' || !nextCursor || Buffer.byteLength(nextCursor) > 8192))) {
    throw new HttpError(502, '暂时无法读取当前微信通讯录，请稍后重试。')
  }
  const current = upstream.items.filter(row => row && row.deleted !== true)
  if (view === 'groups') {
    const groups = current.filter(room => isGroupPermissionId(String(room.id || room.roomId || '')))
    const listed = await filterListedGroups(client, groups, { signal: groupReadSignal })
    const items = listed.map(publicRoom).map(room => groupPermissionRow(policy, room.id, room.name))
    return { ...info, items, nextCursor }
  }
  const rows = current.filter(isPersonalContact).map(publicContact)
    .map(({ name, ...contact }) => ({ ...contact, displayName: name }))
  return { ...info, items: contactPolicyPayload(policy, rows).candidates, nextCursor }
}

/** 新授予群白名单须核实当前群目录，已登记的群可无需连接即撤销或继承。 */
async function updateGroupPermission(body, client, directory) {
  const { recipient_id: id, mode } = body
  if (!isGroupPermissionId(id) || !['allowlist', 'observe', 'inherit'].includes(mode)) {
    throw new HttpError(400, '请选择精确的微信群和有效权限。')
  }
  const automation = requestedAutomation(body)
  const policy = await client.contactPolicy()
  if (policy.invalid) throw new HttpError(409, '联系人权限文件不可读或不属于当前微信账号。')
  const known = mode !== 'allowlist' && !Object.values(automation).includes(true) &&
    [...policy.groupPermissions, ...policy.recipients].find(item => item.thread_id === id)
  const room = known ? { roomId: id, displayName: known.label } : await directory.findExactGroup(id).catch(() => {
    throw new HttpError(502, '暂时无法核对当前微信群目录，请稍后重试。')
  })
  if (!room) throw new HttpError(404, '当前目录中找不到这个微信群，请重新选择。')
  const name = displayNameOf(room, id)
  const saved = await client.saveGroupPermission(id, name, mode, automation).catch(() => {
    throw new HttpError(500, '保存群权限失败，请检查本机权限文件是否可写。')
  })
  if (!saved || saved.invalid) throw new HttpError(409, '联系人权限文件不可读或不属于当前微信账号。')
  return { ok: true, account_id: client.accountId, item: groupPermissionRow(saved, id, name) }
}

/** 兼容旧回执，并保留上游给出的本地消息 ID 和创建时间供已投递消息对账。 */
function statePayload(state) {
  const accepted = state?.state === 'accepted'
  const localId = accepted && typeof state.localMessageId === 'string' ? state.localMessageId : ''
  const serverId = accepted && typeof state.serverId === 'string' ? state.serverId : ''
  const createdAt = accepted && typeof state.createdAt === 'string' ? state.createdAt : ''
  return {
    send_id: String(state?.id || ''),
    state: String(state?.state || ''),
    message_id: String(state?.localMessageId || state?.serverId || ''),
    ...(localId ? { local_message_id: localId } : {}),
    ...(serverId ? { server_message_id: serverId } : {}),
    ...(createdAt ? { message_created_at: createdAt } : {}),
    error_code: state?.errorCode ? String(state.errorCode) : null,
  }
}

/**
 * 启动控制面。
 *
 * 返回 `{ server, close }`；调用方负责在收到信号时关闭。
 */
export async function startControlServer({ config, client, directory, switches, adapter, logger = console }) {
  const server = http.createServer(async (request, response) => {
    const pathname = new URL(request.url || '/', 'http://127.0.0.1').pathname
    try {
      if (request.method === 'OPTIONS') {
        sendJson(response, 200, { ok: true })
        return
      }
      if (pathname === '/health' && request.method === 'GET') {
        const policy = await client.contactPolicy()
        sendJson(response, 200, {
          ok: true,
          adapter: 'agent-wechat',
          upstream_base_url: config.upstreamBaseUrl,
          account_id: adapter?.account?.id || client.accountId || '',
          own_username: adapter?.selfUsername || '',
          dry_run: Boolean(config.dryRun),
          send_channel_enabled: switches.sendChannelEnabled,
          send_channel_updated_at: switches.sendChannelUpdatedAt,
          auto_send_reply: switches.autoSendReply,
          direct_contact_policy_enabled: policy.direct,
          outbound_recipient_policy_enabled: policy.direct || policy.group,
          contact_policy_editable: !config.dryRun,
          auto_send_reply_updated_at: switches.autoSendReplyUpdatedAt,
          known_contact_count: directory.contacts.length,
          known_room_count: directory.groups.length,
          cursor: String(adapter?.cursor || ''),
        })
        return
      }

      if (config.controlToken && !tokenMatches(requestToken(request), config.controlToken)) {
        sendJson(response, 401, { ok: false, error: 'unauthorized' })
        return
      }

      if (pathname === '/media' && request.method === 'GET') {
        // Unlike ordinary loopback controls, media always requires a trusted Runtime credential.
        const token = config.controlToken || config.agentToken
        if (!token || !tokenMatches(requestToken(request), token)) {
          sendJson(response, 401, { ok: false, error: 'media_unauthorized', code: 'media_unauthorized' })
          return
        }
        if (config.dryRun) throw new UpstreamError('media is disabled in dry-run mode', { status: 409, code: 'media_dry_run' })
        const query = new URL(request.url, 'http://127.0.0.1').searchParams
        const keys = ['account_id', 'thread_id', 'message_id', 'representation']
        if (keys.some(key => query.getAll(key).length !== 1)) {
          throw new UpstreamError('an exact media reference is required', { status: 400, code: 'invalid_media_reference' })
        }
        const accountId = query.get('account_id')
        const threadId = query.get('thread_id')
        const requireScope = async () => {
          const scope = await consoleConversationScope(client, accountId)
          if (!scope.has(threadId)) {
            throw new UpstreamError('media source was not found', { status: 404, code: 'media_source_mismatch' })
          }
        }
        const controller = new AbortController()
        response.once('close', () => controller.abort())
        await requireScope()
        if (controller.signal.aborted) return
        const media = await readObservedMedia(client, {
          accountId, threadId, messageId: query.get('message_id'), representation: query.get('representation'),
          signal: controller.signal,
        })
        // A permission removal during a download also invalidates the response.
        await requireScope()
        if (!response.destroyed) sendMedia(response, media)
        return
      }

      if (pathname === '/send-channel' && request.method === 'GET') {
        sendJson(response, 200, {
          ok: true,
          send_channel_enabled: switches.sendChannelEnabled,
          updated_at: switches.sendChannelUpdatedAt,
        })
        return
      }
      if (pathname === '/send-channel' && request.method === 'POST') {
        const body = await readJsonBody(request, config.requestBodyLimitBytes)
        if (typeof body.enabled !== 'boolean') throw new HttpError(400, 'enabled must be boolean')
        if (body.enabled && config.dryRun) throw new HttpError(409, 'dry_run mode cannot enable sending')
        switches.sendChannelEnabled = body.enabled
        switches.sendChannelUpdatedAt = new Date().toISOString()
        logger.info?.(`send_channel_enabled=${switches.sendChannelEnabled}`)
        sendJson(response, 200, {
          ok: true,
          send_channel_enabled: switches.sendChannelEnabled,
          updated_at: switches.sendChannelUpdatedAt,
        })
        return
      }

      if (pathname === '/auto-send' && request.method === 'GET') {
        sendJson(response, 200, {
          ok: true,
          auto_send_reply: switches.autoSendReply,
          updated_at: switches.autoSendReplyUpdatedAt,
        })
        return
      }
      if (pathname === '/auto-send' && request.method === 'POST') {
        const body = await readJsonBody(request, config.requestBodyLimitBytes)
        if (typeof body.enabled !== 'boolean') throw new HttpError(400, 'enabled must be boolean')
        if (body.enabled && config.dryRun) throw new HttpError(409, 'dry_run mode cannot enable sending')
        switches.autoSendReply = body.enabled
        switches.autoSendReplyUpdatedAt = new Date().toISOString()
        logger.info?.(`auto_send_reply=${switches.autoSendReply}`)
        sendJson(response, 200, {
          ok: true,
          auto_send_reply: switches.autoSendReply,
          updated_at: switches.autoSendReplyUpdatedAt,
        })
        return
      }

      if (pathname === '/contacts' && request.method === 'GET') {
        await directory.refresh()
        sendJson(response, 200, { ok: true, contacts: directory.contacts.filter(isPersonalContact).map(publicContact) })
        return
      }
      if (['/contacts/refresh', '/contact-policy/refresh'].includes(pathname) && request.method === 'POST') {
        const token = config.controlToken || config.agentToken
        if (!token || !tokenMatches(requestToken(request), token)) {
          sendJson(response, 401, { ok: false, error: 'unauthorized' })
          return
        }
        const policyRecord = pathname === '/contact-policy/refresh'
        if (policyRecord && config.dryRun) throw new HttpError(409, '只读演练期间不能刷新联系人资料。')
        const body = await readJsonBody(request, config.requestBodyLimitBytes)
        const result = await refreshSelectedContact(body, client, directory, { policyRecord })
        sendJson(response, result.pending ? 202 : 200, result)
        return
      }
      if ((pathname === '/permission-roster' && request.method === 'GET') ||
          (pathname === '/group-permission' && request.method === 'POST')) {
        const token = config.controlToken || config.agentToken
        if (!token || !tokenMatches(requestToken(request), token)) {
          sendJson(response, 401, { ok: false, error: 'unauthorized' })
          return
        }
        if (pathname === '/permission-roster') {
          const params = new URL(request.url, 'http://127.0.0.1').searchParams
          sendJson(response, 200, await readPermissionRoster(client, directory, params))
          return
        }
        if (config.dryRun) throw new HttpError(409, '只读演练模式不能保存群权限。')
        const body = await readJsonBody(request, config.requestBodyLimitBytes)
        sendJson(response, 200, await updateGroupPermission(body, client, directory))
        return
      }
      if (['/console-conversations', '/console-conversation'].includes(pathname) && request.method === 'GET') {
        const token = config.controlToken || config.agentToken
        if (!token || !tokenMatches(requestToken(request), token)) {
          sendJson(response, 401, { ok: false, error: 'unauthorized' })
          return
        }
        if (pathname === '/console-conversations') {
          sendJson(response, 200, await consoleConversations(client))
          return
        }
        const params = new URL(request.url, 'http://127.0.0.1').searchParams
        const threadId = params.get('thread_id') || ''
        const kind = params.get('kind') || ''
        const cursor = params.get('cursor') || ''
        const limitText = params.get('limit') || '40'
        const limit = Number(limitText)
        if (!(kind === 'direct' && isDirectContactId(threadId) || kind === 'group' && isGroupPermissionId(threadId)) ||
          Buffer.byteLength(cursor) > 8192 || /\p{Cc}/u.test(cursor) || !/^\d{1,2}$/u.test(limitText) || limit < 1 || limit > 60) {
          throw new HttpError(400, '请选择有效的微信会话。')
        }
        const { status, payload } = await consoleConversation({ client, directory, ledgerPath: config.deliveryLedgerPath,
          threadId, kind, cursor, limit })
        sendJson(response, status, payload)
        return
      }
      if (pathname === '/contact-policy' && ['GET', 'POST'].includes(request.method)) {
        const token = config.controlToken || config.agentToken
        if (!token || !tokenMatches(requestToken(request), token)) {
          sendJson(response, 401, { ok: false, error: 'unauthorized' })
          return
        }
        if (request.method === 'POST') {
          if (config.dryRun) throw new HttpError(409, '只读演练期间不能修改联系人权限。')
          const body = await readJsonBody(request, config.requestBodyLimitBytes)
          sendJson(response, 200, await updateContactPermission(body, client, directory))
          return
        }
        const query = (new URL(request.url, 'http://127.0.0.1').searchParams.get('query') || '').trim()
        if (Buffer.byteLength(query) > 256) throw new HttpError(400, '搜索词太长。')
        const policy = await client.contactPolicy()
        const result = query && !policy.invalid ? await directory.resolve(query, { kind: 'direct' }) : { candidates: [] }
        sendJson(response, 200, contactPolicyPayload(policy, result.candidates))
        return
      }
      if (pathname === '/rooms' && request.method === 'GET') {
        await directory.refresh()
        const rooms = await filterListedGroups(client, directory.groups)
        sendJson(response, 200, { ok: true, rooms: rooms.map(publicRoom) })
        return
      }

      if (pathname === '/resolve' && request.method === 'GET') {
        const url = new URL(request.url || '/', 'http://127.0.0.1')
        const query = url.searchParams.get('query') || ''
        const rawKind = url.searchParams.get('kind') || ''
        const kind = ['room', 'group'].includes(rawKind.toLowerCase())
          ? 'group'
          : ['contact', 'direct'].includes(rawKind.toLowerCase())
            ? 'direct'
            : ''
        const policy = await client.contactPolicy()
        const resolved = await directory.resolve(query, { kind, ...permissionLookup(policy, true) })
        sendJson(response, 200, { ok: true, ...resolved })
        return
      }

      if (pathname === '/send' && request.method === 'POST') {
        const body = await readJsonBody(request, config.requestBodyLimitBytes)
        const result = await handleSend({ body, config, client, directory, switches, logger })
        sendJson(response, 200, result)
        return
      }

      if (pathname === '/deliver-ingress-reply' && request.method === 'POST') {
        const token = config.controlToken || config.agentToken
        if (!token || !tokenMatches(requestToken(request), token)) {
          sendJson(response, 401, { ok: false, error: 'unauthorized' })
          return
        }
        const { payload, result } = await readJsonBody(request, config.requestBodyLimitBytes)
        const deliveryState = await deliverIngressReply(adapter, payload, result)
        sendJson(response, 200, { ok: true, delivery_state: deliveryState })
        return
      }

      if (pathname === '/replay-spool' && request.method === 'POST') {
        await adapter?.replaySpool?.()
        sendJson(response, 200, { ok: true, replayed: true })
        return
      }

      sendJson(response, 404, { ok: false, error: 'not found' })
    } catch (error) {
      if (pathname === '/media') {
        if (response.destroyed) return
        const pending = ['AbortError', 'TimeoutError'].includes(error?.name)
        const status = pending ? 503 : error instanceof UpstreamError &&
          [400, 404, 409, 410, 413, 503].includes(error.status) ? error.status : 502
        const code = pending ? 'media_pending' : error instanceof UpstreamError &&
          /^[a-z0-9_]{1,80}$/.test(error.code || '') ? error.code : 'media_unavailable'
        sendJson(response, status, { ok: false, error: code, code })
        return
      }
      const status = error instanceof HttpError ? error.status : 500
      if (status >= 500) logger.error?.(`control ${pathname} failed: ${error?.stack || error}`)
      sendJson(response, status, { ok: false, error: error?.message || String(error), ...(error?.details || {}) })
    }
  })

  await new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(config.controlPort, config.controlHost, () => {
      server.off('error', reject)
      resolve()
    })
  })
  logger.info?.(`control server=http://${config.controlHost}:${config.controlPort}`)

  return {
    server,
    close: () =>
      new Promise((resolve) => {
        server.close(() => resolve())
      }),
  }
}

/** 只交还同一账号的原生入站回复，继续走既有自动外发授权、去重和人工回复检查。 */
async function deliverIngressReply(adapter, payload, result) {
  const raw = payload?.payload
  const source = payload?.source_message_id
  if (payload?.channel !== 'agent-wechat' || payload.self_message ||
      raw?.direction !== 'incoming' || !adapter?.account?.id || raw.account_id !== adapter.account.id ||
      typeof source !== 'string' || !source || raw.id !== source ||
      typeof raw.threadId !== 'string' || !raw.threadId ||
      result?.ok !== true || result.channel !== 'agent-wechat' ||
      !result.route_result || typeof result.route_result !== 'object' || Array.isArray(result.route_result)) {
    throw new HttpError(400, 'invalid ingress reply source')
  }
  const outcome = await adapter.maybeSendReply(payload, result)
  const state = outcome?.state
  if (['accepted', 'failed', 'needs_review', 'cancelled', 'already_delivered', 'owner_already_replied'].includes(state)) {
    return state
  }
  // 等待超时留下的非终态可能已被微信接受，保留待核对结果，避免后台重复投递。
  return state ? 'needs_review' : 'withheld'
}

/** 执行一次外发；仅在控制面调用，业务权限和内容审查已在 Runtime 完成。 */
async function handleSend({ body, config, client, directory, switches, logger }) {
  if (config.dryRun) throw new HttpError(409, 'dry_run mode does not send messages')
  if (!switches.sendChannelEnabled) throw new HttpError(409, 'send channel is disabled')
  const manual = body.source == null || body.source === 'human_approved_invite'
  const policy = await client.contactPolicy()
  const hasGroupOverlay = policy.groupPermissions?.some(entry => entry.mode === 'allowlist')
  if (!switches.autoSendReply && !(manual && (policy.direct || policy.group || hasGroupOverlay))) {
    // 定向模式只单独放开手动发送及人工审批；自动路径仍须经过自动回复开关。
    throw new HttpError(409, 'auto send switch is disabled')
  }
  const text = String(body.text || '')
  if (!text.trim()) throw new HttpError(400, 'text is required')
  if (text.length > MAX_TEXT_LENGTH) throw new HttpError(413, `text exceeds ${MAX_TEXT_LENGTH} characters`)

  const targetType = String(body.target_type || body.targetType || '').toLowerCase()
  const kind = ['room', 'group'].includes(targetType)
    ? 'group'
    : ['contact', 'direct', 'private'].includes(targetType)
      ? 'direct'
      : ''
  if ('exact_target' in body && typeof body.exact_target !== 'boolean') {
    throw new HttpError(400, 'exact_target must be boolean')
  }
  const exactOnly = body.exact_target === true
  const target = exactOnly ? body.to : (body.to || body.room_id || body.roomId || body.contact_id || body.weixin || '')
  // 控制台选中后只能提交一个无空白的会话标识和类型，禁止再解释输入框中的名称。
  if (exactOnly && (typeof target !== 'string' || !/^[^\s\p{Cc}]+$/u.test(target) ||
    !kind || isChatroomId(target) !== (kind === 'group'))) {
    throw new HttpError(400, 'exact target requires a matching conversation id and target_type')
  }
  const lookup = permissionLookup(policy, manual)
  const resolved = policy.group || (policy.direct && kind === 'direct')
    ? resolveAllowedTarget(lookup.allowedRecipients, target, kind, { exactOnly })
    : await resolveTarget(directory, target, kind, { exactOnly, ...lookup })
  // 延迟与恢复回执不能借用手动、看板或已单独授权的旧托管群发送路径。
  const legacyManagedGroupReply = body.source === 'managed_group_domain' && body.message_type !== 'game_board' &&
    Boolean(body.source_message_id || body.native_reply_source)
  const incomingReply = !manual && (Boolean(body.native_reply_source) || legacyManagedGroupReply ||
    ['delayed_input_batch_reply', 'recovered_agent_run_reply'].includes(body.source))
  if (!permitsRecipientSend(policy, resolved.threadId, manual, { incomingReply, legacyManagedGroupReply })) {
    throw new HttpError(403, 'recipient is outside the outbound allowlist')
  }
  const scoped = hasManualRecipientScope(policy, resolved.threadId)
  if (!switches.autoSendReply && !(manual && scoped)) {
    throw new HttpError(409, 'auto send switch is disabled')
  }

  const anchor = String(body.source_trace_id || body.sourceTraceId || body.trace_id || body.source_message_id || '').trim()
  const idempotencyKey = sendIdempotencyKey({ sourceTraceId: anchor, target: resolved.threadId })
  // 只有现有控制面明确知道来源的发送才标识为店主或 Agent，回执须经原生确认。
  const author = manual ? 'owner' : ['delayed_input_batch_reply', 'recovered_agent_run_reply', 'managed_group_domain']
    .includes(body.source) ? 'agent' : null

  if (anchor) {
    // 上游按幂等键保存发送意图；重试时直接复用已接受的结果。
    try {
      const existing = await client.getSendByKey(idempotencyKey)
      if (existing && String(existing.state || '') === 'accepted') {
        if (author) recordConversationSend(config.deliveryLedgerPath, { client, threadId: resolved.threadId,
          state: existing, author, logger })
        return {
          ok: true,
          deduplicated: true,
          to: target,
          target_type: resolved.kind,
          thread_id: resolved.threadId,
          ...statePayload(existing),
          reference_pending: false,
        }
      }
    } catch (error) {
      if (Number(error?.status) !== 404) throw error
    }
  }

  if (manual && groupPermissionMode(policy, resolved.threadId) === 'allowlist') {
    const room = await directory.findExactGroup(resolved.threadId).catch(() => {
      throw new HttpError(502, '暂时无法核对当前微信群目录，请稍后重试。')
    })
    if (!room) throw new HttpError(409, '当前目录中找不到这个微信群，请重新选择。')
  }
  // 名单解析、群目录核对和回执查询期间开关可能已调整，提交新的外发前再核对。
  if (!switches.sendChannelEnabled) throw new HttpError(409, 'send channel is disabled')
  if (!switches.autoSendReply && !(manual && scoped)) throw new HttpError(409, 'auto send switch is disabled')
  let intent
  try {
    intent = await client.sendText({
      threadId: resolved.threadId, text, idempotencyKey,
      manual, requireManualScope: !switches.autoSendReply,
      ...(incomingReply ? { incomingReply, legacyManagedGroupReply } : {}),
      ...(!manual && body.native_reply_source ? { replySource: body.native_reply_source } : {}),
    })
  } catch (error) {
    if (error?.code === 'owner_already_replied') {
      // 抑制结果是正常终态，调用方不用重试，也不能把它记为已投递。
      return { ok: true, suppressed: true, reason: 'owner_already_replied' }
    }
    if (error?.code === 'outbound_recipient_denied') throw new HttpError(403, error.message)
    if (error?.code === 'auto_send_required') throw new HttpError(409, error.message)
    throw error
  }
  if (!intent?.id) throw new HttpError(502, 'upstream returned no send id', { send: intent })
  const state = await client.waitForSend(intent.id, { timeoutMs: config.sendWaitTimeoutMs })
  const payload = statePayload(state || intent)
  logger.info?.(
    `manual send target=${resolved.threadId} kind=${resolved.kind} state=${payload.state || '-'} send=${payload.send_id || '-'}`,
  )
  if (payload.state === 'accepted') {
    if (author) recordConversationSend(config.deliveryLedgerPath, { client, threadId: resolved.threadId,
      state: state || intent, author, logger })
    return {
      ok: true,
      to: target,
      target_type: resolved.kind,
      thread_id: resolved.threadId,
      deduplicated: false,
      ...payload,
      reference_pending: false,
    }
  }
  if (payload.state === 'needs_review') {
    return {
      ok: false,
      error: 'send outcome is unknown; check the chat manually and do NOT retry automatically',
      to: target,
      target_type: resolved.kind,
      thread_id: resolved.threadId,
      ...payload,
    }
  }
  return {
    ok: false,
    error: `send did not reach accepted state: ${payload.state || 'unknown'}`,
    to: target,
    target_type: resolved.kind,
    thread_id: resolved.threadId,
    ...payload,
  }
}
