/**
 * 本地控制面：兼容 wechaty 桥接的外发合同，并提供联系人消歧与开关查询。
 *
 * 只绑定回环地址。`/health` 永远可读；配置了控制 token 时其余路由必须带
 * `X-Mahjong-Agent-Token` 或 `Authorization: Bearer`。Python Runtime 通过
 * `MAHJONG_WECHATY_OUTBOUND_TOKEN` 发送同一个 token。
 */
import http from 'node:http'
import { timingSafeEqual } from 'node:crypto'
import { displayNameOf, sendIdempotencyKey } from './normalize.mjs'

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
  return contact ? { threadId: target, kind: 'direct', displayName: displayNameOf(contact, target) } : null
}

/**
 * 把用户输入解析成唯一会话。
 *
 * 允许“精确 wxid/群 id”或“唯一完全匹配的名字”；重名、仅前缀命中和模糊命中一律
 * 返回 409 + 候选，绝不替业务系统挑人。
 */
export async function resolveTarget(directory, rawTarget, rawKind = '') {
  const target = String(rawTarget || '').trim()
  if (!target) throw new HttpError(400, 'target is required')
  const kindHint = String(rawKind || '').toLowerCase()
  const kind = ['room', 'group'].includes(kindHint) ? 'group' : ['contact', 'direct'].includes(kindHint) ? 'direct' : ''

  await directory.refresh()
  if (target.endsWith('@chatroom')) {
    return { threadId: target, kind: 'group', displayName: exactTarget(directory, 'group', target)?.displayName || target }
  }
  const byId = exactTarget(directory, kind || 'direct', target) || (kind ? null : exactTarget(directory, 'group', target))
  if (byId) return byId

  const resolved = await directory.resolve(target, { kind })
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

/** 限定外发按本机确认过的精确会话或名称寻址，碰到重名时要求使用会话标识。 */
async function resolveAllowedTarget(client, rawTarget, kind) {
  const target = String(rawTarget || '').trim()
  if (!target) throw new HttpError(400, 'target is required')
  const recipients = await client.outboundRecipients()
  const eligible = (recipients || []).filter(item =>
    !kind || (item.thread_id.endsWith('@chatroom') ? 'group' : 'direct') === kind)
  const exact = eligible.find(item => item.thread_id === target)
  const named = exact ? [exact] : eligible.filter(item => [item.label, ...(item.aliases || [])].includes(target))
  if (named.length > 1) throw new HttpError(409, 'multiple allowlisted recipients have this name; use an exact identifier')
  if (!named.length) throw new HttpError(403, 'recipient is outside the outbound allowlist')
  const recipient = named[0]
  return { threadId: recipient.thread_id, kind: recipient.thread_id.endsWith('@chatroom') ? 'group' : 'direct' }
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

/** 把上游发送结果收敛为 Runtime 兼容的回执字段。 */
function statePayload(state) {
  return {
    send_id: String(state?.id || ''),
    state: String(state?.state || ''),
    message_id: String(state?.localMessageId || state?.serverId || ''),
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
          outbound_recipient_policy_enabled: Boolean(config.outboundAllowlistPath),
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
        sendJson(response, 200, { ok: true, contacts: directory.contacts.map(publicContact) })
        return
      }
      if (pathname === '/rooms' && request.method === 'GET') {
        await directory.refresh()
        sendJson(response, 200, { ok: true, rooms: directory.groups.map(publicRoom) })
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
        const resolved = await directory.resolve(query, { kind })
        sendJson(response, 200, { ok: true, ...resolved })
        return
      }

      if (pathname === '/send' && request.method === 'POST') {
        const body = await readJsonBody(request, config.requestBodyLimitBytes)
        const result = await handleSend({ body, config, client, directory, switches, logger })
        sendJson(response, 200, result)
        return
      }

      if (pathname === '/replay-spool' && request.method === 'POST') {
        await adapter?.replaySpool?.()
        sendJson(response, 200, { ok: true, replayed: true })
        return
      }

      sendJson(response, 404, { ok: false, error: 'not found' })
    } catch (error) {
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

/** 执行一次外发；仅在控制面调用，业务权限和内容审查已在 Runtime 完成。 */
async function handleSend({ body, config, client, directory, switches, logger }) {
  if (config.dryRun) throw new HttpError(409, 'dry_run mode does not send messages')
  if (!switches.sendChannelEnabled) throw new HttpError(409, 'send channel is disabled')
  const manual = body.source == null || body.source === 'human_approved_invite'
  if (!switches.autoSendReply && !(config.outboundAllowlistPath && manual)) {
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
  const target = body.to || body.room_id || body.roomId || body.contact_id || body.weixin || ''
  const resolved = config.outboundAllowlistPath
    ? await resolveAllowedTarget(client, target, kind)
    : await resolveTarget(directory, target, kind)

  const anchor = String(body.source_trace_id || body.sourceTraceId || body.trace_id || body.source_message_id || '').trim()
  const idempotencyKey = sendIdempotencyKey({ sourceTraceId: anchor, target: resolved.threadId })

  if (anchor) {
    // 上游按幂等键保存发送意图；重试时直接复用已接受的结果。
    try {
      const existing = await client.getSendByKey(idempotencyKey)
      if (existing && String(existing.state || '') === 'accepted') {
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

  let intent
  try {
    intent = await client.sendText({ threadId: resolved.threadId, text, idempotencyKey })
  } catch (error) {
    if (error?.code === 'outbound_recipient_denied') throw new HttpError(403, error.message)
    throw error
  }
  if (!intent?.id) throw new HttpError(502, 'upstream returned no send id', { send: intent })
  const state = await client.waitForSend(intent.id, { timeoutMs: config.sendWaitTimeoutMs })
  const payload = statePayload(state || intent)
  logger.info?.(
    `manual send target=${resolved.threadId} kind=${resolved.kind} state=${payload.state || '-'} send=${payload.send_id || '-'}`,
  )
  if (payload.state === 'accepted') {
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
