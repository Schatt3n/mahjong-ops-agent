/**
 * agent-wechat 守护进程的 HTTP/SSE 客户端。
 *
 * 只覆盖适配器真正使用的路由：账号、消息、事件、联系人/群/会话检索、单个联系人刷新、文本外发。
 * 所有响应都按 JSON 合同读取，不解析 daemon 的本地数据库或内部结构。
 */
import { readFile } from 'node:fs/promises'
import { isChatroomId, isOwnerMessageEvent, SYSTEM_MESSAGE_TYPE } from './normalize.mjs'

export class UpstreamError extends Error {
  constructor(message, { status = 0, code = '', body = null } = {}) {
    super(message)
    this.name = 'UpstreamError'
    this.status = status
    this.code = code
    this.body = body
  }
}

/** 把任意值编码为稳定查询串，忽略 null/undefined。 */
function buildQuery(params = {}) {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === '') continue
    search.set(key, String(value))
  }
  const text = search.toString()
  return text ? `?${text}` : ''
}

export class AgentWechatClient {
  /** 绑定上游账号和可选的本地外发名单，真实名单不进入源码。 */
  constructor({ baseUrl, token, accountId = '', fetchImpl = globalThis.fetch, timeoutMs = 30_000, outboundAllowlistPath = '' }) {
    this.baseUrl = String(baseUrl || '').replace(/\/+$/, '')
    this.token = token || ''
    this.accountId = accountId
    this.fetchImpl = fetchImpl
    this.timeoutMs = timeoutMs
    this.outboundAllowlistPath = outboundAllowlistPath
  }

  /** 配置启用后按账号热读精确会话。文件不可读、损坏或不属于本账号时全部拒绝。 */
  async outboundRecipients() {
    if (!this.outboundAllowlistPath) return null
    try {
      const policy = JSON.parse(await readFile(this.outboundAllowlistPath, 'utf8'))
      const recipients = policy.recipients
      if (!this.accountId || policy.account_id !== this.accountId || !Array.isArray(recipients)) return []
      const valid = recipients.every(item =>
        item && typeof item.thread_id === 'string' && /^[^\s\p{Cc}]+$/u.test(item.thread_id) &&
        typeof item.label === 'string' && Boolean(item.label.trim()) &&
        (!('aliases' in item) || (Array.isArray(item.aliases) && item.aliases.every(alias =>
          typeof alias === 'string' && Boolean(alias.trim())))))
      return valid ? recipients : []
    } catch {
      return []
    }
  }

  /** 发起普通 JSON 请求；非 2xx 统一转成带状态码的上游错误。 */
  async request(pathSuffix, { method = 'GET', body, headers = {}, signal = null, timeoutMs = null } = {}) {
    const url = `${this.baseUrl}${pathSuffix}`
    const finalHeaders = { Accept: 'application/json', ...headers }
    if (this.token) finalHeaders.Authorization = `Bearer ${this.token}`
    if (body !== undefined) finalHeaders['Content-Type'] = 'application/json; charset=utf-8'

    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs ?? this.timeoutMs)
    const onAbort = () => controller.abort()
    if (signal) {
      if (signal.aborted) controller.abort()
      else signal.addEventListener('abort', onAbort, { once: true })
    }

    try {
      const response = await this.fetchImpl(url, {
        method,
        headers: finalHeaders,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      })
      const text = await response.text()
      let parsed = null
      if (text) {
        try {
          parsed = JSON.parse(text)
        } catch {
          parsed = { raw: text }
        }
      }
      if (!response.ok) {
        const code = String(parsed?.error?.code || parsed?.code || '')
        throw new UpstreamError(`agent-wechat ${method} ${pathSuffix} -> HTTP ${response.status} ${code}`.trim(), {
          status: response.status,
          code,
          body: parsed,
        })
      }
      return parsed
    } finally {
      clearTimeout(timer)
      if (signal) signal.removeEventListener('abort', onAbort)
    }
  }

  /** 返回可用账号；用于启动时确认只有一个可操作账号。 */
  async listAccounts() {
    const payload = await this.request('/v1/accounts')
    return Array.isArray(payload) ? payload : payload?.items || []
  }

  async status() {
    return this.request('/v1/status')
  }

  /** 读取单条已保存消息的规范化视图。 */
  async getMessage(messageId, { signal } = {}) {
    return this.request(`/v1/accounts/${encodeURIComponent(this.accountId)}/messages/${encodeURIComponent(messageId)}`, { signal })
  }

  /** 读取一页事件；返回 { items, nextCursor }。 */
  async listEvents({ after = '0', limit = 200, signal } = {}) {
    const query = buildQuery({ after, limit })
    const payload = await this.request(`/v1/accounts/${encodeURIComponent(this.accountId)}/events${query}`, { signal })
    return {
      items: Array.isArray(payload?.items) ? payload.items : [],
      nextCursor: String(payload?.nextCursor ?? after),
    }
  }

  /** 外发前独立复查事件尾部，覆盖等待同步 Runtime 响应时 SSE 尚未处理的老板回复。 */
  async checkForOwnerReply(threadId, source) {
    const unavailable = () => new UpstreamError('could not check for a newer owner reply', {
      status: 503, code: 'owner_reply_check_unavailable',
    })
    if (source.account_id !== this.accountId || source.thread_id !== threadId ||
        typeof source.cursor !== 'string' || !/^[1-9]\d{0,18}$/.test(source.cursor)) throw unavailable()

    const room = isChatroomId(threadId)
    if (room && !source.server_id) return false
    const sourceTime = Date.parse(source.created_at)
    const signal = AbortSignal.timeout(2000)
    const limit = 200
    let after = source.cursor
    // 历史积压时只查有限页，超过时本次不发，避免长期占用入站跟随循环。
    for (let page = 0; page < 4; page += 1) {
      const { items, nextCursor } = await this.listEvents({ after, limit, signal })
      for (const event of items) {
        if (!isOwnerMessageEvent(event, threadId)) continue
        const message = await this.getMessage(event.data.messageId, { signal })
        if (!message?.id || message.threadId !== threadId) throw unavailable()
        if (message.direction !== 'outgoing' || message.recall || Number(message.type) === SYSTEM_MESSAGE_TYPE ||
            Date.parse(message.createdAt) < sourceTime) continue
        const quotedId = message.quote?.serverId
        // 群内可能有多人提问，只按精确引用关联。私聊中明确引用其他消息时也保留本轮。
        if (quotedId && source.server_id) {
          if (quotedId === source.server_id) return true
        } else if (!room) {
          return true
        }
      }
      if (items.length < limit) return false
      if (after === nextCursor) throw unavailable()
      after = nextCursor
    }
    throw unavailable()
  }

  /**
   * 以 SSE 方式跟随事件。
   *
   * daemon 会先回一行注释表示已连接，然后每 500ms 推送新增事件。这里按行解析
   * `data:` 负载；`error` 事件抛出，由调用方决定重连还是补齐游标。
   */
  async streamEvents({ after = '0', signal, onEvent }) {
    const query = buildQuery({ after })
    const url = `${this.baseUrl}/v1/accounts/${encodeURIComponent(this.accountId)}/events/stream${query}`
    const headers = { Accept: 'text/event-stream' }
    if (this.token) headers.Authorization = `Bearer ${this.token}`

    const response = await this.fetchImpl(url, { headers, signal })
    if (!response.ok || !response.body) {
      const text = await response.text().catch(() => '')
      let code = ''
      try {
        const error = JSON.parse(text)
        code = String(error?.error?.code || error?.code || '')
      } catch {
        /* 非 JSON 的 HTTP 错误只保留状态码。 */
      }
      throw new UpstreamError(`event stream failed: HTTP ${response.status} ${text}`.slice(0, 400), {
        status: response.status,
        code,
      })
    }

    const decoder = new TextDecoder()
    let buffer = ''
    let eventType = ''
    let dataLines = []

    for await (const chunk of response.body) {
      buffer += decoder.decode(chunk, { stream: true })
      let index = buffer.indexOf('\n')
      while (index >= 0) {
        const rawLine = buffer.slice(0, index).replace(/\r$/, '')
        buffer = buffer.slice(index + 1)
        if (rawLine === '') {
          if (dataLines.length > 0) {
            const payloadText = dataLines.join('\n')
            dataLines = []
            if (eventType === 'error') {
              let code = 'stream_failed'
              try {
                code = JSON.parse(payloadText)?.error?.code || code
              } catch {
                /* 保留默认错误码 */
              }
              throw new UpstreamError(`event stream requires recovery: ${code}`, { code })
            }
            const parsed = JSON.parse(payloadText)
            if (signal?.aborted) return
            await onEvent(parsed)
          }
          eventType = ''
        } else if (rawLine.startsWith(':')) {
          /* 心跳注释，忽略 */
        } else if (rawLine.startsWith('data:')) {
          dataLines.push(rawLine.slice(5).replace(/^ /, ''))
        } else if (rawLine.startsWith('event:')) {
          eventType = rawLine.slice(6).trim()
        }
        index = buffer.indexOf('\n')
      }
    }
  }

  /** 提交文本外发；同一 Idempotency-Key 重放会返回原始结果。 */
  async sendText({ threadId, text, idempotencyKey, quote = null, mentions = null, replySource = null }) {
    // 在每条真实上游发送路径的汇合点重查精确接收人，涵盖自动回复和带引用外发。
    const recipients = await this.outboundRecipients()
    if (recipients && !recipients.some(item => item.thread_id === threadId)) {
      throw new UpstreamError('recipient is outside the outbound allowlist', { status: 403, code: 'outbound_recipient_denied' })
    }
    if (!idempotencyKey) throw new Error('idempotencyKey is required')
    if (replySource && await this.checkForOwnerReply(threadId, replySource)) {
      throw new UpstreamError('owner already replied', { status: 409, code: 'owner_already_replied' })
    }
    const body = { threadId, text }
    if (quote) body.quote = quote
    if (mentions) body.mentions = mentions
    return this.request(`/v1/accounts/${encodeURIComponent(this.accountId)}/sends`, {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
      body,
    })
  }

  async getSend(sendId) {
    return this.request(`/v1/accounts/${encodeURIComponent(this.accountId)}/sends/${encodeURIComponent(sendId)}`)
  }

  async getSendByKey(key) {
    return this.request(
      `/v1/accounts/${encodeURIComponent(this.accountId)}/sends/by-idempotency-key${buildQuery({ key })}`,
    )
  }

  async listContacts({ search = '', cursor = null, limit = 200 } = {}) {
    return this.request(
      `/v1/accounts/${encodeURIComponent(this.accountId)}/contacts${buildQuery({ search, cursor, limit })}`,
    )
  }

  /** 按精确微信标识读取守护进程已保存的联系人资料。 */
  async getContact(contactId, { signal } = {}) {
    const path = `/v1/accounts/${encodeURIComponent(this.accountId)}/contacts/${encodeURIComponent(contactId)}`
    return this.request(path, { signal })
  }

  /** 请求上游拉取并保存一个联系人，返回用于核对结果的操作记录。 */
  async startContactRefresh(contactId, { signal } = {}) {
    const path = `/v1/accounts/${encodeURIComponent(this.accountId)}/contacts/${encodeURIComponent(contactId)}/refresh`
    return this.request(path, { method: 'POST', body: {}, signal })
  }

  /** 在当前账号下查询操作，避免用另一个账号的结果覆盖本机目录。 */
  async getOperation(operationId, { signal } = {}) {
    const path = `/v1/accounts/${encodeURIComponent(this.accountId)}/operations/${encodeURIComponent(operationId)}`
    return this.request(path, { signal })
  }

  async listGroups({ search = '', cursor = null, limit = 200 } = {}) {
    return this.request(
      `/v1/accounts/${encodeURIComponent(this.accountId)}/groups${buildQuery({ search, cursor, limit })}`,
    )
  }

  async listThreads({ search = '', kind = '', cursor = null, limit = 200 } = {}) {
    return this.request(
      `/v1/accounts/${encodeURIComponent(this.accountId)}/threads${buildQuery({ search, kind, cursor, limit })}`,
    )
  }

  async getThread(threadId) {
    return this.request(`/v1/accounts/${encodeURIComponent(this.accountId)}/threads/${encodeURIComponent(threadId)}`)
  }

  /** 轮询外发状态直到终态；needs_review 必须交人工确认，不能自动重发。 */
  async waitForSend(sendId, { timeoutMs = 60_000, intervalMs = 1_000, signal = null } = {}) {
    const deadline = Date.now() + timeoutMs
    let last = null
    while (Date.now() < deadline) {
      if (signal?.aborted) throw new Error('send wait aborted')
      last = await this.getSend(sendId)
      const state = String(last?.state || '')
      if (['accepted', 'failed', 'needs_review', 'cancelled'].includes(state)) return last
      await new Promise((resolve) => setTimeout(resolve, intervalMs))
    }
    return last
  }
}
