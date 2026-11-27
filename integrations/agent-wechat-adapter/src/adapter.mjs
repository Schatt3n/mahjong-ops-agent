/**
 * 通道主循环：跟随上游事件、翻译消息、投递 Runtime、按开关回发。
 *
 * 关键边界：
 * - 只通过 HTTP/SSE 读取上游，不访问 daemon 的数据库、锁或本地文件；
 * - 收到 Runtime 持久接收确认后推进游标；自动回复通过后台处理和独立回调返回；
 * - 外发默认关闭，且 needs_review 一律不自动重发。
 */
import { readFile } from 'node:fs/promises'
import { buildChannelPayload, isChatroomId, isOwnerMessageEvent, replyIdempotencyKey, shouldForwardMessage, SYSTEM_MESSAGE_TYPE } from './normalize.mjs'
import { appendDelivery, deliveryReceiptKey, loadCursor, loadDeliveryLedger, persistToSpool,
  listSpool, removeSpoolFile, saveCursor } from './state.mjs'
import { groupPermissionRule } from './contact-policy.mjs'

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const MAX_EVENT_CURSOR = 9_223_372_036_854_775_807n
const EVENT_CURSOR_EXPIRED = 'event_cursor_expired'

/** 统一日志前缀，便于和 Runtime、上游 daemon 的日志对齐排查。 */
export function createLogger(level = 'info') {
  const order = { debug: 10, info: 20, warn: 30, error: 40 }
  const threshold = order[level] ?? order.info
  return {
    debug: (...args) => threshold <= order.debug && console.log('[agent-wechat-adapter]', ...args),
    info: (...args) => threshold <= order.info && console.log('[agent-wechat-adapter]', ...args),
    warn: (...args) => threshold <= order.warn && console.warn('[agent-wechat-adapter]', ...args),
    error: (...args) => threshold <= order.error && console.error('[agent-wechat-adapter]', ...args),
  }
}

/** 从 Runtime 响应里提取最终回复；没有可外发内容时返回空串。 */
export function extractFinalReply(result) {
  const candidate = result?.route_result?.agent_result?.final_reply
  return typeof candidate === 'string' ? candidate.trim() : ''
}

/** 关联经审查的同一条群消息，外发前再读授权。服务器 ID 保留原始字符串精度。 */
async function scopedGroupReplyRequest(accountId, payload, result, policyPath, rule = null) {
  const raw = payload?.payload
  const senderRoom = payload?.room?.id
  const source = payload?.source_message_id
  const approved = result?.route_result?.scoped_group_reply
  if (typeof accountId !== 'string' || !accountId || result?.channel !== 'agent-wechat' ||
      !payload?.is_room || payload.self_message || raw?.direction !== 'incoming' || raw?.native_type !== 1 ||
      raw?.account_id !== accountId || typeof source !== 'string' || !source.trim() ||
      typeof senderRoom !== 'string' || !senderRoom.endsWith('@chatroom') || raw?.threadId !== senderRoom ||
      raw?.id !== source || payload.message_id !== source ||
      approved?.room_id !== senderRoom || approved?.source_message_id !== source ||
      typeof approved?.text !== 'string' || !approved.text.trim()) return null
  let policy
  try {
    policy = rule
      ? { account_id: accountId, room_ids: [rule.thread_id], enabled_at: rule.auto_reply_at }
      : JSON.parse(await readFile(policyPath, 'utf8'))
  } catch {
    return null
  }
  if (policy?.account_id !== accountId || !Array.isArray(policy?.room_ids) ||
      !policy.room_ids.includes(senderRoom) ||
      !(Date.parse(raw.createdAt) >= Date.parse(policy.enabled_at))) return null

  const serverId = raw.serverId
  const quote = typeof serverId === 'string' && /^[1-9][0-9]*$/.test(serverId)
    ? { messageId: source, serverId } : null
  return { threadId: senderRoom, text: approved.text.trim(), quote }
}

/** 向 Runtime raw 入口投递载荷；异步请求收到持久接收确认即可推进游标。 */
export async function postAgentRaw(endpoint, token, payload, {
  fetchImpl = globalThis.fetch, attempts = 3, logger = console, respondAsync = false, deferredReply = false,
  replyBlockReason = '',
} = {}) {
  let lastError = null
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const headers = { 'Content-Type': 'application/json; charset=utf-8' }
      if (token) headers['X-Mahjong-Agent-Token'] = token
      if (respondAsync) headers.Prefer = 'respond-async'
      if (deferredReply) headers['X-Mahjong-Agent-Wechat-Deferred-Reply'] = '1'
      if (replyBlockReason) headers['X-Mahjong-Agent-Wechat-Reply-Blocked'] = replyBlockReason
      const response = await fetchImpl(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        ...(respondAsync ? { signal: AbortSignal.timeout(10_000) } : {}),
      })
      const text = await response.text()
      let parsed = null
      try {
        parsed = JSON.parse(text)
      } catch {
        parsed = { raw_response: text }
      }
      if ((respondAsync && response.ok && response.status !== 202) ||
          (response.status === 202 && (parsed?.ok !== true || parsed?.accepted !== true))) {
        throw new Error('runtime did not confirm durable ingress')
      }
      if (response.ok && deferredReply && parsed?.deferred_reply !== true) {
        throw new Error('runtime did not confirm deferred reply')
      }
      if (response.ok) return parsed
      const error = new Error(`runtime HTTP ${response.status}: ${text.slice(0, 400)}`)
      error.status = response.status
      if (response.status < 500) throw error
      lastError = error
    } catch (error) {
      lastError = error
      if (error?.status && error.status < 500) break
    }
    if (attempt < attempts) await sleep(500 * 2 ** (attempt - 1))
  }
  logger.warn?.(`runtime delivery failed: ${lastError?.message || lastError}`)
  throw lastError || new Error('runtime delivery failed')
}

/** 通过上游事件游标跟随消息，并在 Runtime 与上游之间交接进出站请求。 */
export class AgentWechatAdapter {
  constructor({
    client, directory, config, switches = null, logger = createLogger(config?.logLevel),
    fetchImpl = globalThis.fetch, deferAutomaticReplies = false,
  }) {
    this.client = client
    this.directory = directory
    this.config = config
    this.logger = logger
    // 控制面可在运行时切换这两个开关；缺省值来自环境配置，进程重启后回到安全默认。
    this.switches = switches || {
      sendChannelEnabled: Boolean(config.sendEnabled),
      autoSendReply: Boolean(config.autoSendReply),
    }
    this.fetchImpl = fetchImpl
    this.deferAutomaticReplies = deferAutomaticReplies
    this.delivered = loadDeliveryLedger(config.deliveryLedgerPath)
    this.cursor = '0'
    this.account = null
    this.selfUsername = ''
    this.stopping = false
    this.bootstrapped = false
    // 监听状态是当前 SSE 连接的观测，不用 HTTP 进程存活推断消息订阅。
    this.listener = { ready: false, last_connected_at: null, consecutive_failures: 0, last_error: '' }
  }

  /** 确认唯一账号；多账号必须显式配置，避免把消息路由到错误的微信号。 */
  async resolveAccount() {
    const payload = await this.client.listAccounts()
    const accounts = Array.isArray(payload) ? payload : payload?.items || []
    if (this.config.accountId) {
      const found = accounts.find(
        (item) => item.id === this.config.accountId || item.alias === this.config.accountId,
      )
      if (!found) throw new Error(`configured account not found: ${this.config.accountId}`)
      this.account = found
    } else if (accounts.length === 1) {
      this.account = accounts[0]
    } else if (accounts.length === 0) {
      throw new Error('no agent-wechat account is registered; run `wx auth login` first')
    } else {
      const list = accounts.map((item) => item.id).join(', ')
      throw new Error(`multiple agent-wechat accounts found (${list}); set MAHJONG_AGENT_WECHAT_ACCOUNT`)
    }
    // 显式配置支持 id 或 alias；后续所有路由都必须使用真实账号 id。
    this.client.accountId = this.account.id
    this.selfUsername = String(this.account.ownUsername || '').trim()
    const connection = String(this.account.connection || '')
    this.logger.info(`account=${this.account.id} own=${this.selfUsername || '-'} connection=${connection || '-'}`)
    return this.account
  }

  /** 初始化事件游标：显式配置 > 本地文件 > 快进到日志尾部。 */
  async bootstrapCursor() {
    if (this.bootstrapped) return this.cursor
    if (this.config.startCursor) {
      this.cursor = this.config.startCursor
      this.bootstrapped = true
      this.logger.info(`cursor from MAHJONG_AGENT_WECHAT_START_CURSOR=${this.cursor}`)
      return this.cursor
    }
    const saved = loadCursor(this.config.cursorPath)
    if (saved && (!saved.accountId || saved.accountId === this.account.id)) {
      this.cursor = saved.cursor
      this.bootstrapped = true
      this.logger.info(`cursor restored=${this.cursor}`)
      return this.cursor
    }
    if (this.config.replayHistory) {
      this.cursor = '0'
      this.bootstrapped = true
      this.logger.info('cursor starts at 0 because history replay is enabled')
      return this.cursor
    }
    this.cursor = await this.fastForwardCursor()
    this.logger.info(`cursor fast-forwarded to ${this.cursor} (history replay disabled)`)
    this.persistCursor()
    this.bootstrapped = true
    return this.cursor
  }

  /** 只用于首次启动：把游标推到事件日志末尾，避免把历史消息当成新消息处理。 */
  async fastForwardCursor() {
    let cursor = '0'
    let total = 0
    for (let page = 0; page < 200; page += 1) {
      let events
      try {
        events = await this.client.listEvents({ after: cursor, limit: 200 })
      } catch (error) {
        if (cursor !== '0' || error?.code !== EVENT_CURSOR_EXPIRED) throw error
        cursor = await this.findRetainedCursor(cursor)
        events = await this.client.listEvents({ after: cursor, limit: 200 })
      }
      const { items, nextCursor } = events
      total += items.length
      if (!items.length || nextCursor === cursor) break
      cursor = nextCursor
      if (items.length < 200) break
    }
    this.logger.info(`fast-forward scanned ${total} historical events`)
    return cursor
  }

  /** 找到第一条仍保留事件之前的游标；原生查询允许未来的 i64 游标。 */
  async findRetainedCursor(expiredCursor, { signal } = {}) {
    if (!/^\d{1,19}$/.test(expiredCursor)) throw new Error('invalid expired event cursor')
    let expired = BigInt(expiredCursor)
    let retained = MAX_EVENT_CURSOR
    if (expired >= retained) throw new Error('invalid expired event cursor')
    while (retained - expired > 1n) {
      const candidate = (expired + retained) / 2n
      try {
        await this.client.listEvents({ after: String(candidate), limit: 1, signal })
        retained = candidate
      } catch (error) {
        if (error?.code !== EVENT_CURSOR_EXPIRED) throw error
        expired = candidate
      }
    }

    // 保留窗口可能在查询期间继续前移。最后确认相邻事件存在，失败时交给下一轮重连。
    const cursor = String(retained)
    const { items } = await this.client.listEvents({ after: cursor, limit: 1, signal })
    if (String(items[0]?.cursor) !== String(retained + 1n)) {
      throw new Error('event retention boundary could not be confirmed')
    }
    return cursor
  }

  /** 同步写入当前游标，确保落盘错误在这里记录，并保留旧文件供重启续传。 */
  persistCursor() {
    try {
      saveCursor(this.config.cursorPath, {
        cursor: this.cursor,
        accountId: this.account?.id || '',
        serverId: this.serverId || '',
      })
    } catch (error) {
      this.logger.warn(`cursor persist failed: ${error?.message || error}`)
    }
  }

  /** 把一条已保存消息翻译成 raw 载荷；展示信息全部来自上游 HTTP，不猜。 */
  async buildPayload(message) {
    const thread = await this.directory.getThread(message.threadId)
    let sender = null
    if (String(message.direction || '') === 'outgoing') {
      sender = { id: this.selfUsername, name: this.selfUsername, displayName: this.selfUsername }
    } else {
      sender = this.directory.contact(message.senderId)
    }
    return buildChannelPayload({
      message,
      sender,
      thread,
      accountId: this.account?.id || '',
    })
  }

  /** 联系人事件只作废目录缓存；已保存消息仍按原顺序拉取、过滤、投递和回发。 */
  async handleEvent(event) {
    const { kind, data } = event || {}
    if (kind === 'contacts.updated' || kind === 'contact.updated') {
      const id = data?.contactId
      if (typeof id === 'string' && /^[^\s\p{Cc}]+$/u.test(id) && !isChatroomId(id) && !id.endsWith('@im.chatroom')) {
        this.directory.invalidateContacts()
      }
      return
    }
    if (kind === 'metadata.changed') {
      // 备注的完整资料可能在单条联系人事件之后才写入；落盘后还需作废一次。
      if (data?.kind === 'contacts' || data?.kind === 'contact_catalog') this.directory.invalidateContacts()
      return
    }
    if (kind !== 'message.committed') return
    const messageId = String(data?.messageId || '').trim()
    if (!messageId) return

    const message = await this.client.getMessage(messageId)
    if (!message?.id) throw new Error(`message details unavailable for ${messageId}`)
    const ownerMessage = message.direction === 'outgoing' &&
      isOwnerMessageEvent(event, message.threadId) && !message.recall && Number(message.type) !== SYSTEM_MESSAGE_TYPE
    if (
      !shouldForwardMessage(message, {
        forwardSelfMessages: ownerMessage || this.config.forwardSelfMessages,
        forwardSystemMessages: this.config.forwardSystemMessages,
      })
    ) {
      this.logger.debug(`skip message=${messageId} type=${message.type} direction=${message.direction}`)
      return
    }

    const payload = await this.buildPayload(message)
    if (ownerMessage) payload.message_origin = 'owner'
    if (message.direction === 'incoming' && event.cursor) {
      // 将原生游标和账号锚点随消息落库，延迟/恢复的回复也能按同一条消息复查。
      payload.native_reply_source = {
        cursor: String(event.cursor),
        account_id: this.account.id,
        thread_id: message.threadId,
        server_id: message.serverId || '',
        created_at: message.createdAt || '',
      }
    }
    if (this.config.dryRun) {
      // 只读演练：只打印将要投递的摘要，不写 Runtime、不回发。
      this.logger.info(
        `dry-run message=${payload.message_id} conversation=${payload.conversation_id} type=${payload.message_type} text_len=${payload.text.length}`,
      )
      return
    }
    const delivery = this.ingressDeliveryOptions(payload)
    let result = null
    try {
      result = await postAgentRaw(this.config.agentEndpoint, this.config.agentToken, payload, {
        fetchImpl: this.fetchImpl,
        logger: this.logger,
        ...delivery,
      })
      this.logger.info(
        `forwarded message=${payload.message_id} thread=${message.threadId} trace=${result?.trace_id || '-'}`,
      )
    } catch (error) {
      // 未确认的事件用上游游标重读，否则同会话后续事件可能抢先入队。
      if (delivery.respondAsync || ownerMessage) throw error
      const file = persistToSpool(this.config.spoolDir, payload, error)
      this.logger.warn(`queued for replay file=${file}: ${error?.message || error}`)
    }
    if (result && !result.accepted) await this.maybeSendReply(payload, result)
  }

  /** 自动回复关闭且仅允许定向手动外发时，入站仍只等待持久接收。 */
  durableIngressOnly() {
    return !this.switches.autoSendReply && (!this.switches.sendChannelEnabled || Boolean(this.config.outboundAllowlistPath))
  }

  /** 总闸优先于自动外发，入站与提交前使用一致的拒绝原因。 */
  replySwitchBlockReason() {
    if (!this.switches.sendChannelEnabled) return 'send_channel_disabled'
    if (!this.switches.autoSendReply) return 'auto_send_disabled'
    return ''
  }

  /** 运行进程只等待持久接收，并将可能发送的回复交给 Runtime 后台回调。 */
  ingressDeliveryOptions(payload) {
    const respondAsync = this.deferAutomaticReplies || this.durableIngressOnly()
    const replyBlockReason = respondAsync && !payload.self_message ? this.replySwitchBlockReason() : ''
    return {
      respondAsync,
      deferredReply: this.deferAutomaticReplies && !payload.self_message && !replyBlockReason,
      replyBlockReason,
    }
  }

  /** 事件完成处理后才推进游标。失败时保留现场，避免重连跳过未处理的消息。 */
  async consumeEvent(event) {
    try {
      await this.handleEvent(event)
    } catch (error) {
      throw new Error(`event handling failed cursor=${event?.cursor}: ${error?.message || error}`, { cause: error })
    }
    this.cursor = String(event?.cursor || this.cursor)
    this.persistCursor()
  }

  /** 回发 Runtime 的最终回复；默认关闭，失败只记录不重发。 */
  async maybeSendReply(payload, result) {
    if (payload?.self_message || this.config.dryRun) return null
    const threadId = payload?.payload?.threadId
    if (typeof threadId !== 'string' || !threadId) return null
    const sourceMessageId = String(payload.source_message_id || payload.message_id || '')
    const account = this.account?.id
    const receipt = account && sourceMessageId && payload.payload.account_id === account &&
      this.delivered.get(deliveryReceiptKey({ account_id: account, thread_id: threadId, source_message_id: sourceMessageId }))
    if (receipt) {
      // 已确认的交付早于本次权限检查；收紧权限不改变已经发生的发送。
      if (!receipt.server_message_id && !receipt.local_message_id) return { state: 'already_delivered' }
      return { state: 'accepted', accountId: account, destination: threadId,
        serverId: receipt.server_message_id, localMessageId: receipt.local_message_id }
    }
    let blocked = this.replySwitchBlockReason()
    if (blocked) {
      this.logger.info(`reply withheld because send switches are off trace=${result?.trace_id || '-'}`)
      return { state: 'withheld', reason: blocked }
    }
    let request
    let scopedGroupReply
    if (payload?.is_room || isChatroomId(threadId)) {
      const groupPolicy = await this.client.contactPolicy()
      const rule = groupPermissionRule(groupPolicy, threadId)
      if (groupPolicy.invalid) return { state: 'withheld', reason: 'group_reply_policy_unavailable' }
      if (rule?.mode === 'observe') return { state: 'withheld', reason: 'group_observe_only' }
      if (rule?.auto_reply === false) return { state: 'withheld', reason: 'group_auto_reply_disabled' }
      const explicitGroupReply = rule?.auto_reply === true
      // 任何群入站自动回复都必须有逐群授权与同一条原消息的公开审查回执。
      if (!explicitGroupReply && (!this.config.outboundAllowlistPath || !this.config.groupReplyPolicyPath)) {
        return { state: 'withheld', reason: 'group_auto_reply_disabled' }
      }
      request = await scopedGroupReplyRequest(
        this.account?.id, payload, result, this.config.groupReplyPolicyPath, explicitGroupReply ? rule : null,
      )
      if (!request) return { state: 'withheld', reason: 'group_reply_not_authorized' }
      scopedGroupReply = { receivedAt: payload.payload.createdAt, legacy: !explicitGroupReply }
    } else {
      const finalReply = extractFinalReply(result)
      if (!finalReply) return null
      request = { threadId, text: finalReply }
    }
    blocked = this.replySwitchBlockReason()
    if (blocked) return { state: 'withheld', reason: blocked }
    if (sourceMessageId && this.delivered.has(sourceMessageId)) {
      this.logger.info(`reply skipped; already delivered source=${sourceMessageId}`)
      return { state: 'already_delivered' }
    }
    try {
      const idempotencyKey = replyIdempotencyKey(sourceMessageId)
      const intent = await this.client.sendText({
        ...request,
        idempotencyKey,
        ...(scopedGroupReply ? { scopedGroupReply, incomingReply: true } : {}),
        ...(payload.native_reply_source ? { replySource: payload.native_reply_source } : {}),
      })
      const state = await this.client.waitForSend(intent.id, { timeoutMs: this.config.sendWaitTimeoutMs })
      return this.recordSendOutcome({ payload, intent, state, idempotencyKey })
    } catch (error) {
      if (error?.code === 'outbound_recipient_denied') return { state: 'withheld', reason: 'outbound_recipient_denied' }
      if (error?.code === 'owner_already_replied') {
        this.logger.info(`reply skipped; owner already replied source=${sourceMessageId}`)
        return { state: 'owner_already_replied' }
      }
      this.logger.error(`reply send failed source=${sourceMessageId}: ${error?.message || error}`)
      return { state: 'needs_review' }
    }
  }

  /**
   * 处理外发终态。
   *
   * accepted 才写投递账本；needs_review 表示未知，必须人工确认，禁止自动重试。
   */
  recordSendOutcome({ payload, intent, state, idempotencyKey }) {
    const finalState = String(state?.state || '')
    if (finalState === 'accepted') {
      const messageId = String(state?.localMessageId || state?.serverId || state?.id || '')
      const threadId = payload?.payload?.threadId
      const accountId = this.account?.id
      // 发送方身份只使用同账号和会话的原生回执；原去重账本仍保留兼容格式。
      const attribution = accountId && threadId && state.accountId === accountId && state.destination === threadId
        ? { account_id: accountId, thread_id: threadId, author: 'agent',
          local_message_id: state.localMessageId || '', server_message_id: state.serverId || '' } : {}
      const receipt = {
        source_message_id: String(payload.source_message_id || ''),
        conversation_id: String(payload.conversation_id || ''),
        send_id: String(state?.id || intent?.id || ''),
        message_id: messageId,
        idempotency_key: idempotencyKey,
        state: finalState,
        ...attribution,
      }
      this.delivered.set(deliveryReceiptKey(receipt), receipt)
      appendDelivery(this.config.deliveryLedgerPath, receipt)
      this.logger.info(`reply accepted send=${state?.id} message=${messageId}`)
      return state
    }
    if (finalState === 'needs_review') {
      this.logger.error(
        `reply needs_review send=${state?.id}; check the chat manually and do NOT resend automatically`,
      )
      return state
    }
    this.logger.error(`reply not delivered send=${state?.id} state=${finalState} error=${state?.errorCode || '-'}`)
    return state
  }

  /** 重放 Runtime 暂时不可用期间积压的入站消息。 */
  async replaySpool() {
    for (const file of listSpool(this.config.spoolDir)) {
      try {
        const record = JSON.parse(await readFileText(file))
        const result = await postAgentRaw(this.config.agentEndpoint, this.config.agentToken, record.payload, {
          fetchImpl: this.fetchImpl,
          logger: this.logger,
          ...this.ingressDeliveryOptions(record.payload),
        })
        if (!result?.accepted) await this.maybeSendReply(record.payload, result)
        removeSpoolFile(file)
        this.logger.info(`replayed ${file}`)
      } catch (error) {
        this.logger.warn(`replay deferred ${file}: ${error?.message || error}`)
        break
      }
    }
  }

  /** 建立 SSE 长连接；断线后从最后游标重连，游标过期则改用 JSON 分页补齐。 */
  async run(signal) {
    if (!this.account) await this.resolveAccount()
    await this.bootstrapCursor()
    // 联系人仅用于展示和查询，避免目录分页拖住事件跟随与已有游标。
    void this.directory.refresh({ force: true }).catch(error => this.logger.warn(`directory refresh failed: ${error.message}`))

    let backoff = 1_000
    while (!signal.aborted && !this.stopping) {
      try {
        const status = await this.client.status().catch(() => null)
        this.serverId = String(status?.serverId || '')
        await this.client.streamEvents({
          after: this.cursor,
          signal,
          onConnected: () => Object.assign(this.listener, { ready: true, last_connected_at: new Date().toISOString(),
            consecutive_failures: 0, last_error: '' }),
          onEvent: (event) => this.consumeEvent(event),
        })
        this.listener.ready = false
        backoff = 1_000
      } catch (error) {
        this.listener.ready = false
        if (signal.aborted || this.stopping) break
        this.listener.consecutive_failures++
        this.listener.last_error = 'native_unavailable'
        this.logger.warn(`event stream closed: ${error?.message || error}; reconnecting in ${backoff}ms`)
        if (error?.code === EVENT_CURSOR_EXPIRED) {
          await this.recoverExpiredCursor({ signal })
        }
        await sleep(backoff)
        backoff = Math.min(backoff * 2, 30_000)
      }
    }
    this.logger.info('adapter stopped')
  }

  /** 补读原生仍保留的事件。只在确认过期边界后跨过已被清理的前缀。 */
  async recoverExpiredCursor({ signal } = {}) {
    let confirmedExpiry = false
    try {
      let cursor = this.cursor
      for (let page = 0; page < 200; page += 1) {
        let events
        try {
          events = await this.client.listEvents({ after: cursor, limit: 200, signal })
        } catch (error) {
          if (page !== 0 || error?.code !== EVENT_CURSOR_EXPIRED) throw error
          confirmedExpiry = true
          const lostFrom = BigInt(cursor) + 1n
          cursor = await this.findRetainedCursor(cursor, { signal })
          if (signal?.aborted || this.stopping) return
          this.cursor = cursor
          this.persistCursor()
          this.logger.warn(`event retention gap ${lostFrom}..${cursor}; resuming retained events`)
          events = await this.client.listEvents({ after: cursor, limit: 200, signal })
        }
        const { items, nextCursor } = events
        try {
          for (const event of items) {
            if (signal?.aborted || this.stopping) return
            await this.consumeEvent(event)
          }
        } catch (error) {
          // 外层循环会退避重连；保留已经处理的进度，下一次重新读取失败事件。
          this.logger.warn(`cursor recovery paused: ${error.message}`)
          return
        }
        if (items.length < 200 || nextCursor === cursor) break
        cursor = nextCursor
      }
      this.logger.info(`cursor recovered by paging to ${this.cursor}`)
    } catch (error) {
      if (signal?.aborted || this.stopping) return
      if (confirmedExpiry || error?.code === EVENT_CURSOR_EXPIRED || this.deferAutomaticReplies || this.durableIngressOnly()) {
        this.logger.error(`cursor recovery paused without skipping unread events: ${error?.message || error}`)
        return
      }
      this.logger.error(`cursor recovery failed: ${error?.message || error}; fast-forwarding past unread events`)
      this.cursor = await this.fastForwardCursor()
      this.persistCursor()
    }
  }
}

/** 延迟导入 fs，保持模块在测试中可替换 fetch 而不触碰磁盘。 */
async function readFileText(file) {
  const fs = await import('node:fs/promises')
  return fs.readFile(file, 'utf8')
}
