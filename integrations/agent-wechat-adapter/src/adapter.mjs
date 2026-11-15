/**
 * 通道主循环：跟随上游事件、翻译消息、投递 Runtime、按开关回发。
 *
 * 关键边界：
 * - 只通过 HTTP/SSE 读取上游，不访问 daemon 的数据库、锁或本地文件；
 * - 每条上游消息处理完才推进游标，Runtime 不可用时先落重放队列再推进；
 * - 外发默认关闭，且 needs_review 一律不自动重发。
 */
import { buildChannelPayload, replyIdempotencyKey, shouldForwardMessage } from './normalize.mjs'
import { appendDelivery, loadCursor, loadDeliveryLedger, persistToSpool, listSpool, removeSpoolFile } from './state.mjs'

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

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

/** 向 Runtime raw 入口投递一条载荷；4xx 不重试，网络/5xx 指数退避。 */
export async function postAgentRaw(endpoint, token, payload, { fetchImpl = globalThis.fetch, attempts = 3, logger = console } = {}) {
  let lastError = null
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const headers = { 'Content-Type': 'application/json; charset=utf-8' }
      if (token) headers['X-Mahjong-Agent-Token'] = token
      const response = await fetchImpl(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      })
      const text = await response.text()
      let parsed = null
      try {
        parsed = JSON.parse(text)
      } catch {
        parsed = { raw_response: text }
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

export class AgentWechatAdapter {
  constructor({ client, directory, config, switches = null, logger = createLogger(config?.logLevel), fetchImpl = globalThis.fetch }) {
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
    this.delivered = loadDeliveryLedger(config.deliveryLedgerPath)
    this.cursor = '0'
    this.account = null
    this.selfUsername = ''
    this.stopping = false
    this.bootstrapped = false
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
      const { items, nextCursor } = await this.client.listEvents({ after: cursor, limit: 200 })
      total += items.length
      if (!items.length || nextCursor === cursor) break
      cursor = nextCursor
      if (items.length < 200) break
    }
    this.logger.info(`fast-forward scanned ${total} historical events`)
    return cursor
  }

  persistCursor() {
    try {
      saveCursorSafe(this.config.cursorPath, {
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

  /** 处理一条 message.committed：拉取消息 -> 过滤 -> 投递 -> 按开关回发。 */
  async handleEvent(event) {
    if (event?.kind !== 'message.committed') return
    const messageId = String(event?.data?.messageId || '').trim()
    if (!messageId) return

    const message = await this.client.getMessage(messageId)
    if (!message) return
    if (
      !shouldForwardMessage(message, {
        forwardSelfMessages: this.config.forwardSelfMessages,
        forwardSystemMessages: this.config.forwardSystemMessages,
      })
    ) {
      this.logger.debug(`skip message=${messageId} type=${message.type} direction=${message.direction}`)
      return
    }

    const payload = await this.buildPayload(message)
    if (this.config.dryRun) {
      // 只读演练：只打印将要投递的摘要，不写 Runtime、不回发。
      this.logger.info(
        `dry-run message=${payload.message_id} conversation=${payload.conversation_id} type=${payload.message_type} text_len=${payload.text.length}`,
      )
      return
    }
    let result = null
    try {
      result = await postAgentRaw(this.config.agentEndpoint, this.config.agentToken, payload, {
        fetchImpl: this.fetchImpl,
        logger: this.logger,
      })
      this.logger.info(
        `forwarded message=${payload.message_id} thread=${message.threadId} trace=${result?.trace_id || '-'}`,
      )
    } catch (error) {
      const file = persistToSpool(this.config.spoolDir, payload, error)
      this.logger.warn(`queued for replay file=${file}: ${error?.message || error}`)
    }
    if (result) await this.maybeSendReply(payload, result)
  }

  /** 回发 Runtime 的最终回复；默认关闭，失败只记录不重发。 */
  async maybeSendReply(payload, result) {
    const finalReply = extractFinalReply(result)
    if (!finalReply) return null
    if (this.config.dryRun || !this.switches.sendChannelEnabled || !this.switches.autoSendReply) {
      this.logger.info(`reply withheld because send switches are off trace=${result?.trace_id || '-'}`)
      return null
    }
    const sourceMessageId = String(payload?.source_message_id || payload?.message_id || '')
    if (sourceMessageId && this.delivered.has(sourceMessageId)) {
      this.logger.info(`reply skipped; already delivered source=${sourceMessageId}`)
      return null
    }
    try {
      const idempotencyKey = replyIdempotencyKey(sourceMessageId)
      const intent = await this.client.sendText({
        threadId: payload.payload.threadId,
        text: finalReply,
        idempotencyKey,
      })
      const state = await this.client.waitForSend(intent.id, { timeoutMs: this.config.sendWaitTimeoutMs })
      return this.recordSendOutcome({ payload, intent, state, idempotencyKey })
    } catch (error) {
      this.logger.error(`reply send failed source=${sourceMessageId}: ${error?.message || error}`)
      return null
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
      this.delivered.add(String(payload.source_message_id))
      appendDelivery(this.config.deliveryLedgerPath, {
        source_message_id: String(payload.source_message_id || ''),
        conversation_id: String(payload.conversation_id || ''),
        send_id: String(state?.id || intent?.id || ''),
        message_id: messageId,
        idempotency_key: idempotencyKey,
        state: finalState,
      })
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
        })
        await this.maybeSendReply(record.payload, result)
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
    await this.directory.refresh({ force: true })

    let backoff = 1_000
    while (!signal.aborted && !this.stopping) {
      try {
        const status = await this.client.status().catch(() => null)
        this.serverId = String(status?.serverId || '')
        await this.client.streamEvents({
          after: this.cursor,
          signal,
          onEvent: async (event) => {
            try {
              await this.handleEvent(event)
            } catch (error) {
              this.logger.error(`event handling failed cursor=${event?.cursor}: ${error?.message || error}`)
            }
            this.cursor = String(event?.cursor || this.cursor)
            this.persistCursor()
          },
        })
        backoff = 1_000
      } catch (error) {
        if (signal.aborted || this.stopping) break
        this.logger.warn(`event stream closed: ${error?.message || error}; reconnecting in ${backoff}ms`)
        if (String(error?.code || '').includes('cursor_expired')) {
          await this.recoverExpiredCursor()
        }
        await sleep(backoff)
        backoff = Math.min(backoff * 2, 30_000)
      }
    }
    this.logger.info('adapter stopped')
  }

  /** 游标过期时先按 JSON 分页追平；仍失败则快进到尾部并明确记录丢弃窗口。 */
  async recoverExpiredCursor() {
    try {
      let cursor = this.cursor
      for (let page = 0; page < 200; page += 1) {
        const { items, nextCursor } = await this.client.listEvents({ after: cursor, limit: 200 })
        for (const event of items) {
          await this.handleEvent(event)
          cursor = String(event.cursor)
        }
        if (!items.length || nextCursor === cursor) break
        cursor = nextCursor
      }
      this.cursor = cursor
      this.persistCursor()
      this.logger.info(`cursor recovered by paging to ${this.cursor}`)
    } catch (error) {
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

/** 延迟调用游标写入，避免模块加载时触碰测试临时目录。 */
function saveCursorSafe(path, value) {
  // 复用 state.mjs 的原子写入；单独包一层便于测试时替换。
  return import('./state.mjs').then(({ saveCursor }) => saveCursor(path, value))
}
