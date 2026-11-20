/** 将 agent-wechat 本地 HTTP API 接入现有微信消息合同及发送回执。 */
import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'

/** 使用本机原生服务的令牌，按权限访问账号目录和消息 API。 */
function nativeRequest(env) {
  const base = new URL(env.MAHJONG_AGENT_WECHAT_URL || 'http://127.0.0.1:6174')
  if (base.protocol !== 'http:' || !['127.0.0.1', '[::1]'].includes(base.hostname)
      || base.username || base.password || base.pathname !== '/' || base.search || base.hash) {
    throw new Error('agent-wechat URL must be a local loopback HTTP origin')
  }
  const tokenPath = env.MAHJONG_AGENT_WECHAT_TOKEN_PATH?.trim()
  if (!tokenPath) throw new Error('Set MAHJONG_AGENT_WECHAT_TOKEN_PATH')
  const token = fs.readFileSync(tokenPath, 'utf8').trim()
  if (!token) throw new Error('agent-wechat API token file is empty')
  /** 只向本机后台发送鉴权令牌，HTTP 跳转和错误响应都不向日志暴露原文。 */
  return async function request(route, { body, key, timeout = 4000 } = {}) {
    const headers = { Authorization: 'Bearer ' + token }
    if (body !== undefined) headers['Content-Type'] = 'application/json'
    if (key) headers['Idempotency-Key'] = key
    const response = await fetch(new URL(route, base), {
      method: body === undefined ? 'GET' : 'POST',
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      redirect: 'error',
      signal: AbortSignal.timeout(timeout),
    })
    if (!response.ok) {
      const error = new Error('agent-wechat HTTP ' + response.status)
      error.status = response.status
      const details = await response.json().catch(() => null)
      error.code = details?.error?.code
      throw error
    }
    return response.json()
  }
}

/** 保留脚本启动时的精确会话白名单。 */
function ids(value) {
  return (value || '').split(',').map(x => x.trim()).filter(Boolean)
}

/** 绑定传输到单个账号及一份不可变的接入配置。 */
export function createAgentWechatTransport(env, dataDir, setup = null) {
  const request = nativeRequest(env)
  const account = setup?.account || env.MAHJONG_AGENT_WECHAT_ACCOUNT?.trim()
  if (!account) throw new Error('Set MAHJONG_AGENT_WECHAT_ACCOUNT')
  const prefix = '/v1/accounts/' + encodeURIComponent(account)
  const contacts = new Set(setup ? setup.contacts : ids(env.MAHJONG_AGENT_WECHAT_CONTACT_IDS))
  const rooms = new Set(setup ? Object.keys(setup.rooms) : ids(env.MAHJONG_AGENT_WECHAT_ROOM_IDS))
  const pendingDirectory = path.join(dataDir, 'agent-wechat-pending-sends', encodeURIComponent(account))
  const terminalStates = ['accepted', 'failed', 'cancelled']

  /** 私聊历史按微信账号隔离，已配置的原账号继续使用已有会话和牌局。 */
  function conversation(threadId, isRoom) {
    if (isRoom) return 'wechaty:room:' + (setup ? 'bridge:' + encodeURIComponent(account) + ':' : '') + threadId
    const owner = setup && account !== env.MAHJONG_AGENT_WECHAT_ACCOUNT?.trim()
      ? encodeURIComponent(account) + ':' : ''
    return 'wechaty:contact:' + owner + threadId
  }

  /** 只允许精确 ID，避免同名联系人、同名群或跨通道别名导致误发。 */
  function allowedTarget(threadId, isRoom) {
    return typeof threadId === 'string' && threadId.endsWith('@chatroom') === isRoom
      && (isRoom ? rooms : contacts).has(threadId)
  }

  /** 保留顾客正文和群引用，跳过自发回声、系统消息及未获准的会话。 */
  async function readIncoming(event) {
    const threadId = event.data?.threadId
    const id = event.data?.messageId
    if (event.kind !== 'message.committed' || typeof id !== 'string' || typeof threadId !== 'string') return null
    const isRoom = threadId.endsWith('@chatroom')
    if (!allowedTarget(threadId, isRoom)) return null
    const floors = setup?.thread_cursors
    if (floors && Object.hasOwn(floors, threadId) && BigInt(event.cursor) <= BigInt(floors[threadId])) return null
    const message = await request(prefix + '/messages/' + encodeURIComponent(id)).catch(error => {
      // daemon 的消息接口用 media_not_cached 表示本地消息行缺失；通用 404 或资料故障仍保留游标。
      error.terminalMessage = (error.status === 404 && error.code === 'media_not_cached') ||
        (error.status === 410 && error.code === 'message_recalled')
      throw error
    })
    if (message.accountId !== account || message.threadId !== threadId || message.id !== id) {
      throw new Error('agent-wechat event and message identity differ')
    }
    if (message.direction !== 'incoming' || message.recall || !message.senderId
        || (!isRoom && message.senderId !== threadId)) return null
    if (message.type !== 1 && !(message.type === 49 && message.appMessageType === 57)) return null
    if (typeof message.body !== 'string' || !message.body.trim()) return null
    const senderId = message.senderId
    const contact = await request(prefix + '/contacts/' + encodeURIComponent(senderId)).catch(error => {
      if (error.status === 404) return null
      throw error
    })
    const room = isRoom ? await request(prefix + '/groups/' + encodeURIComponent(threadId)).catch(error => {
      if (error.status === 404) return null
      throw error
    }) : null
    const senderName = contact?.nickname || senderId
    const conversationId = conversation(threadId, isRoom)
    const messageId = message.serverId || message.id
    return {
      channel: 'wechaty',
      platform_name: 'agent-wechat',
      agent_wechat_account: account,
      agent_wechat_revision: setup?.revision || 'initial',
      captured_at: new Date().toISOString(),
      conversation_id: conversationId,
      message_id: messageId,
      source_message_id: messageId,
      message_type: 7,
      is_room: isRoom,
      room: isRoom ? { id: threadId, topic: room?.displayName || room?.name || threadId } : null,
      sender_id: senderId,
      sender_name: senderName,
      // 白名单只授权接收消息，好友关系由原生资料证明；unknown 不能升级为好友。
      talker: { id: senderId, name: senderName, friend: contact?.relation === 'friend' },
      text: message.body,
      self_message: false,
      quoted_message: message.quote?.serverId ? {
        message_id: message.quote.serverId,
        sender_id: message.quote.senderId,
        sender_name: message.quote.senderName,
        text: message.quote.embeddedText,
        conversation_id: conversationId,
      } : null,
    }
  }

  /** 旧配置的进度以最后一次持久确认的游标为准。 */
  function cursorPath(serverId) {
    const identity = setup ? [serverId, account, setup.revision] : [serverId, account]
    const namespace = createHash('sha256').update(JSON.stringify(identity)).digest('hex')
    return path.join(dataDir, 'agent-wechat-' + namespace + '.cursor')
  }

  /** 读取确认进度，首次启动只允许沿用显式设置的起点。 */
  function consumedCursor(serverId) {
    let cursor
    try {
      cursor = fs.readFileSync(cursorPath(serverId), 'utf8').trim()
    } catch (error) {
      if (error.code !== 'ENOENT') throw error
      cursor = setup?.cursor ?? env.MAHJONG_AGENT_WECHAT_START_CURSOR?.trim()
    }
    if (cursor === undefined) throw new Error('Set MAHJONG_AGENT_WECHAT_START_CURSOR on first start; 0 explicitly replays history')
    if (!/^\d+$/.test(cursor)) throw new Error('Invalid agent-wechat event cursor')
    return cursor
  }

  /** 按顺序消费持久事件，业务接收成功后才推进本地游标，故障时从旧位置恢复。 */
  async function watch(onMessage, signal) {
    const status = await request('/v1/status')
    const accountStatus = await request(prefix)
    if (!status.serverId || accountStatus.id !== account || (setup && status.serverId !== setup.server_id)) {
      throw new Error('agent-wechat account is unavailable')
    }
    if (signal.aborted) return
    fs.mkdirSync(dataDir, { recursive: true })
    const cursorFile = cursorPath(status.serverId)
    let cursor = consumedCursor(status.serverId)
    if (!fs.existsSync(cursorFile)) fs.writeFileSync(cursorFile, cursor, { mode: 0o600, flag: 'wx' })
    let retry = 1000
    while (!signal.aborted) {
      try {
        const page = await request(prefix + '/events?after=' + cursor + '&limit=100')
        if (!Array.isArray(page.items)) throw new Error('Invalid agent-wechat event page')
        for (const event of page.items) {
          if (signal.aborted) return
          if (!/^\d+$/.test(event.cursor) || BigInt(event.cursor) <= BigInt(cursor)) {
            throw new Error('Invalid agent-wechat event order')
          }
          let message
          try {
            message = await readIncoming(event)
          } catch (error) {
            if (!error.terminalMessage) throw error
            // 审计写入成功后才推进，避免把未落证据的终止事件丢掉。
            fs.appendFileSync(path.join(dataDir, 'agent-wechat-skipped-events.jsonl'), JSON.stringify({
              at: new Date().toISOString(), account, cursor: event.cursor,
              message_id: event.data.messageId, reason: error.code,
            }) + '\n', { mode: 0o600 })
          }
          if (signal.aborted) return
          if (message) await onMessage(message)
          fs.writeFileSync(cursorFile + '.tmp', event.cursor, { mode: 0o600 })
          fs.renameSync(cursorFile + '.tmp', cursorFile)
          cursor = event.cursor
        }
        retry = 1000
        if (page.items.length < 100) await delay(1000, undefined, { signal })
      } catch (error) {
        if (signal.aborted) return
        console.error('agent-wechat event processing deferred:', error.message)
        await delay(retry)
        retry = Math.min(retry * 2, 15000)
      }
    }
  }

  /** 使用服务端持久幂等键，只有取得微信确认及平台消息 ID 才返回成功。 */
  async function send(threadId, text, isRoom, deliveryKey) {
    if (!allowedTarget(threadId, isRoom)) throw new Error('Target is outside the agent-wechat allowlist')
    const digest = createHash('sha256').update(
      JSON.stringify([account, threadId, deliveryKey || randomUUID()]),
    ).digest('hex')
    const key = 'mahjong:' + digest
    const pendingPath = path.join(pendingDirectory, digest + '.json')
    fs.mkdirSync(pendingDirectory, { recursive: true, mode: 0o700 })
    const pending = JSON.stringify({ threadId, key })
    try {
      fs.writeFileSync(pendingPath, pending, { mode: 0o600, flag: 'wx', flush: true })
    } catch (error) {
      if (error.code !== 'EEXIST' || fs.readFileSync(pendingPath, 'utf8') !== pending) throw error
    }
    const deadline = Date.now() + 2200
    let receipt = await request(prefix + '/sends', { body: { threadId, text }, key, timeout: 2200 })
    while (['queued', 'submitting'].includes(receipt.state) && Date.now() < deadline) {
      if (!receipt.id) throw new Error('agent-wechat returned no send ID')
      await delay(125)
      receipt = await request(prefix + '/sends/' + encodeURIComponent(receipt.id), {
        timeout: Math.max(100, deadline - Date.now()),
      })
    }
    if (receipt.destination !== threadId || receipt.accountId !== account) {
      throw new Error('agent-wechat returned an unexpected send destination')
    }
    if (terminalStates.includes(receipt.state)) fs.rmSync(pendingPath, { force: true })
    if (receipt.state !== 'accepted' || typeof receipt.serverId !== 'string' || !receipt.serverId) {
      throw new Error('agent-wechat send not confirmed: ' + (receipt.state || 'unknown'))
    }
    return { id: receipt.serverId }
  }

  /** 持久记录覆盖 HTTP 回执丢失和重启。回查无确证时保留原账号。 */
  async function settlePendingSends() {
    if (!fs.existsSync(pendingDirectory)) return
    for (const file of fs.readdirSync(pendingDirectory)) {
      const pendingPath = path.join(pendingDirectory, file)
      const { threadId, key } = JSON.parse(fs.readFileSync(pendingPath, 'utf8'))
      const matches = receipt => receipt.accountId === account && receipt.destination === threadId
      let receipt = await request(prefix + '/sends/by-idempotency-key?' + new URLSearchParams({ key }))
      if (matches(receipt) && receipt.state === 'queued' && receipt.id) {
        receipt = await request(prefix + '/sends/' + encodeURIComponent(receipt.id) + '/cancel', { body: {} })
      }
      if (!matches(receipt) || !terminalStates.includes(receipt.state)) {
        throw new Error('Previous account has an unresolved send')
      }
      fs.rmSync(pendingPath, { force: true })
    }
  }

  return { watch, send, conversation, account, consumedCursor, allowedTarget, settlePendingSends, revision: setup?.revision || 'initial' }
}

/** 目录只投影可接入的普通私聊和群聊，避免把系统和公众号会话交给 Agent。 */
function candidate(item, source, ownUsername) {
  const id = item?.id
  const category = item?.classification?.category
  if (!validId(id) || id === ownUsername) return null
  const group = id.endsWith('@chatroom')
  if (group ? source === 'contacts' || (source === 'threads' && item.kind !== 'group')
    : source === 'groups' || (source === 'threads' && item.kind !== 'direct') || (category && category !== 'unknown')) return null
  // 微信目录可能用控制字符占位。只清理显示名称，保留精确的投递 ID。
  const name = [item.displayName, item.nickname].filter(value => typeof value === 'string')
    .map(value => value.replace(/\p{Cc}/gu, '').trim())
    .find(value => value !== id && /[^\p{Cf}\p{White_Space}]/u.test(value)) || id
  return { id, name, kind: group ? 'group' : 'private', has_thread: source === 'threads' }
}

/** 原生账号和会话 ID 按原值匹配，拒绝不能安全编码成路径段的输入。 */
function validId(value) {
  return typeof value === 'string' && value.length > 0 && value.length <= 256 &&
    value !== '.' && value !== '..' && !/[\x00-\x20\x7f/\\]/.test(value)
}

/** 将可操作的失败原因交给本地控制台，原生响应正文留在服务端。 */
function setupError(status, message) {
  return Object.assign(new Error(message), { setupStatus: status })
}

/** 保存当前接入范围并在同一桥接进程中切换监听器。 */
export function createAgentWechatSetup(env, dataDir, configPath, onAccountChanged) {
  const request = nativeRequest(env)
  const legacyAccount = env.MAHJONG_AGENT_WECHAT_ACCOUNT?.trim()
  let active = null
  try {
    active = JSON.parse(fs.readFileSync(configPath, 'utf8'))
    if (!validId(active?.account) || !active.revision || !active.server_id || !/^\d+$/.test(active.cursor)
        || !Array.isArray(active.contacts) || !active.rooms) throw new Error('Invalid WeChat setup configuration')
  } catch (error) {
    if (error.code !== 'ENOENT') throw error
  }
  let transport = createAgentWechatTransport(env, dataDir, active)
  let reconfigure = new AbortController()
  let applying = false
  let switching = false
  let operations = 0

  /** 原生账号目录按令牌权限过滤，页面只接收状态和用于展示的名称。 */
  async function status() {
    const { items } = await request('/v1/accounts')
    return {
      ok: true, active_account: active?.account ?? legacyAccount,
      revision: active?.revision ?? 'initial', applied_at: active?.applied_at ?? null,
      selections: active?.selections ?? [
        ...ids(env.MAHJONG_AGENT_WECHAT_CONTACT_IDS).map(id => ({ id, name: id, mode: 'private' })),
        ...ids(env.MAHJONG_AGENT_WECHAT_ROOM_IDS).map(id => ({ id, name: id, mode: null })),
      ],
      accounts: items.map(item => ({ id: item.id, name: item.alias || item.id,
        online: item.authenticated === true && item.connection === 'online', synced: item.contactSyncReady === true })),
    }
  }

  /** 账号详情同时校验账号范围，可浏览本地已同步的离线目录。 */
  async function accountOf(account) {
    if (!validId(account)) throw setupError(400, '请选择微信账号')
    const prefix = '/v1/accounts/' + encodeURIComponent(account)
    const item = await request(prefix)
    if (item.id !== account) throw setupError(404, '找不到该微信账号')
    return { prefix, item }
  }

  /** 跟随原生目录分页，合并联系人、群资料及已有会话，按精确 ID 去重。 */
  async function inventory(account) {
    const { prefix, item } = await accountOf(account)
    const sources = ['threads', 'contacts', 'groups']
    const lists = await Promise.all(sources.map(async source => {
      const rows = []
      let cursor = ''
      const seen = new Set()
      do {
        const query = new URLSearchParams({ limit: '200' })
        if (cursor) query.set('cursor', cursor)
        const page = await request(prefix + '/' + source + '?' + query)
        for (const row of page.items) {
          const target = candidate(row, source, item.ownUsername)
          if (target) rows.push(target)
        }
        cursor = page.nextCursor || ''
        if (cursor && seen.has(cursor)) throw setupError(502, '读取会话目录失败，请刷新重试')
        seen.add(cursor)
      } while (cursor)
      return rows
    }))
    const entries = new Map()
    for (const row of lists.flat()) {
      const prior = entries.get(row.id)
      entries.set(row.id, { ...row, name: row.name === row.id ? prior?.name || row.id : row.name,
        has_thread: row.has_thread || prior?.has_thread === true })
    }
    return { ok: true, account, items: [...entries.values()] }
  }

  /** 核对实时账号和目标，写入原子配置后才停旧监听和启用新监听。 */
  async function apply(payload) {
    if (applying) throw setupError(409, '接入配置正在应用，请稍后重试')
    if (payload?.revision !== (active?.revision ?? 'initial')) throw setupError(409, '接入配置已更新，请刷新页面')
    if (!Array.isArray(payload.selections)) throw setupError(400, '会话范围格式有误')
    const selected = new Map()
    for (const row of payload.selections) {
      if (!validId(row?.id) || selected.has(row.id) || !(row.id.endsWith('@chatroom')
        ? ['observe', 'managed'].includes(row.mode) : row.mode === 'private') ||
        ('auto_reply' in row && typeof row.auto_reply !== 'boolean') || (row.auto_reply && row.mode !== 'managed')) {
        throw setupError(400, '会话范围或接入模式有误')
      }
      selected.set(row.id, { mode: row.mode, ...(row.mode === 'managed' ? { auto_reply: row.auto_reply === true } : {}) })
    }
    if (active?.account === payload.account && selected.size === active.selections.length &&
        active.selections.every(row => selected.get(row.id)?.mode === row.mode &&
          Boolean(selected.get(row.id)?.auto_reply) === Boolean(row.auto_reply))) {
      return { ok: true, active_account: active.account, account_changed: false,
        revision: active.revision, applied_at: active.applied_at, selections: active.selections }
    }
    applying = true
    try {
      const { prefix, item } = await accountOf(payload.account)
      if (!item.authenticated || item.connection !== 'online') throw setupError(409, '该账号当前离线，请在 AgentWechat 登录后重试')
      const catalog = await inventory(payload.account)
      const selections = catalog.items.filter(row => selected.has(row.id))
        .map(row => ({ ...row, ...selected.get(row.id) }))
      if (selections.length !== selected.size) throw setupError(409, '部分会话已无法选用，请刷新目录')
      const [server, head] = await Promise.all([request('/v1/status'), request(prefix + '/threads?limit=1')])
      if (!server.serverId || typeof head.asOfEventCursor !== 'string' || !/^\d+$/.test(head.asOfEventCursor)) {
        throw setupError(502, '无法确定新消息的起点，请重试')
      }
      if (active && active.server_id !== server.serverId) {
        throw setupError(409, 'AgentWechat 服务身份已改变，请核对本机服务及账号')
      }
      await request(prefix + '/events?after=' + head.asOfEventCursor + '&limit=1')
      if (operations) throw setupError(409, '当前账号正在处理消息，请稍后重试')
      const accountChanged = item.id !== transport.account
      const sameStream = !accountChanged && (!active || active.server_id === server.serverId)
      switching = true
      try {
        await transport.settlePendingSends()
      } catch {
        throw setupError(409, '当前账号还有发送任务待确认，请核对后重试')
      }
      const retained = row => sameStream && transport.allowedTarget(row.id, row.kind === 'group')
      const continuing = selections.some(retained)
      const config = {
        account: item.id, server_id: server.serverId, revision: randomUUID(),
        cursor: continuing ? transport.consumedCursor(server.serverId) : head.asOfEventCursor,
        thread_cursors: Object.fromEntries(selections.flatMap(row => {
          const previous = active?.selections.find(prior => prior.id === row.id)
          const newlyAuthorized = previous && ((row.mode === 'managed' && previous.mode !== 'managed') ||
            (row.auto_reply && !previous.auto_reply))
          const floor = retained(row) && !newlyAuthorized
            ? active?.thread_cursors?.[row.id] ?? active?.cursor : head.asOfEventCursor
          return floor === undefined ? [] : [[row.id, floor]]
        })),
        applied_at: new Date().toISOString(), selections,
        contacts: selections.filter(row => row.mode === 'private').map(row => row.id),
        rooms: Object.fromEntries(selections.filter(row => row.kind === 'group').map(row => [row.id, row.mode])),
      }
      const next = createAgentWechatTransport(env, dataDir, config)
      fs.mkdirSync(path.dirname(configPath), { recursive: true })
      const temporary = configPath + '.' + config.revision + '.tmp'
      fs.writeFileSync(temporary, JSON.stringify(config), { mode: 0o600, flag: 'wx' })
      try {
        fs.renameSync(temporary, configPath)
      } finally {
        fs.rmSync(temporary, { force: true })
      }
      active = config
      transport = next
      // 两个外发开关属于当前微信账号。调整同一账号的会话范围时沿用操作员的选择。
      if (accountChanged) onAccountChanged()
      reconfigure.abort()
      reconfigure = new AbortController()
      return { ok: true, active_account: config.account, account_changed: accountChanged, revision: config.revision,
        applied_at: config.applied_at, selections: config.selections }
    } finally {
      switching = false
      applying = false
    }
  }

  /** 等旧监听退出再接管新游标。账号离线或 daemon 重启时继续等待恢复。 */
  async function watch(onMessage, signal) {
    while (!signal.aborted) {
      const current = AbortSignal.any([signal, reconfigure.signal])
      try {
        await transport.watch(async message => {
          if (switching) throw new Error('WeChat setup is switching')
          operations++
          try { await onMessage(message) } finally { operations-- }
        }, current)
      } catch (error) {
        console.error('agent-wechat listener unavailable:', error.message)
        await delay(2000, undefined, { signal: current }).catch(() => {})
      }
    }
  }

  /** 等待微信发送确认期间阻止切换账号；仅观察群不能外发。 */
  async function send(id, message, isRoom, key, options) {
    if (switching) throw new Error('WeChat setup is switching')
    const automated = options.source && !['manual_operator', 'human_approved_invite'].includes(options.source)
    if ((automated || options.exact_target === true) &&
        (options.agent_wechat_account !== transport.account || options.agent_wechat_revision !== transport.revision)) {
      throw new Error('WeChat reply belongs to another account or setup revision')
    }
    // 页面只授予同一配置版本内、由入站消息触发的群内答复权限。看板推送和跨会话通知仍走原策略。
    const configuredReply = active && isRoom && options.message_type === 'agent_reply' && options.source_message_id &&
      options.agent_wechat_revision === active.revision && active.selections.some(row => row.id === id && row.auto_reply === true)
    if (active && ((isRoom && active.rooms[id] !== 'managed') || (options.source === 'managed_group_domain' && !configuredReply))) {
      throw new Error('The active WeChat setup does not permit this send')
    }
    operations++
    try {
      const conversationId = transport.conversation(id, isRoom)
      const receipt = await transport.send(id, message, isRoom, key)
      return { ...receipt, conversationId, account: transport.account, revision: transport.revision }
    } finally {
      operations--
    }
  }

  return { status, inventory, apply, watch, send, get configured() { return active !== null },
    get binding() { return { agent_wechat_account: transport.account, agent_wechat_revision: transport.revision } } }
}
