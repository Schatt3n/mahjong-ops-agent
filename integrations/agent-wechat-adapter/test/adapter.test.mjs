import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { AgentWechatAdapter, postAgentRaw } from '../src/adapter.mjs'
import { AgentWechatClient } from '../src/client.mjs'
import { loadCursor, saveCursor } from '../src/state.mjs'

const silentLogger = { debug() {}, info() {}, warn() {}, error() {} }

/** 构造测试配置，所有状态文件都落在临时目录。 */
function makeConfig(dir, overrides = {}) {
  return {
    agentEndpoint: 'http://runtime.test/api/channels/agent-wechat/raw',
    agentToken: 'runtime-token',
    stateDir: dir,
    cursorPath: path.join(dir, 'events_cursor.json'),
    deliveryLedgerPath: path.join(dir, 'delivery_ledger.jsonl'),
    spoolDir: path.join(dir, 'spool'),
    sendEnabled: false,
    autoSendReply: false,
    forwardSelfMessages: false,
    forwardSystemMessages: false,
    replayHistory: false,
    startCursor: '',
    dryRun: false,
    sendWaitTimeoutMs: 1_000,
    logLevel: 'error',
    ...overrides,
  }
}

/** 构造最小目录替身，只返回测试需要的展示信息。 */
function fakeDirectory(thread = { displayName: '周末局' }, contact = { remark: '老王' }) {
  return {
    contacts: [],
    groups: [],
    getThread: async () => thread,
    contact: () => contact,
    refresh: async () => {},
    resolve: async () => ({ candidates: [], exactCount: 0, ambiguous: true, recommendedId: null }),
  }
}

/** 构造一条入站文本消息，默认是群聊客户消息。 */
function incoming(id = 'm1', overrides = {}) {
  return {
    id,
    threadId: '123@chatroom',
    senderId: 'wxid_a',
    body: '晚上三缺一',
    type: 1,
    appMessageType: null,
    direction: 'incoming',
    ...overrides,
  }
}

/** 组装适配器与可控客户端，便于分别覆盖入站和外发分支。 */
function makeAdapter({ dir, messages, fetchImpl, clientOverrides = {}, configOverrides = {}, switches = {}, deferAutomaticReplies = false }) {
  const client = {
    accountId: 'acc',
    listAccounts: async () => [{ id: 'acc', ownUsername: 'me' }],
    getMessage: async (id) => messages.get(id),
    getThread: async () => ({ displayName: '周末局' }),
    sendText: async () => ({ id: 'send-1' }),
    waitForSend: async () => ({ id: 'send-1', state: 'accepted', localMessageId: 'wx-out-1' }),
    ...clientOverrides,
  }
  const config = makeConfig(dir, configOverrides)
  const adapter = new AgentWechatAdapter({
    client,
    directory: fakeDirectory(),
    config,
    switches: { sendChannelEnabled: false, autoSendReply: false, ...switches },
    logger: silentLogger,
    fetchImpl,
    deferAutomaticReplies,
  })
  adapter.account = { id: 'acc', ownUsername: 'me' }
  adapter.selfUsername = 'me'
  return { adapter, client, config }
}

/** 创建本次测试独占的临时目录。 */
function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'agent-wechat-adapter-'))
}

const okFetch = async () =>
  new Response(JSON.stringify({ ok: true, trace_id: 'trace-1', route_result: { agent_result: { final_reply: '收到' } } }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })

test('正常入站投递到 Runtime', async () => {
  const calls = []
  const { adapter } = makeAdapter({
    dir: tempDir(),
    messages: new Map([['m1', incoming()]]),
    fetchImpl: async (url, init) => {
      calls.push({ url, body: JSON.parse(init.body) })
      return Response.json({ ok: true, accepted: true }, { status: 202 })
    },
  })
  await adapter.handleEvent({ kind: 'message.committed', data: { messageId: 'm1' } })
  assert.equal(calls.length, 1)
  assert.equal(calls[0].body.conversation_id, 'wechaty:room:123@chatroom')
  assert.equal(calls[0].body.channel, 'agent-wechat')
})

test('只入站时持久接收即可推进游标且不调用外发路径', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const { adapter, config } = makeAdapter({
    dir,
    messages: new Map([['m1', incoming()]]),
    fetchImpl: async (_url, init) => {
      assert.equal(init.headers.Prefer, 'respond-async')
      return Response.json({ ok: true, accepted: true }, { status: 202 })
    },
  })
  adapter.maybeSendReply = async () => assert.fail('持久接收确认不能触发外发')
  await adapter.consumeEvent({ cursor: '1', kind: 'message.committed', data: { messageId: 'm1' } })
  assert.equal(loadCursor(config.cursorPath)?.cursor, '1')
})

test('自动回复开启后，持久接收确认即可推进游标，结果交给后台外发', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const calls = []
  const { adapter, config } = makeAdapter({
    dir,
    messages: new Map([
      ['first', incoming('first', { threadId: 'wxid_a' })],
      ['second', incoming('second', { threadId: 'wxid_b', senderId: 'wxid_b' })],
    ]),
    switches: { sendChannelEnabled: true, autoSendReply: true },
    deferAutomaticReplies: true,
    fetchImpl: async (_url, init) => {
      const headers = new Headers(init.headers)
      assert.equal(headers.get('Prefer'), 'respond-async')
      assert.equal(headers.get('X-Mahjong-Agent-Wechat-Deferred-Reply'), '1')
      calls.push(JSON.parse(init.body).message_id)
      return Response.json({ ok: true, accepted: true, deferred_reply: true }, { status: 202 })
    },
  })
  adapter.maybeSendReply = async () => assert.fail('外发结果必须由后台交接')
  await adapter.consumeEvent({ cursor: '1', kind: 'message.committed', data: { messageId: 'first' } })
  await adapter.consumeEvent({ cursor: '2', kind: 'message.committed', data: { messageId: 'second' } })
  assert.deepEqual(calls, ['first', 'second'])
  assert.equal(loadCursor(config.cursorPath)?.cursor, '2')
})

test('本人消息和关闭任一外发开关时只要求持久接收', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const forwarded = []
  const threadId = 'wxid_a'
  const { adapter, config } = makeAdapter({
    dir,
    messages: new Map([
      ['owner', incoming('owner', { threadId, direction: 'outgoing', senderId: 'me' })],
      ['send-off', incoming('send-off', { threadId })],
      ['auto-off', incoming('auto-off', { threadId })],
    ]),
    switches: { sendChannelEnabled: true, autoSendReply: true },
    deferAutomaticReplies: true,
    fetchImpl: async (_url, init) => {
      const headers = new Headers(init.headers)
      assert.equal(headers.get('Prefer'), 'respond-async')
      assert.equal(headers.get('X-Mahjong-Agent-Wechat-Deferred-Reply'), null)
      forwarded.push(JSON.parse(init.body))
      return Response.json({ ok: true, accepted: true }, { status: 202 })
    },
  })
  adapter.maybeSendReply = async () => assert.fail('只入站的消息不能触发自动外发')
  await adapter.consumeEvent({
    cursor: '1', kind: 'message.committed',
    data: { messageId: 'owner', origin: 'live', threadId, direction: 'outgoing' },
  })
  adapter.switches.sendChannelEnabled = false
  await adapter.consumeEvent({ cursor: '2', kind: 'message.committed', data: { messageId: 'send-off' } })
  adapter.switches.sendChannelEnabled = true
  adapter.switches.autoSendReply = false
  await adapter.consumeEvent({ cursor: '3', kind: 'message.committed', data: { messageId: 'auto-off' } })
  assert.deepEqual(forwarded.map(payload => payload.message_id), ['owner', 'send-off', 'auto-off'])
  assert.equal(forwarded[0].message_origin, 'owner')
  assert.equal(loadCursor(config.cursorPath)?.cursor, '3')
})

test('Runtime 未确认后台外发合同时，自动回复模式保留游标供重试', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const { adapter, config } = makeAdapter({
    dir,
    messages: new Map([['m1', incoming()]]),
    switches: { sendChannelEnabled: true, autoSendReply: true },
    deferAutomaticReplies: true,
    fetchImpl: async () => Response.json({ ok: true, accepted: true }, { status: 202 }),
  })
  adapter.maybeSendReply = async () => assert.fail('无后台确认不能直接外发')
  await assert.rejects(adapter.consumeEvent({ cursor: '1', kind: 'message.committed', data: { messageId: 'm1' } }))
  assert.equal(adapter.cursor, '0')
  assert.equal(loadCursor(config.cursorPath), null)
})

test('手机端本人消息进入上下文且不触发外发，本地发送回声仍跳过', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const received = []
  const outgoing = (id) => incoming(id, { threadId: 'wxid_a', direction: 'outgoing', senderId: 'me' })
  const { adapter } = makeAdapter({
    dir,
    messages: new Map([['owner', outgoing('owner')], ['echo', outgoing('echo')]]),
    fetchImpl: async (_url, init) => {
      received.push(JSON.parse(init.body))
      return okFetch()
    },
    clientOverrides: { sendText: async () => assert.fail('本人发言不能触发自动外发') },
    switches: { sendChannelEnabled: true, autoSendReply: true },
  })
  for (const [messageId, origin] of [['owner', 'live'], ['echo', 'local_send']]) {
    await adapter.handleEvent({ kind: 'message.committed', data: {
      messageId, origin, threadId: 'wxid_a', direction: 'outgoing',
    } })
  }
  assert.equal(received.length, 1)
  assert.equal(received[0].message_origin, 'owner')
  assert.equal(received[0].conversation_id, 'wechaty:contact:wxid_a')
})

test('同步生成期间提交的老板回复会挡住旧回复，客户新一轮仍可回复', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const threadId = 'wxid_a'
  const before = '2026-11-18T10:00:00Z'
  const after = '2026-11-18T10:00:01Z'
  const { adapter } = makeAdapter({
    dir,
    messages: new Map([
      ['first', incoming('first', { threadId, createdAt: before })],
      ['next', incoming('next', { threadId, createdAt: after })],
    ]),
    fetchImpl: okFetch,
    switches: { sendChannelEnabled: true, autoSendReply: true },
  })
  const posts = []
  const upstream = new AgentWechatClient({
    baseUrl: 'http://upstream.test',
    accountId: 'acc',
    fetchImpl: async (url, init) => {
      const request = new URL(url)
      if (request.pathname.endsWith('/events')) {
        const items = request.searchParams.get('after') === '100' ? [
          { kind: 'message.committed', data: { direction: 'outgoing', origin: 'local_send', threadId, messageId: 'echo' } },
          { kind: 'message.committed', data: { direction: 'outgoing', origin: 'live', threadId, messageId: 'owner' } },
        ] : []
        return Response.json({ items, nextCursor: '102' })
      }
      if (request.pathname.endsWith('/messages/owner')) {
        return Response.json(incoming('owner', { threadId, direction: 'outgoing', createdAt: after }))
      }
      assert.equal(init.method, 'POST')
      posts.push(JSON.parse(init.body))
      return Response.json({ id: 'send-1' })
    },
  })
  adapter.client.sendText = upstream.sendText.bind(upstream)
  await adapter.consumeEvent({ cursor: '100', kind: 'message.committed', data: { messageId: 'first' } })
  assert.equal(posts.length, 0)
  await adapter.consumeEvent({ cursor: '103', kind: 'message.committed', data: { messageId: 'next' } })
  assert.equal(posts.length, 1)
})

test('群聊只按精确引用识别老板回复，忽略其他人或旧问题的回复', async () => {
  const threadId = '123@chatroom'
  const now = '2026-11-18T10:00:00Z'
  const source = { cursor: '100', account_id: 'acc', thread_id: threadId, server_id: '1234567890123456790', created_at: now }
  const upstream = new AgentWechatClient({ baseUrl: 'http://stub.test', accountId: 'acc' })
  upstream.listEvents = async () => ({ items: [
    { kind: 'message.committed', data: { direction: 'outgoing', origin: 'live', threadId, messageId: 'owner' } },
  ], nextCursor: '101' })
  const detail = { id: 'owner', threadId, direction: 'outgoing', type: 49, createdAt: now, quote: null }
  upstream.getMessage = async () => detail
  assert.equal(await upstream.checkForOwnerReply(threadId, source), false)
  detail.quote = { serverId: '1234567890123456791' }
  assert.equal(await upstream.checkForOwnerReply(threadId, source), false)
  detail.quote.serverId = source.server_id
  assert.equal(await upstream.checkForOwnerReply(threadId, source), true)
  detail.createdAt = '2026-11-17T10:00:00Z'
  assert.equal(await upstream.checkForOwnerReply(threadId, source), false)
})

test('定向手动外发不会让入站等待模型回复', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const { adapter } = makeAdapter({
    dir,
    messages: new Map([['m1', incoming()]]),
    configOverrides: { outboundAllowlistPath: path.join(dir, 'policy.json') },
    switches: { sendChannelEnabled: true, autoSendReply: false },
    fetchImpl: async (_url, init) => {
      assert.equal(init.headers.Prefer, 'respond-async')
      return Response.json({ ok: true, accepted: true }, { status: 202 })
    },
  })
  adapter.maybeSendReply = async () => assert.fail('手动外发名单不能触发自动回复')
  await adapter.handleEvent({ kind: 'message.committed', data: { messageId: 'm1' } })
})

test('只入站时落库失败保留游标供上游按顺序重放', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const { adapter, config } = makeAdapter({
    dir,
    messages: new Map([['m1', incoming()]]),
    fetchImpl: async () => Response.json({ ok: false }, { status: 401 }),
  })
  await assert.rejects(
    adapter.consumeEvent({ cursor: '1', kind: 'message.committed', data: { messageId: 'm1' } }),
    /runtime HTTP 401/,
  )
  assert.equal(adapter.cursor, '0')
  assert.equal(loadCursor(config.cursorPath), null)
  assert.equal(fs.existsSync(config.spoolDir), false)
})

test('消息详情返回 503 时从原游标重连并补投同一条消息', { timeout: 5_000 }, async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const config = makeConfig(dir)
  saveCursor(config.cursorPath, { cursor: '40', accountId: 'acc' })
  const controller = new AbortController()
  const event = { cursor: '41', kind: 'message.committed', data: { messageId: 'm1' } }
  const connections = []
  const deliveries = []
  let detailRequests = 0
  const client = new AgentWechatClient({
    baseUrl: 'http://upstream.test',
    fetchImpl: async (url) => {
      const { pathname, searchParams } = new URL(url)
      if (pathname === '/v1/accounts') return Response.json([{ id: 'acc' }])
      if (pathname === '/v1/status') return Response.json({ serverId: 'server' })
      if (pathname === '/v1/accounts/acc/events/stream') {
        const after = searchParams.get('after')
        connections.push({ after, persistedCursor: loadCursor(config.cursorPath)?.cursor })
        if (after !== '40' || connections.length > 2) controller.abort()
        return new Response(after === '40' ? `data: ${JSON.stringify(event)}\n\n` : '', {
          headers: { 'Content-Type': 'text/event-stream' },
        })
      }
      if (pathname === '/v1/accounts/acc/messages/m1') {
        detailRequests += 1
        if (detailRequests === 1) return Response.json({ error: { code: 'unavailable' } }, { status: 503 })
        return Response.json(incoming())
      }
      assert.fail(`unexpected upstream request: ${url}`)
    },
  })
  const adapter = new AgentWechatAdapter({
    client,
    directory: fakeDirectory(),
    config,
    logger: silentLogger,
    fetchImpl: async (url, init) => {
      deliveries.push({ url, messageId: JSON.parse(init.body).source_message_id })
      controller.abort()
      return Response.json({ ok: true, accepted: true }, { status: 202 })
    },
  })

  await adapter.run(controller.signal)

  assert.deepEqual(connections, [
    { after: '40', persistedCursor: '40' },
    { after: '40', persistedCursor: '40' },
  ])
  assert.equal(detailRequests, 2)
  assert.deepEqual(deliveries, [{ url: config.agentEndpoint, messageId: 'm1' }])
  assert.equal(loadCursor(config.cursorPath)?.cursor, '41')
})

test('游标过期补读遇到消息详情失败时保留进度，下一次继续补读', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const pages = []
  const deliveries = []
  let detailRequests = 0
  const { adapter, config } = makeAdapter({
    dir,
    clientOverrides: {
      listEvents: async ({ after }) => {
        pages.push(after)
        if (after !== '40') return { items: [], nextCursor: after }
        return {
          items: [{ cursor: '41', kind: 'message.committed', data: { messageId: 'm1' } }],
          nextCursor: '41',
        }
      },
      getMessage: async () => {
        detailRequests += 1
        return detailRequests === 1 ? null : incoming()
      },
    },
    fetchImpl: async (_url, init) => {
      deliveries.push(JSON.parse(init.body).source_message_id)
      return Response.json({ ok: true, accepted: true }, { status: 202 })
    },
  })
  saveCursor(config.cursorPath, { cursor: '40', accountId: 'acc' })
  await adapter.bootstrapCursor()

  await adapter.recoverExpiredCursor()
  assert.equal(adapter.cursor, '40')
  assert.equal(loadCursor(config.cursorPath)?.cursor, '40')
  assert.deepEqual(deliveries, [])

  await adapter.recoverExpiredCursor()
  assert.deepEqual(pages, ['40', '40'])
  assert.deepEqual(deliveries, ['m1'])
  assert.equal(loadCursor(config.cursorPath)?.cursor, '41')
})

/** 模拟守护进程拒绝保留窗口之前的游标的分页语义。 */
function retainedEventsPage(events, after, limit) {
  if (BigInt(after) < BigInt(events[0].cursor) - 1n) {
    throw Object.assign(new Error('event cursor expired'), { code: 'event_cursor_expired' })
  }
  const items = events.filter(event => BigInt(event.cursor) > BigInt(after)).slice(0, limit)
  return { items, nextCursor: items.at(-1)?.cursor || after }
}

test('确认过期后从精确保留边界补读多页，详情失败时留下最后确认游标', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const first = 9_007_199_254_740_993n
  const events = Array.from({ length: 201 }, (_unused, index) => ({
    cursor: String(first + BigInt(index)),
    kind: index === 0 || index === 200 ? 'message.committed' : 'account.updated',
    data: { messageId: index === 0 ? 'm1' : 'm2' },
  }))
  const posted = []
  const warnings = []
  const { adapter, config } = makeAdapter({
    dir,
    deferAutomaticReplies: true,
    clientOverrides: {
      listEvents: async ({ after, limit }) => retainedEventsPage(events, after, limit),
      getMessage: async id => {
        if (id === 'm1') assert.equal(loadCursor(config.cursorPath)?.cursor, String(first - 1n))
        if (id === 'm2' && posted.length === 1) return null
        return incoming(id)
      },
    },
    fetchImpl: async (_url, init) => {
      posted.push(JSON.parse(init.body).source_message_id)
      return Response.json({ ok: true, accepted: true }, { status: 202 })
    },
  })
  adapter.logger = { ...silentLogger, warn: warning => warnings.push(warning) }
  saveCursor(config.cursorPath, { cursor: String(first - 20n), accountId: 'acc' })
  await adapter.bootstrapCursor()
  await adapter.recoverExpiredCursor()
  assert.deepEqual(posted, ['m1'])
  assert.equal(loadCursor(config.cursorPath)?.cursor, String(first + 199n))
  assert.match(warnings[0], new RegExp(`\\b${first - 19n}\\.\\.${first - 1n}\\b`))
  assert.match(warnings[1], /details unavailable/)

  adapter.client.getMessage = async id => incoming(id)
  await adapter.recoverExpiredCursor()
  assert.deepEqual(posted, ['m1', 'm2'])
  assert.equal(loadCursor(config.cursorPath)?.cursor, String(first + 200n))
})

test('查找保留边界时遇到临时 HTTP 错误，已保存游标保持原值', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const { adapter, config } = makeAdapter({
    dir,
    deferAutomaticReplies: true,
    clientOverrides: {
      listEvents: async ({ after }) => {
        if (after === '40') throw Object.assign(new Error('event cursor expired'), { code: 'event_cursor_expired' })
        throw Object.assign(new Error('temporarily unavailable'), { status: 503 })
      },
    },
  })
  saveCursor(config.cursorPath, { cursor: '40', accountId: 'acc' })
  await adapter.bootstrapCursor()
  await adapter.recoverExpiredCursor()
  assert.equal(adapter.cursor, '40')
  assert.equal(loadCursor(config.cursorPath)?.cursor, '40')
})

test('首次启动禁用历史重放时能在游标零过期后定位事件日志末尾', async (t) => {
  const dir = tempDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const { adapter, config } = makeAdapter({
    dir,
    clientOverrides: {
      listEvents: async ({ after, limit }) => retainedEventsPage([{ cursor: '500' }, { cursor: '501' }], after, limit),
    },
  })
  assert.equal(await adapter.bootstrapCursor(), '501')
  assert.equal(loadCursor(config.cursorPath)?.cursor, '501')
})

test('初次 SSE 请求的 HTTP 错误保留原生事件游标过期代码', async () => {
  const client = new AgentWechatClient({
    baseUrl: 'http://upstream.test',
    accountId: 'acc',
    fetchImpl: async () => Response.json({ error: { code: 'event_cursor_expired' } }, { status: 410 }),
  })
  await assert.rejects(client.streamEvents({ after: '40' }), {
    status: 410,
    code: 'event_cursor_expired',
  })
})

test('同步投递 Runtime 4xx 时把载荷落重放队列', async () => {
  const dir = tempDir()
  const { adapter, config } = makeAdapter({
    dir,
    messages: new Map([['m1', incoming()]]),
    switches: { sendChannelEnabled: true },
    fetchImpl: async () => new Response(JSON.stringify({ error: 'bad' }), { status: 400 }),
  })
  await adapter.handleEvent({ kind: 'message.committed', data: { messageId: 'm1' } })
  const files = fs.readdirSync(config.spoolDir)
  assert.equal(files.length, 1)
  const record = JSON.parse(fs.readFileSync(path.join(config.spoolDir, files[0]), 'utf8'))
  assert.equal(record.payload.source_message_id, 'm1')
})

test('dry-run 不投递也不外发', async () => {
  let called = 0
  const { adapter } = makeAdapter({
    dir: tempDir(),
    messages: new Map([['m1', incoming()]]),
    configOverrides: { dryRun: true },
    fetchImpl: async () => {
      called += 1
      return okFetch()
    },
  })
  await adapter.handleEvent({ kind: 'message.committed', data: { messageId: 'm1' } })
  assert.equal(called, 0)
})

test('自消息和系统消息默认不投递', async () => {
  let called = 0
  const messages = new Map([
    ['m1', incoming('m1', { direction: 'outgoing' })],
    ['m2', incoming('m2', { type: 10000 })],
  ])
  const { adapter } = makeAdapter({
    dir: tempDir(),
    messages,
    fetchImpl: async () => {
      called += 1
      return okFetch()
    },
  })
  await adapter.handleEvent({ kind: 'message.committed', data: { messageId: 'm1' } })
  await adapter.handleEvent({ kind: 'message.committed', data: { messageId: 'm2' } })
  assert.equal(called, 0)
})

test('accepted 才写投递账本', async () => {
  const dir = tempDir()
  const sent = []
  const { adapter, config } = makeAdapter({
    dir,
    messages: new Map([['m1', incoming()]]),
    switches: { sendChannelEnabled: true, autoSendReply: true },
    clientOverrides: {
      sendText: async (payload) => {
        sent.push(payload)
        return { id: 'send-1' }
      },
    },
    fetchImpl: okFetch,
  })
  await adapter.handleEvent({ kind: 'message.committed', data: { messageId: 'm1' } })
  assert.equal(sent.length, 1)
  assert.equal(sent[0].idempotencyKey, 'mahjong-reply:m1')
  const ledger = fs.readFileSync(config.deliveryLedgerPath, 'utf8').trim().split('\n')
  assert.equal(ledger.length, 1)
  assert.equal(JSON.parse(ledger[0]).source_message_id, 'm1')
})

test('needs_review 不写账本且沿用同一幂等键', async () => {
  const dir = tempDir()
  const keys = []
  const { adapter, config } = makeAdapter({
    dir,
    messages: new Map([['m1', incoming()]]),
    switches: { sendChannelEnabled: true, autoSendReply: true },
    clientOverrides: {
      sendText: async (payload) => {
        keys.push(payload.idempotencyKey)
        return { id: 'send-1' }
      },
      waitForSend: async () => ({ id: 'send-1', state: 'needs_review', errorCode: 'unknown' }),
    },
    fetchImpl: okFetch,
  })
  await adapter.handleEvent({ kind: 'message.committed', data: { messageId: 'm1' } })
  assert.equal(fs.existsSync(config.deliveryLedgerPath), false)
  assert.equal(adapter.delivered.has('m1'), false)
  // 即使事件重放，幂等键也保持不变，由上游阻止重复发送。
  await adapter.handleEvent({ kind: 'message.committed', data: { messageId: 'm1' } })
  assert.deepEqual(keys, ['mahjong-reply:m1', 'mahjong-reply:m1'])
})

test('外发开关关闭时不发送', async () => {
  let sent = 0
  const { adapter } = makeAdapter({
    dir: tempDir(),
    messages: new Map([['m1', incoming()]]),
    switches: { sendChannelEnabled: true, autoSendReply: false },
    clientOverrides: {
      sendText: async () => {
        sent += 1
        return { id: 'send-1' }
      },
    },
    fetchImpl: okFetch,
  })
  await adapter.handleEvent({ kind: 'message.committed', data: { messageId: 'm1' } })
  assert.equal(sent, 0)
})

test('多账号未显式配置时拒绝启动', async () => {
  const { adapter } = makeAdapter({
    dir: tempDir(),
    messages: new Map(),
    fetchImpl: okFetch,
    clientOverrides: { listAccounts: async () => [{ id: 'a' }, { id: 'b' }] },
  })
  await assert.rejects(() => adapter.resolveAccount(), /multiple agent-wechat accounts/)
})

test('多账号显式选择时同步真实账号 id 给上游客户端', async () => {
  const { adapter, client } = makeAdapter({
    dir: tempDir(),
    messages: new Map(),
    fetchImpl: okFetch,
    configOverrides: { accountId: 'mahjong-work' },
    clientOverrides: {
      listAccounts: async () => [
        { id: 'acc-a', alias: 'other' },
        { id: 'acc-b', alias: 'mahjong-work', ownUsername: 'work' },
      ],
    },
  })
  const account = await adapter.resolveAccount()
  assert.equal(account.id, 'acc-b')
  assert.equal(adapter.account.id, 'acc-b')
  assert.equal(client.accountId, 'acc-b')
  assert.equal(adapter.selfUsername, 'work')
})

test('定向群回复核对策略、账号和原消息，并保留引用的完整服务器 ID', async () => {
  const dir = tempDir()
  const groupReplyPolicyPath = path.join(dir, 'group-replies.json')
  const roomId = '123@chatroom'
  const sent = []
  const { adapter } = makeAdapter({
    dir,
    messages: new Map(),
    configOverrides: { outboundAllowlistPath: 'configured.json', groupReplyPolicyPath },
    switches: { sendChannelEnabled: true, autoSendReply: true },
    clientOverrides: {
      sendText: async (request) => {
        sent.push(request)
        return { id: 'send-1' }
      },
    },
  })
  const payload = await adapter.buildPayload(incoming('m1', {
    createdAt: '2026-01-02T12:00:00Z',
    serverId: '9876543210123456789',
  }))
  const approval = { room_id: roomId, source_message_id: 'm1', text: '你大概几点方便？' }
  const result = (reply, channel = 'agent-wechat') => ({
    channel,
    route_result: { scoped_group_reply: reply, agent_result: { final_reply: '通用话术' } },
  })
  const policy = { account_id: 'acc', room_ids: [roomId], enabled_at: '2026-01-02T11:00:00Z' }
  fs.writeFileSync(groupReplyPolicyPath, JSON.stringify(policy))
  await adapter.maybeSendReply(payload, result(null))
  await adapter.maybeSendReply(payload, result({ ...approval, source_message_id: 'other' }))
  await adapter.maybeSendReply(payload, result({ ...approval, room_id: '321@chatroom' }))
  await adapter.maybeSendReply(payload, result(approval, 'wechaty'))
  await adapter.maybeSendReply({ ...payload, self_message: true }, result(approval))
  await adapter.maybeSendReply({ ...payload, payload: { ...payload.payload, account_id: 'other' } }, result(approval))
  fs.writeFileSync(groupReplyPolicyPath, JSON.stringify({ ...policy, room_ids: [] }))
  await adapter.maybeSendReply(payload, result(approval))
  fs.writeFileSync(groupReplyPolicyPath, JSON.stringify({ ...policy, enabled_at: '2026-01-03T00:00:00Z' }))
  await adapter.maybeSendReply(payload, result(approval))
  fs.writeFileSync(groupReplyPolicyPath, JSON.stringify(policy))
  adapter.switches.autoSendReply = false
  await adapter.maybeSendReply(payload, result(approval))
  assert.deepEqual(sent, [])

  adapter.switches.autoSendReply = true
  await adapter.maybeSendReply(payload, result(approval))
  await adapter.maybeSendReply(payload, result(approval))
  assert.deepEqual(sent, [{
    threadId: roomId,
    text: approval.text,
    quote: { messageId: 'm1', serverId: '9876543210123456789' },
    idempotencyKey: 'mahjong-reply:m1',
  }])
})

test('只入站拒绝没有持久接收语义的 HTTP 200', async () => {
  await assert.rejects(
    postAgentRaw('http://runtime.test/raw', '', {}, {
      fetchImpl: okFetch, attempts: 1, logger: silentLogger, respondAsync: true,
    }),
    /did not confirm durable ingress/,
  )
})
