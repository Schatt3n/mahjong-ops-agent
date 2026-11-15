import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { AgentWechatAdapter } from '../src/adapter.mjs'

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
function makeAdapter({ dir, messages, fetchImpl, clientOverrides = {}, configOverrides = {}, switches = {} }) {
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
      return okFetch()
    },
  })
  await adapter.handleEvent({ kind: 'message.committed', data: { messageId: 'm1' } })
  assert.equal(calls.length, 1)
  assert.equal(calls[0].body.conversation_id, 'wechaty:room:123@chatroom')
  assert.equal(calls[0].body.channel, 'agent-wechat')
})

test('Runtime 4xx 时把载荷落重放队列', async () => {
  const dir = tempDir()
  const { adapter, config } = makeAdapter({
    dir,
    messages: new Map([['m1', incoming()]]),
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
