import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { AgentWechatClient } from '../src/client.mjs'
import { ControlSwitches, startControlServer } from '../src/control.mjs'

/** 构造控制面测试配置，端口 0 由系统分配。 */
function makeConfig(overrides = {}) {
  return {
    upstreamBaseUrl: 'http://127.0.0.1:6174',
    controlHost: '127.0.0.1',
    controlPort: 0,
    controlToken: '',
    requestBodyLimitBytes: 64 * 1024,
    sendWaitTimeoutMs: 1_000,
    dryRun: false,
    ...overrides,
  }
}

/** 构造包含重名联系人和群的最小目录替身。 */
function makeDirectory(overrides = {}) {
  return {
    contacts: [
      { id: 'wxid_wang', username: 'wxid_wang', remark: '王老板', nickname: '老王', displayName: '王老板' },
      { id: 'wxid_li', username: 'wxid_li', remark: '李姐', nickname: '李', displayName: '李姐' },
    ],
    groups: [{ id: '100@chatroom', roomId: '100@chatroom', displayName: '周末麻将群' }],
    refresh: async () => {},
    resolve: async (query, { kind } = {}) => {
      const exact = [
        { id: 'wxid_li', threadId: 'wxid_li', kind: 'direct', displayName: '李姐', remark: '李姐' },
      ].filter((item) => item.displayName === query && (!kind || item.kind === kind))
      return {
        query,
        kind: kind || 'any',
        candidates: exact,
        exactCount: exact.length,
        ambiguous: exact.length !== 1,
        recommendedId: exact.length === 1 ? exact[0].id : null,
      }
    },
    ...overrides,
  }
}

/** 启动临时控制面并确保测试结束后关闭。 */
async function withServer(options, fn) {
  const dir = makeDirectory(options.directoryOverrides)
  const client = {
    accountId: 'acc',
    getSendByKey: async () => {
      throw Object.assign(new Error('not found'), { status: 404 })
    },
    sendText: async () => ({ id: 'send-1' }),
    waitForSend: async () => ({ id: 'send-1', state: 'accepted', localMessageId: 'wx-out-1' }),
    ...options.clientOverrides,
  }
  const switches = new ControlSwitches({
    sendChannelEnabled: options.sendChannelEnabled ?? true,
    autoSendReply: options.autoSendReply ?? true,
  })
  const config = makeConfig(options.configOverrides)
  const logger = { info() {}, warn() {}, error() {} }
  const control = await startControlServer({
    config,
    client,
    directory: dir,
    switches,
    adapter: { account: { id: 'acc' }, selfUsername: 'me', cursor: '42', ...options.adapterOverrides },
    logger,
  })
  const port = control.server.address().port
  const call = async (pathname, init = {}) => {
    const response = await fetch(`http://127.0.0.1:${port}${pathname}`, init)
    return { status: response.status, body: await response.json() }
  }
  try {
    await fn({ call, switches, client, config })
  } finally {
    await control.close()
  }
}

test('health 无需 token，且不泄漏上游 token', async () => {
  await withServer({ configOverrides: { controlToken: 'secret' } }, async ({ call }) => {
    const { status, body } = await call('/health')
    assert.equal(status, 200)
    assert.equal(body.ok, true)
    assert.equal(body.adapter, 'agent-wechat')
    assert.equal(body.send_channel_enabled, true)
    assert.equal(body.cursor, '42')
    assert.equal(body.upstream_token, undefined)
  })
})

test('配置控制 token 后其余路由必须鉴权', async () => {
  await withServer({ configOverrides: { controlToken: 'secret' } }, async ({ call }) => {
    const denied = await call('/contacts')
    assert.equal(denied.status, 401)
    const allowed = await call('/contacts', { headers: { 'X-Mahjong-Agent-Token': 'secret' } })
    assert.equal(allowed.status, 200)
    assert.equal(allowed.body.contacts.length, 2)
  })
})

test('后台自动回复通过鉴权后复用外发检查并返回发送终态', async () => {
  const calls = []
  const payload = {
    channel: 'agent-wechat',
    self_message: false,
    source_message_id: 'inbound-1',
    message_id: 'inbound-1',
    conversation_id: 'wechaty:contact:wxid_li',
    payload: { id: 'inbound-1', account_id: 'acc', direction: 'incoming', threadId: 'wxid_li' },
  }
  const result = {
    ok: true, trace_id: 'trace-1', channel: 'agent-wechat',
    route_result: { agent_result: { final_reply: '收到' } },
  }
  await withServer({
    configOverrides: { agentToken: 'runtime-secret' },
    adapterOverrides: {
      maybeSendReply: async (...args) => {
        calls.push(args)
        return calls.length === 1 ? { id: 'send-1', state: 'accepted' } : null
      },
    },
  }, async ({ call }) => {
    const deliver = (body, token) => call('/deliver-ingress-reply', {
      method: 'POST',
      headers: token ? { 'X-Mahjong-Agent-Token': token } : {},
      body: JSON.stringify(body),
    })
    const missingToken = await deliver({ payload, result })
    assert.equal(missingToken.status, 401)
    assert.equal(calls.length, 0)
    const token = 'runtime-secret'
    for (const nested of [
      { ...payload.payload, account_id: 'wrong-account' },
      { ...payload.payload, id: 'wrong-source' },
      { ...payload.payload, direction: 'outgoing' },
    ]) {
      const rejected = await deliver({ payload: { ...payload, payload: nested }, result }, token)
      assert.equal(rejected.status, 400)
      assert.equal(rejected.body.ok, false)
    }
    assert.equal(calls.length, 0)
    assert.deepEqual(await deliver({ payload, result }, token), {
      status: 200, body: { ok: true, delivery_state: 'accepted' },
    })
    assert.deepEqual(calls, [[payload, result]])
    assert.deepEqual(await deliver({ payload, result }, token), {
      status: 200, body: { ok: true, delivery_state: 'withheld' },
    })
  })
})

test('发送总闸关闭时拒绝发送', async () => {
  await withServer({ sendChannelEnabled: false }, async ({ call }) => {
    const { status, body } = await call('/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ to: '李姐', target_type: 'contact', text: '你好' }),
    })
    assert.equal(status, 409)
    assert.equal(body.ok, false)
    assert.match(body.error, /send channel is disabled/)
  })
})

test('延迟回复遇到老板已回时返回静默终态，且不轮询发送任务', async () => {
  const source = { cursor: '100', account_id: 'acc', thread_id: 'wxid_li' }
  await withServer({ clientOverrides: {
    sendText: async (attempt) => {
      assert.deepEqual(attempt.replySource, source)
      throw Object.assign(new Error('owner replied'), { code: 'owner_already_replied' })
    },
    waitForSend: async () => assert.fail('没有创建外发任务'),
  } }, async ({ call }) => {
    const { status, body } = await call('/send', {
      method: 'POST',
      body: JSON.stringify({
        to: '李姐', text: '测试用例', source: 'delayed_input_batch_reply', native_reply_source: source,
      }),
    })
    assert.equal(status, 200)
    assert.deepEqual(body, { ok: true, suppressed: true, reason: 'owner_already_replied' })
  })
})

test('定向名单允许手动发送，未授权、已撤销及账号不符均不请求上游', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'wechat-outbound-'))
  const path = join(dir, 'policy.json')
  const one = { thread_id: 'wxid_li', label: '李姐', aliases: ['老李'] }
  const group = { thread_id: '100@chatroom', label: '周末麻将群' }
  const upstream = []
  const sdk = new AgentWechatClient({ baseUrl: 'http://stub', accountId: 'acc', outboundAllowlistPath: path,
    fetchImpl: async (_url, init) => {
      upstream.push(JSON.parse(init.body).threadId)
      return new Response(JSON.stringify({ id: 'send-1' }))
    },
  })
  const policy = (account, recipients) => writeFile(path, JSON.stringify({ account_id: account, recipients }))
  try {
    await policy('acc', [one, group])
    await withServer({ autoSendReply: false, configOverrides: { outboundAllowlistPath: path }, clientOverrides: {
      outboundRecipients: sdk.outboundRecipients.bind(sdk), sendText: sdk.sendText.bind(sdk),
    } }, async ({ call }) => {
      const send = (to, targetType, source) => call('/send', { method: 'POST', body: JSON.stringify({
        to, target_type: targetType, text: '测试用例', source,
      }) })
      assert.equal((await send('老李', 'contact')).status, 200)
      assert.equal((await send('周末麻将群', 'room')).status, 200)
      assert.equal((await send('周末麻将群', 'contact')).status, 403)
      assert.equal((await send('wxid_wang', 'contact')).status, 403)
      assert.equal((await send('101@chatroom', 'room')).status, 403)
      assert.equal((await send('老李', 'contact', 'delayed_input_batch_reply')).status, 409)
      await policy('acc', [group])
      await assert.rejects(sdk.sendText({ threadId: one.thread_id, text: '测试用例', idempotencyKey: 'revoked' }),
        { code: 'outbound_recipient_denied' })
      await policy('other-account', [one, group])
      assert.equal((await send('老李', 'contact')).status, 403)
      await rm(path)
      assert.equal((await send('周末麻将群', 'room')).status, 403)
      assert.deepEqual(upstream, ['wxid_li', '100@chatroom'])
    })
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('无法唯一解析收件人时返回候选而不发', async () => {
  let sent = 0
  await withServer(
    {
      clientOverrides: {
        sendText: async () => {
          sent += 1
          return { id: 'send-1' }
        },
      },
    },
    async ({ call }) => {
      const { status, body } = await call('/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ to: '王老板', target_type: 'contact', text: '你好' }),
      })
      assert.equal(status, 409)
      assert.equal(body.ok, false)
      assert.match(body.error, /ambiguous/)
      assert.equal(sent, 0)
    },
  )
})

test('按唯一完全匹配发送并返回平台消息 id', async () => {
  const calls = []
  await withServer(
    {
      clientOverrides: {
        sendText: async (payload) => {
          calls.push(payload)
          return { id: 'send-1' }
        },
      },
    },
    async ({ call }) => {
      const { status, body } = await call('/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          to: '李姐',
          target_type: 'contact',
          text: '晚上来吗',
          source_trace_id: 'trace-9',
        }),
      })
      assert.equal(status, 200)
      assert.equal(body.ok, true)
      assert.equal(body.message_id, 'wx-out-1')
      assert.equal(body.state, 'accepted')
      assert.equal(calls.length, 1)
      assert.equal(calls[0].threadId, 'wxid_li')
      assert.equal(calls[0].idempotencyKey, 'mahjong-send:trace-9:wxid_li')
    },
  )
})

test('同一幂等键已 accepted 时直接去重', async () => {
  let sent = 0
  await withServer(
    {
      clientOverrides: {
        getSendByKey: async () => ({ id: 'send-old', state: 'accepted', localMessageId: 'wx-old' }),
        sendText: async () => {
          sent += 1
          return { id: 'send-new' }
        },
      },
    },
    async ({ call }) => {
      const { status, body } = await call('/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          to: '李姐',
          target_type: 'contact',
          text: '晚上来吗',
          source_trace_id: 'trace-9',
        }),
      })
      assert.equal(status, 200)
      assert.equal(body.ok, true)
      assert.equal(body.deduplicated, true)
      assert.equal(body.message_id, 'wx-old')
      assert.equal(sent, 0)
    },
  )
})

test('needs_review 返回 ok=false 且提示人工核对', async () => {
  await withServer(
    {
      clientOverrides: {
        waitForSend: async () => ({ id: 'send-1', state: 'needs_review' }),
      },
    },
    async ({ call }) => {
      const { status, body } = await call('/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ to: '李姐', target_type: 'contact', text: '晚上来吗' }),
      })
      assert.equal(status, 200)
      assert.equal(body.ok, false)
      assert.equal(body.state, 'needs_review')
      assert.match(body.error, /do NOT retry automatically/)
    },
  )
})

test('dry-run 下开关不能开启', async () => {
  await withServer(
    {
      sendChannelEnabled: false,
      configOverrides: { dryRun: true },
    },
    async ({ call }) => {
      const { status, body } = await call('/send-channel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: true }),
      })
      assert.equal(status, 409)
      assert.equal(body.ok, false)
      assert.match(body.error, /dry_run/)
    },
  )
})
