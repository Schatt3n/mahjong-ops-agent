import test from 'node:test'
import assert from 'node:assert/strict'
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
    adapter: { account: { id: 'acc' }, selfUsername: 'me', cursor: '42' },
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
