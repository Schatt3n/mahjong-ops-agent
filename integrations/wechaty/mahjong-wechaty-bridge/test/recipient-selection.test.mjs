import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { Readable } from 'node:stream'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { runInNewContext } from 'node:vm'

const bridgeUrl = new URL('../src/bridge.mjs', import.meta.url)
const controlSource = readFileSync(bridgeUrl, 'utf8')
  .split("\nbot.on('scan',")[0]
  .replace(/^import .*\n/gm, '')
  .replace('import.meta.url', JSON.stringify(bridgeUrl.href))

/** 用内存通道运行同一 HTTP 路由和账本逻辑，避免启动实际服务或发送微信消息。 */
function bridgeFixture({ ledger = new Map(), waitToSend = Promise.resolve(), sendingEnabled = true } = {}) {
  let handler
  const calls = []
  const contacts = new Set()
  const rooms = new Set()
  const makeTarget = (kind, ids, id) => {
    if (!ids.has(id)) throw new Error('missing fixture ID')
    return {
      id,
      ready: async () => {},
      topic: async () => '群聊',
      say: async (text) => {
        calls.push({ kind, id, text })
        await waitToSend
        return { id: `message-${calls.length}` }
      },
    }
  }
  const bot = {
    Contact: {
      load: (id) => makeTarget('direct', contacts, id),
      find: () => assert.fail('selected contact must not fall back to Contact.find'),
      findAll: () => assert.fail('selected contact must not fall back to Contact.findAll'),
    },
    Room: {
      load: (id) => makeTarget('group', rooms, id),
      find: () => assert.fail('selected group must not fall back to Room.find'),
      findAll: () => assert.fail('selected group must not fall back to Room.findAll'),
    },
  }
  const fixture = runInNewContext(`${controlSource}
startOutboundServer()
globalThis.fixture = { rememberContact, rememberRoom }
fixture
`, {
    URL,
    Buffer,
    path,
    fileURLToPath,
    createHash,
    process: { env: {
      MAHJONG_WECHATY_LOCAL_DATA_DIR: '/synthetic-wechaty-data',
      MAHJONG_WECHATY_SEND_ENABLED: sendingEnabled ? '1' : '0',
    } },
    console: { log() {} },
    fs: {
      mkdirSync() {},
      readFileSync(file) {
        if (!ledger.has(file)) throw new Error('missing fixture file')
        return ledger.get(file)
      },
      appendFileSync(file, text) { ledger.set(file, (ledger.get(file) || '') + text) },
    },
    http: { createServer(listener) { handler = listener; return { listen() {} } } },
    WechatyBuilder: { build: () => bot },
  })
  return {
    calls,
    addContact(id, name, alias = '', weixin = '') {
      contacts.add(id)
      fixture.rememberContact({ id, name, alias, payload: { weixin, private_note: 'not for search' } })
    },
    addRoom(id, topic) {
      rooms.add(id)
      fixture.rememberRoom({ id, topic })
    },
    async request(method, url, payload = null) {
      const request = Object.assign(Readable.from(payload ? [Buffer.from(JSON.stringify(payload))] : []), { method, url })
      let status
      let body
      await handler(request, {
        writeHead(code) { status = code },
        end(text) { body = JSON.parse(text) },
      })
      return { status, body }
    },
  }
}

test('模糊检索按精确、前缀、包含排序，且只暴露候选字段', async () => {
  const fixture = bridgeFixture({ sendingEnabled: false })
  fixture.addContact('wxid_exact', 'LI')
  fixture.addContact('wxid_prefix', '旧昵称', 'Li 哥')
  fixture.addContact('wxid_substring', '第三位', '', 'x_li_x')
  fixture.addRoom('123@chatroom', '老友 Li 群')
  for (let n = 0; n < 12; n += 1) fixture.addContact(`wxid_other_${n}`, `其它 Li 好友${n}`)
  const result = await fixture.request('GET', '/resolve?query=%20li%20')
  assert.equal(result.status, 200)
  assert.equal(result.body.candidates.length, 10)
  assert.deepEqual(result.body.candidates.slice(0, 2).map(item => item.id), ['wxid_exact', 'wxid_prefix'])
  assert.deepEqual(result.body.candidates[1], {
    id: 'wxid_prefix', kind: 'direct', displayName: 'Li 哥', remark: 'Li 哥', nickname: '旧昵称', alias: null,
  })
  assert.deepEqual((await fixture.request('GET', '/resolve?query=x_li_')).body.candidates.map(item => item.id),
    ['wxid_substring'])
  assert.deepEqual((await fixture.request('GET', '/resolve?query=123%40chatroom')).body.candidates.map(item => item.kind),
    ['group'])
  assert.deepEqual((await fixture.request('GET', '/resolve?query=%20')).body.candidates, [])
  assert.deepEqual(fixture.calls, [])
})

test('精确发送拒绝把过期 ID 当备注使用，已选群按群 ID 发送', async () => {
  const fixture = bridgeFixture()
  fixture.addContact('wxid_person', '昵称', 'wxid_previous')
  fixture.addRoom('123@chatroom', '常用群')
  const payload = { text: '周末好', exact_target: true, source_trace_id: 'console-request-1' }
  const stale = await fixture.request('POST', '/send', { ...payload, to: 'wxid_previous', target_type: 'direct' })
  assert.equal(stale.body.ok, false)
  assert.match(stale.body.error, /selected recipient is no longer available/)
  assert.deepEqual(fixture.calls, [])

  const group = await fixture.request('POST', '/send', { ...payload, to: '123@chatroom', target_type: 'group' })
  assert.equal(group.body.ok, true)
  assert.equal(group.body.room_id, '123@chatroom')
  assert.deepEqual(fixture.calls, [{ kind: 'group', id: '123@chatroom', text: '周末好' }])
})

test('并发和重启后重试保留同一收件人与消息 ID，且不重复外发', async () => {
  const ledger = new Map()
  let releaseSend
  const waitToSend = new Promise(resolve => { releaseSend = resolve })
  const fixture = bridgeFixture({ ledger, waitToSend })
  fixture.addContact('wxid_person', '昵称', '备注')
  const payload = {
    to: 'wxid_person', target_type: 'direct', exact_target: true, text: '周末好', source_trace_id: 'console-request-2',
  }
  const pending = [fixture.request('POST', '/send', payload), fixture.request('POST', '/send', payload)]
  await new Promise(setImmediate)
  releaseSend()
  const [first, second] = await Promise.all(pending)
  assert.deepEqual(fixture.calls, [{ kind: 'direct', id: 'wxid_person', text: '周末好' }])
  assert.equal(first.body.contact_id, 'wxid_person')
  assert.equal(first.body.message_id, 'message-1')
  assert.equal(second.body.deduplicated, true)
  assert.equal(second.body.message_id, first.body.message_id)

  const restarted = bridgeFixture({ ledger })
  const retry = await restarted.request('POST', '/send', payload)
  assert.equal(retry.body.deduplicated, true)
  assert.equal(retry.body.contact_id, 'wxid_person')
  assert.equal(retry.body.message_id, first.body.message_id)
  assert.deepEqual(restarted.calls, [])

  const paused = bridgeFixture({ ledger, sendingEnabled: false })
  const blocked = await paused.request('POST', '/send', payload)
  assert.equal(blocked.body.ok, false)
  assert.match(blocked.body.error, /paused/)
  assert.deepEqual(paused.calls, [])
})
