import assert from 'node:assert/strict'
import { createHash, timingSafeEqual } from 'node:crypto'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { Readable } from 'node:stream'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { runInNewContext } from 'node:vm'
import { listenerStatus, sourceBuild } from '../src/bridge-operations.mjs'

const bridgeUrl = new URL('../src/bridge.mjs', import.meta.url)
const bridgeControlToken = 'fixture-bridge-control'
const controlSource = readFileSync(bridgeUrl, 'utf8')
  .replace(/\r\n/g, '\n')
  .split("\nbot?.on('scan',")[0]
  .replace(/^import .*\n/gm, '')
  .replace('import.meta.url', JSON.stringify(bridgeUrl.href))
  .replace("(await import('wechaty')).WechatyBuilder.build({ name: botName })", 'fixtureBot')

/** 用内存通道运行同一 HTTP 路由和账本逻辑，避免启动实际服务或发送微信消息。 */
function bridgeFixture({ ledger = new Map(), waitToSend = Promise.resolve(), sendingEnabled = true,
  autoSend = false, aliases = '' } = {}) {
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
globalThis.fixture = { rememberContact, rememberRoom, observePrivateIncoming }
fixture
`, {
    URL,
    Buffer,
    path,
    fileURLToPath,
    createHash,
    timingSafeEqual,
    listenerStatus,
    sourceBuild,
    fixtureBot: bot,
    process: { env: {
      MAHJONG_WECHATY_LOCAL_DATA_DIR: '/synthetic-wechaty-data',
      MAHJONG_WECHATY_SEND_ENABLED: sendingEnabled ? '1' : '0',
      MAHJONG_WECHATY_AUTO_SEND_REPLY: autoSend ? '1' : '0',
      MAHJONG_WECHATY_CONTACT_ALIASES: aliases,
      MAHJONG_WECHATY_CONTROL_TOKEN: bridgeControlToken,
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
  })
  return {
    calls,
    privateMessage(id, target, self = false) {
      fixture.observePrivateIncoming({
        id, self: () => self, room: () => null, talker: () => ({ id: target }),
        listener: () => ({ id: target }), text: () => '老板已回复',
      })
    },
    addContact(id, name, alias = '', weixin = '') {
      contacts.add(id)
      fixture.rememberContact({ id, name, alias, payload: { weixin, private_note: 'not for search' } })
    },
    addRoom(id, topic) {
      rooms.add(id)
      fixture.rememberRoom({ id, topic })
    },
    async request(method, url, payload = null) {
      const headers = { 'x-mahjong-bridge-token': bridgeControlToken, 'content-type': 'application/json' }
      const request = Object.assign(Readable.from(payload ? [Buffer.from(JSON.stringify(payload))] : []), {
        method, url, headers,
      })
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

const fullTableNotice = {
  to: 'wxid_guest', target_type: 'contact', exact_target: true, text: '人齐了，七点开。',
  source: 'managed_group_domain', source_trace_id: 'notice-run-1', message_type: 'group_game_full_notice',
  draft_id: 'table-full-draft', source_message_id: 'table-full-draft',
}

test('满桌通知只发精确联系人，并继续服从自动外发开关', async () => {
  const fixture = bridgeFixture({ autoSend: true, aliases: 'wxid_guest=wxid_decoy' })
  fixture.addContact('wxid_decoy', '其他顾客', 'wxid_guest')
  for (const changed of [
    { exact_target: false }, { draft_id: '' }, { source: 'manual_operator' },
    { target_type: 'room' }, { to: '123@chatroom' },
  ]) {
    const rejected = await fixture.request('POST', '/send', { ...fullTableNotice, ...changed })
    assert.equal(rejected.status, 400)
  }
  const stale = await fixture.request('POST', '/send', fullTableNotice)
  assert.equal(stale.body.ok, false)
  assert.match(stale.body.error, /selected recipient is no longer available/)
  assert.deepEqual(fixture.calls, [])

  fixture.addContact('wxid_guest', '原顾客')
  const sent = await fixture.request('POST', '/send', fullTableNotice)
  assert.equal(sent.body.ok, true)
  assert.equal(sent.body.contact_id, 'wxid_guest')
  assert.equal(sent.body.message_id, 'message-1')
  assert.deepEqual(fixture.calls, [{ kind: 'direct', id: 'wxid_guest', text: fullTableNotice.text }])

  const paused = bridgeFixture()
  paused.addContact('wxid_guest', '原顾客')
  const blocked = await paused.request('POST', '/send', fullTableNotice)
  assert.equal(blocked.status, 409)
  assert.match(blocked.body.error, /automatic outbound is paused/)
  assert.deepEqual(paused.calls, [])
})

test('满桌通知跨追踪和重启复用原回执，收件人或正文变更不再次投递', async () => {
  const ledger = new Map()
  let releaseSend
  const waitToSend = new Promise(resolve => { releaseSend = resolve })
  const fixture = bridgeFixture({ autoSend: true, ledger, waitToSend })
  fixture.addContact('wxid_guest', '原顾客')
  const retry = { ...fullTableNotice, source_trace_id: 'notice-run-2' }
  const concurrent = [fixture.request('POST', '/send', fullTableNotice), fixture.request('POST', '/send', retry)]
  await new Promise(setImmediate)
  const changedInFlight = await fixture.request('POST', '/send', { ...retry, text: '人齐了，八点开。' })
  assert.equal(changedInFlight.body.ok, false)
  assert.match(changedInFlight.body.error, /notice recipient or text changed/)
  releaseSend()
  const [first, second] = await Promise.all(concurrent)
  assert.equal(first.body.message_id, 'message-1')
  assert.equal(second.body.deduplicated, true)
  assert.equal(second.body.message_id, first.body.message_id)
  assert.equal(fixture.calls.length, 1)

  const restarted = bridgeFixture({ autoSend: true, ledger })
  restarted.addContact('wxid_other', '其他顾客')
  const delivered = await restarted.request('POST', '/send', retry)
  assert.equal(delivered.body.deduplicated, true)
  assert.equal(delivered.body.contact_id, 'wxid_guest')
  assert.equal(delivered.body.text, fullTableNotice.text)
  assert.equal(delivered.body.message_id, first.body.message_id)
  for (const changed of [{ to: 'wxid_other' }, { text: '人齐了，八点开。' }]) {
    const rejected = await restarted.request('POST', '/send', { ...retry, ...changed })
    assert.equal(rejected.body.ok, false)
    assert.match(rejected.body.error, /notice recipient or text changed/)
  }
  assert.deepEqual(restarted.calls, [])
  assert.ok(![...ledger.values()].join('').includes(fullTableNotice.text))
})

test('晚到征询和回告只发真实选定的收件人，重试保留同一回执', async () => {
  const fixture = bridgeFixture({ autoSend: true })
  fixture.addContact('wxid_guest', '原顾客')
  for (const [message_type, text] of [
    ['late_arrival_question', '五点半那桌有人六点五十五才到，能等吗？'],
    ['late_arrival_update', '五点半那桌说等得到六点五十五，你还来吗？'],
  ]) {
    const payload = { ...fullTableNotice, message_type, text, draft_id: message_type }
    const rejected = await fixture.request('POST', '/send', { ...payload, exact_target: false })
    assert.equal(rejected.status, 400)
    const first = await fixture.request('POST', '/send', payload)
    assert.equal(first.body.contact_id, 'wxid_guest')
    assert.ok(first.body.message_id)
    const retry = await fixture.request('POST', '/send', { ...payload, source_trace_id: 'retry' })
    assert.equal(retry.body.message_id, first.body.message_id)
    assert.equal(retry.body.deduplicated, true)
  }
  assert.equal(fixture.calls.length, 2)
})

const runtimeWait = {
  to: 'wxid_guest', target_type: 'contact', exact_target: true, text: '稍等一下',
  source: 'agent_runtime_wait', source_trace_id: 'run-1:runtime-wait', source_message_id: 'incoming-1',
}
const runtimeFinal = {
  to: 'wxid_guest', target_type: 'contact', exact_target: true, text: '七点的桌还有一位。',
  source: 'recovered_agent_run_reply', source_trace_id: 'run-1', source_message_id: 'recovery-1',
  runtime_wait_origin_source: 'incoming-1',
}

test('在途的等待先结算，补发正常回复，重复等待被抑制', async () => {
  let releaseSend
  const waitToSend = new Promise(resolve => { releaseSend = resolve })
  const fixture = bridgeFixture({ autoSend: true, waitToSend })
  fixture.addContact('wxid_guest', '原顾客')
  fixture.privateMessage('incoming-1', 'wxid_guest')
  const waiting = fixture.request('POST', '/send', runtimeWait)
  await new Promise(setImmediate)
  const final = fixture.request('POST', '/send', runtimeFinal)
  const duplicateFallback = fixture.request('POST', '/send', { ...runtimeFinal, text: runtimeWait.text })
  await new Promise(setImmediate)
  assert.deepEqual(fixture.calls.map(item => item.text), [runtimeWait.text])
  releaseSend()
  assert.equal((await waiting).body.message_id, 'message-1')
  assert.equal((await final).body.ok, true)
  assert.equal((await duplicateFallback).body.suppressed, true)
  assert.deepEqual(fixture.calls.map(item => item.text), [runtimeWait.text, runtimeFinal.text])
})

test('新来话、店家回复、最终回复或自动外发开关阻止旧等待', async () => {
  for (const later of ['inbound', 'owner', 'final', 'paused']) {
    const fixture = bridgeFixture({ autoSend: later !== 'paused' })
    fixture.addContact('wxid_guest', '原顾客')
    fixture.privateMessage('incoming-1', 'wxid_guest')
    if (later === 'inbound') fixture.privateMessage('incoming-2', 'wxid_guest')
    if (later === 'owner') fixture.privateMessage('owner-reply', 'wxid_guest', true)
    if (later === 'final') assert.equal((await fixture.request('POST', '/send', runtimeFinal)).body.ok, true)
    const skipped = await fixture.request('POST', '/send', runtimeWait)
    if (later === 'paused') assert.equal(skipped.body.ok, false)
    else assert.equal(skipped.body.suppressed, true)
    assert.deepEqual(fixture.calls.map(item => item.text), later === 'final' ? [runtimeFinal.text] : [])
  }
})
