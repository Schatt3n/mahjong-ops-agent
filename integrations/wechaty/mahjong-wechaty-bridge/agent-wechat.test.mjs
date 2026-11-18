import assert from 'node:assert/strict'
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { setTimeout as delay } from 'node:timers/promises'
import { createAgentWechatSetup, createAgentWechatTransport } from './src/agent-wechat.mjs'
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { fileURLToPath } from 'node:url'

test('配置按账号核对并保存，切换时从当前事件位置接收', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mahjong-setup-'))
  const file = path.join(dir, 'setup.json')
  const token = path.join(dir, 'token')
  fs.writeFileSync(token, 'fixture-key')
  const seen = []
  const requests = []
  const server = http.createServer(async (request, response) => {
    const route = new URL(request.url, 'http://127.0.0.1')
    requests.push(request.url)
    const [, , , account, source, messageId] = route.pathname.split('/')
    const group = { id: 'room@chatroom', displayName: '样例群' }
    const peer = { id: 'same-id', displayName: '样例联系人' }
    let payload
    if (route.pathname === '/v1/status') payload = { serverId: 'fixture' }
    else if (route.pathname === '/v1/accounts') payload = { items: [{ id: 'old' }, { id: 'new', connection: 'online', authenticated: true }] }
    else if (!source) payload = { id: account, connection: account === 'new' ? 'online' : 'disconnected', authenticated: account === 'new' }
    else if (source === 'threads') payload = { items: [{ ...peer, kind: 'direct' }], asOfEventCursor: '10' }
    else if (source === 'contacts') payload = messageId ? peer : {
      items: route.searchParams.has('cursor') ? [peer, { id: 'unidentified', displayName: '\x7f\u200b' }]
        : [{ id: 'system', classification: { category: 'system' } }],
      nextCursor: route.searchParams.has('cursor') ? null : 'another-page',
    }
    else if (source === 'groups') payload = { items: [group] }
    else if (source === 'events') payload = { items: account === 'new' && route.searchParams.get('after') === '10' ? [
      { cursor: '11', kind: 'message.committed', data: { threadId: peer.id, messageId: 'fresh' } },
    ] : [] }
    else if (source === 'messages' && messageId === 'fresh') payload = {
      id: messageId, accountId: account, threadId: peer.id, senderId: peer.id, type: 1, direction: 'incoming', body: '新消息',
    }
    else if (source === 'sends') {
      let raw = ''
      for await (const chunk of request) raw += chunk
      payload = { accountId: account, destination: JSON.parse(raw).threadId, state: 'accepted', serverId: 'fixture-reply' }
    }
    response.writeHead(payload ? 200 : 404, { 'Content-Type': 'application/json' }).end(JSON.stringify(payload || {}))
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  t.after(() => { server.close(); fs.rmSync(dir, { recursive: true, force: true }) })
  const env = { MAHJONG_AGENT_WECHAT_URL: `http://127.0.0.1:${server.address().port}`,
    MAHJONG_AGENT_WECHAT_TOKEN_PATH: token, MAHJONG_AGENT_WECHAT_ACCOUNT: 'old', MAHJONG_AGENT_WECHAT_START_CURSOR: '0' }
  const setup = createAgentWechatSetup(env, dir, file, () => seen.push('外发关闭'))
  const draft = { account: 'new', revision: 'initial', selections: [
    { id: 'same-id', mode: 'private' }, { id: 'room@chatroom', mode: 'observe' },
  ] }
  assert.equal((await setup.status()).active_account, 'old')
  const catalog = (await setup.inventory('new')).items
  assert.deepEqual(catalog.map(x => x.id), ['same-id', 'unidentified', 'room@chatroom'])
  assert.equal(catalog[0].has_thread, true)
  assert.equal(catalog[1].has_thread, false)
  assert.equal(catalog[1].name, catalog[1].id)
  await assert.rejects(setup.apply({ ...draft, account: 'old' }), { setupStatus: 409 })
  await assert.rejects(setup.apply({ ...draft, selections: [{ id: 'system', mode: 'private' }] }), { setupStatus: 409 })
  assert.equal(fs.existsSync(file), false)
  const controller = new AbortController()
  const watching = setup.watch(message => { seen.push(message); controller.abort() }, controller.signal)
  t.after(() => controller.abort())
  const applied = await setup.apply(draft)
  await watching
  assert.equal(applied.active_account, 'new')
  assert.equal(applied.account_changed, true)
  assert.deepEqual(JSON.parse(fs.readFileSync(file, 'utf8')).rooms, { 'room@chatroom': 'observe' })
  assert.equal(seen[0], '外发关闭')
  assert.equal(seen[1].agent_wechat_revision, applied.revision)
  assert.equal(seen[1].agent_wechat_account, 'new')
  assert.equal(seen[1].talker.friend, false, '接收白名单不能证明好友关系')
  assert.equal(seen[1].conversation_id, 'wechaty:contact:new:same-id')
  assert.ok(requests.filter(route => route.startsWith('/v1/accounts/new/events?')).every(route => route.includes('after=10')))
  await assert.rejects(setup.apply(draft), { setupStatus: 409 })
  await assert.rejects(setup.send('room@chatroom', '不可发送', true, 'key', {}), /does not permit/)
  assert.equal((await createAgentWechatSetup(env, dir, file, () => {}).status()).revision, applied.revision)
  await assert.rejects(setup.apply({ ...draft, revision: applied.revision, selections: [
    { id: 'room@chatroom', mode: 'observe', auto_reply: true },
  ] }), { setupStatus: 400 })
  const permitted = await setup.apply({ ...draft, revision: applied.revision, selections: [
    { id: 'same-id', mode: 'private' },
    { id: 'room@chatroom', mode: 'managed', auto_reply: true },
  ] })
  assert.equal(permitted.account_changed, false)
  assert.equal(seen.filter(event => event === '外发关闭').length, 1)
  const reply = { source: 'managed_group_domain', message_type: 'agent_reply', source_message_id: 'fresh',
    agent_wechat_account: 'new', agent_wechat_revision: permitted.revision }
  await assert.rejects(setup.send('room@chatroom', '旧授权', true, 'key', { ...reply, agent_wechat_revision: applied.revision }), /another account or setup revision/)
  await assert.rejects(setup.send('room@chatroom', '看板', true, 'key', { ...reply, message_type: 'game_board' }), /does not permit/)
  assert.equal((await setup.send('room@chatroom', '答复', true, 'key', reply)).id, 'fixture-reply')
  for (const source of ['delayed_input_batch_reply', 'recovered_agent_run_reply', 'agent_wechat_immediate_reply']) {
    const options = { source, agent_wechat_account: 'new', agent_wechat_revision: permitted.revision }
    const count = requests.filter(route => route.endsWith('/sends')).length
    await assert.rejects(setup.send('same-id', '旧账号的答复', false, source, {
      ...options, agent_wechat_account: 'old',
    }), /another account/)
    await assert.rejects(setup.send('same-id', '旧配置的答复', false, source, {
      ...options, agent_wechat_revision: applied.revision,
    }), /setup revision/)
    await assert.rejects(setup.send('same-id', '缺失来源', false, source, { source }), /another account/)
    assert.equal(requests.filter(route => route.endsWith('/sends')).length, count)
    assert.equal((await setup.send('same-id', '当前账号的答复', false, source, options)).id, 'fixture-reply')
  }
})

test('仅转发授权会话，保留群引用与发送确认', async (t) => {
  const data = fs.mkdtempSync(path.join(os.tmpdir(), 'mahjong-wechat-'))
  t.after(() => fs.rmSync(data, { recursive: true, force: true }))
  const token = path.join(data, 'token')
  fs.writeFileSync(token, 'synthetic-token')
  const group = 'test@chatroom'
  const source = [
    { threadId: 'unauthorized', messageId: 'hidden' },
    { threadId: 'alice', messageId: 'echo' },
    { threadId: 'alice', messageId: 'direct' },
    { threadId: 'alice', messageId: 'missing' },
    { threadId: group, messageId: 'group' },
  ]
  const receivedKeys = []
  let accepted = true
  let eventsRequested = 0
  let temporaryFailure = true
  const server = http.createServer(async (request, response) => {
    const url = new URL(request.url, 'http://localhost')
    const parts = url.pathname.split('/')
    let result
    if (request.headers.authorization !== 'Bearer synthetic-token') {
      response.writeHead(401).end()
      return
    }
    if (url.pathname === '/v1/status') result = { serverId: 'mock' }
    if (url.pathname === '/v1/accounts/test') result = { id: 'test' }
    if (parts[4] === 'events') {
      eventsRequested += 1
      const after = Number(url.searchParams.get('after'))
      result = { items: source.map((data, i) => ({
        data, cursor: String(i + 1), kind: 'message.committed',
      })).slice(after) }
    }
    if (parts[4] === 'messages') {
      const id = parts[5]
      if (id === 'missing') {
        response.writeHead(404, { 'Content-Type': 'application/json' })
        response.end(JSON.stringify({ error: { code: 'media_not_cached' } }))
        return
      }
      if (id === 'group' && temporaryFailure) {
        temporaryFailure = false
        response.writeHead(503, { 'Content-Type': 'application/json' })
        response.end(JSON.stringify({ error: { code: 'profile_io_failed' } }))
        return
      }
      const { threadId } = source.find(x => x.messageId === id)
      result = {
        id, accountId: 'test', threadId, senderId: 'alice',
        direction: id === 'echo' ? 'outgoing' : 'incoming', type: id === 'group' ? 49 : 1,
        appMessageType: 57, body: '2号我来',
        serverId: id === 'group' ? '9223372036854775807' : '123',
        quote: id === 'group' ? { serverId: '9223372036854775806', embeddedText: '旧看板' } : null,
      }
    }
    if (parts[4] === 'contacts') result = { nickname: '知', remark: '私人备注' }
    if (parts[4] === 'groups') result = { name: '测试群' }
    if (parts[4] === 'sends') {
      let raw = ''
      for await (const chunk of request) raw += chunk
      const sent = JSON.parse(raw)
      receivedKeys.push(request.headers['idempotency-key'])
      result = {
        accountId: 'test', destination: sent.threadId,
        state: accepted ? 'accepted' : 'needs_review', serverId: accepted ? '999' : null,
      }
    }
    response.writeHead(result ? 200 : 404, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify(result || {}))
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  t.after(() => server.close())
  const config = {
    MAHJONG_AGENT_WECHAT_URL: 'http://127.0.0.1:' + server.address().port,
    MAHJONG_AGENT_WECHAT_TOKEN_PATH: token,
    MAHJONG_AGENT_WECHAT_ACCOUNT: 'test',
    MAHJONG_AGENT_WECHAT_CONTACT_IDS: 'alice',
    MAHJONG_AGENT_WECHAT_ROOM_IDS: group,
  }
  const noCursor = createAgentWechatTransport(config, data)
  await assert.rejects(noCursor.watch(() => assert.fail('must not replay history'), new AbortController().signal),
    /START_CURSOR on first start/)
  assert.equal(eventsRequested, 0)
  const transport = createAgentWechatTransport({ ...config, MAHJONG_AGENT_WECHAT_START_CURSOR: '0' }, data)
  const abort = new AbortController()
  const incoming = []
  const watch = transport.watch(async message => {
    incoming.push(message)
    if (incoming.length === 2) abort.abort()
  }, abort.signal)
  await Promise.race([watch, delay(8000, null, { ref: false }).then(() => {
    abort.abort()
    throw new Error('Timed out reading test events')
  })])
  assert.equal(incoming.length, 2)
  assert.equal(incoming[0].conversation_id, 'wechaty:contact:alice')
  assert.equal(incoming[0].sender_name, '知')
  assert.equal(incoming[0].agent_wechat_account, 'test')
  assert.equal(incoming[0].agent_wechat_revision, 'initial')
  assert.equal(incoming[0].talker.friend, false)
  assert.equal(incoming[1].room.id, group)
  assert.equal(incoming[1].quoted_message.message_id, '9223372036854775806')
  assert.equal(incoming[1].message_id, '9223372036854775807')
  const cursor = fs.readdirSync(data).find(file => file.endsWith('.cursor'))
  assert.equal(fs.readFileSync(path.join(data, cursor), 'utf8'), '5')
  assert.equal(eventsRequested, 2, 'a transient failure must retry from the previous cursor')
  const skipped = fs.readFileSync(path.join(data, 'agent-wechat-skipped-events.jsonl'), 'utf8').trim().split('\n').map(JSON.parse)
  assert.deepEqual(skipped.map(event => [event.cursor,event.message_id,event.reason]), [['4','missing','media_not_cached']])
  await assert.rejects(transport.send('other', '正文', false, 'blocked'), /allowlist/)
  assert.deepEqual(await transport.send('alice', '有位置', false, 'reply'), { id: '999' })
  assert.deepEqual(await transport.send(group, '1号\n2号', true, 'board'), { id: '999' })
  await transport.send(group, '1号\n2号', true, 'board')
  assert.equal(receivedKeys[1], receivedKeys[2])
  assert.notEqual(receivedKeys[0], receivedKeys[1])
  accepted = false
  await assert.rejects(transport.send(group, '新版', true, 'new-board'), /needs_review/)
})

test('实际桥接在入站观察回写期间暂停时，不再调用微信发送接口', { timeout: 20000 }, async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mahjong-pause-'))
  const token = path.join(dir, 'token')
  fs.writeFileSync(token, 'native-fixture')
  const portServer = http.createServer()
  await new Promise(resolve => portServer.listen(0, '127.0.0.1', resolve))
  const bridgeUrl = `http://127.0.0.1:${portServer.address().port}`
  await new Promise(resolve => portServer.close(resolve))
  const control = async (route, body) => fetch(bridgeUrl + route, {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Mahjong-Bridge-Token': 'control-fixture' },
    body: JSON.stringify(body), signal: AbortSignal.timeout(2000),
  })
  let eventNumber = 0
  let pausePath = '/auto-send'
  let sends = 0
  let fixtureError
  const observations = []
  const server = http.createServer(async (request, response) => {
    try {
      const route = new URL(request.url, 'http://localhost')
      let raw = ''
      for await (const chunk of request) raw += chunk
      const body = raw ? JSON.parse(raw) : null
      let result = { ok: true }
      if (route.pathname === '/v1/status') result = { serverId: 'pause-fixture' }
      else if (route.pathname === '/v1/accounts/test') result = { id: 'test' }
      else if (route.pathname.endsWith('/events')) result = { items: eventNumber > Number(route.searchParams.get('after')) ? [
        { cursor: String(eventNumber), kind: 'message.committed', data: { threadId: 'alice', messageId: String(eventNumber) } },
      ] : [] }
      else if (route.pathname.includes('/messages/')) result = { id: String(eventNumber), accountId: 'test', threadId: 'alice',
        senderId: 'alice', type: 1, direction: 'incoming', body: '明晚有空位吗' }
      else if (route.pathname.includes('/contacts/')) result = { nickname: '样例客户', relation: 'unknown' }
      else if (route.pathname.endsWith('/sends')) {
        sends++
        result = { accountId: 'test', destination: body.threadId, state: 'accepted', serverId: 'confirmed-fixture' }
      } else if (route.pathname === '/api/channels/wechaty/raw') result = { ok: true, trace_id: 'trace_' + eventNumber,
        route_result: { agent_result: { final_reply: '明晚还有空位。' } } }
      else if (route.pathname.endsWith('/delivery-observation')) {
        if (body.outcome === 'sending') {
          const paused = await control(pausePath, { enabled: false })
          assert.equal(paused.status, 200)
        }
        observations.push(body)
      }
      response.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(result))
    } catch (error) {
      fixtureError = error
      response.writeHead(500).end('{}')
    }
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const apiUrl = `http://127.0.0.1:${server.address().port}`
  const child = spawn(process.execPath, [fileURLToPath(new URL('./src/bridge.mjs', import.meta.url))], {
    windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, MAHJONG_WECHAT_TRANSPORT: 'agent-wechat', MAHJONG_AGENT_WECHAT_URL: apiUrl,
      MAHJONG_AGENT_WECHAT_ACCOUNT: 'test', MAHJONG_AGENT_WECHAT_CONTACT_IDS: 'alice', MAHJONG_AGENT_WECHAT_ROOM_IDS: '',
      MAHJONG_AGENT_WECHAT_CONFIG_PATH: path.join(dir, 'missing-setup.json'), MAHJONG_AGENT_WECHAT_TOKEN_PATH: token,
      MAHJONG_AGENT_WECHAT_START_CURSOR: '0', MAHJONG_WECHATY_LOCAL_DATA_DIR: dir,
      MAHJONG_WECHATY_INGRESS_SPOOL_DIR: path.join(dir, 'spool'), MAHJONG_WECHATY_DELIVERY_LEDGER_PATH: path.join(dir, 'ledger'),
      MAHJONG_WECHATY_CONTACT_ALIASES_PATH: path.join(dir, 'aliases'), MAHJONG_WECHATY_CONTACT_ALIASES: '',
      MAHJONG_WECHATY_RAW_ENDPOINT: apiUrl + '/api/channels/wechaty/raw',
      MAHJONG_WECHATY_REFERENCE_ENDPOINT: apiUrl + '/api/message-references/link',
      MAHJONG_AGENT_API_TOKEN: 'runtime-fixture', MAHJONG_WECHATY_CONTROL_TOKEN: 'control-fixture',
      MAHJONG_WECHATY_OUTBOUND_ENABLED: 'true', MAHJONG_WECHATY_SEND_ENABLED: 'true', MAHJONG_WECHATY_AUTO_SEND_REPLY: 'true',
      MAHJONG_WECHATY_OUTBOUND_PORT: new URL(bridgeUrl).port },
  })
  let output = ''
  child.stdout.on('data', data => { output += data })
  child.stderr.on('data', data => { output += data })
  t.after(async () => {
    if (child.exitCode === null) {
      const exited = once(child, 'exit')
      child.kill()
      await exited
    }
    await new Promise(resolve => server.close(resolve))
    fs.rmSync(dir, { recursive: true, force: true })
  })
  const until = async predicate => {
    const deadline = Date.now() + 8000
    while (!predicate()) {
      if (fixtureError) throw fixtureError
      if (child.exitCode !== null) assert.fail(output)
      assert.ok(Date.now() < deadline, output)
      await delay(30)
    }
  }
  await until(() => output.includes('outbound server='))
  for (const paused of ['/auto-send', '/send-channel']) {
    pausePath = paused
    assert.equal((await control('/send-channel', { enabled: true })).status, 200)
    assert.equal((await control('/auto-send', { enabled: true })).status, 200)
    eventNumber++
    await until(() => observations.some(item => item.trace_id === 'trace_' + eventNumber && item.outcome === 'blocked'))
    const blocked = observations.find(item => item.trace_id === 'trace_' + eventNumber && item.outcome === 'blocked')
    assert.equal(blocked.reason, paused === '/auto-send' ? 'auto_send_disabled' : 'send_channel_disabled')
    assert.equal(sends, 0, '等待观察写入期间暂停必须生效')
    for (const source of ['delayed_input_batch_reply', 'recovered_agent_run_reply']) {
      const rejected = await control('/send', { to: 'alice', text: '稍后回复', source,
        agent_wechat_account: 'test', agent_wechat_revision: 'initial' })
      assert.equal(rejected.status, 500)
      assert.equal((await rejected.json()).ok, false)
      assert.equal(sends, 0)
    }
  }
  assert.equal((await control('/send-channel', { enabled: true })).status, 200)
  assert.equal((await control('/auto-send', { enabled: false })).status, 200)
  assert.equal((await control('/send', { to: 'alice', text: '人工答复', source: 'manual_operator' })).status, 200)
  assert.equal(sends, 1, '暂停自动外发时仍遵守单独的人工外发授权')
})
