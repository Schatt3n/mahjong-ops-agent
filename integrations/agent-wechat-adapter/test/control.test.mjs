import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { AgentWechatClient } from '../src/client.mjs'
import { ControlSwitches, startControlServer } from '../src/control.mjs'
import { Directory } from '../src/directory.mjs'

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
    refreshSavedContacts: async () => {},
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
  const dir = options.directory || makeDirectory(options.directoryOverrides)
  const client = {
    accountId: 'acc',
    contactPolicy: async () => ({
      direct: Boolean(config.outboundAllowlistPath),
      group: Boolean(config.outboundAllowlistPath),
      recipients: await client.outboundRecipients?.() || [],
      observed: [],
      groupPermissions: [],
    }),
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
  const fetchControl = (pathname, init) => fetch(`http://127.0.0.1:${port}${pathname}`, init)
  const call = async (pathname, init = {}) => {
    const response = await fetchControl(pathname, init)
    return { status: response.status, body: await response.json() }
  }
  try {
    await fn({ call, fetchControl, switches, client, config })
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

test('media requires the active account, exact saved thread, auth, and current workbench permission', async () => {
  const bytes = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aFfsAAAAASUVORK5CYII=', 'base64')
  const sha256 = createHash('sha256').update(bytes).digest('hex')
  const thread = 'wxid_li'
  const otherThread = 'wxid_wang'
  const id = 'saved-image'
  const hubRequests = []
  let revokeOnContent = false
  let permitted = true
  const sdk = new AgentWechatClient({
    baseUrl: 'http://stub',
    token: 'test-hub-token',
    accountId: 'acc',
    fetchImpl: async (url, init) => {
      const path = new URL(url).pathname
      assert.equal(init.headers.Authorization, 'Bearer test-hub-token')
      assert.equal(init.redirect, 'error')
      assert.ok(!init.method || init.method === 'GET')
      hubRequests.push(path)
      if (path === `/v1/accounts/acc/messages/${id}`) {
        return Response.json({ id, accountId: 'acc', threadId: thread, type: 3 })
      }
      if (path === `/v1/accounts/acc/messages/${id}/media`) {
        return Response.json({ items: [{ id: 'asset', accountId: 'acc', messageId: id, kind: 'image',
          variant: 'original', actualVariant: 'original', state: 'available', contentType: 'image/png',
          byteLength: bytes.length, sha256 }] })
      }
      assert.equal(path, '/v1/accounts/acc/media/asset/content')
      if (revokeOnContent) permitted = false
      return new Response(bytes, { headers: { 'content-type': 'image/png', 'content-length': String(bytes.length) } })
    },
  })
  const url = (account = 'acc', threadId = thread) => `/media?${new URLSearchParams({
    account_id: account, thread_id: threadId, message_id: id, representation: 'original',
  })}`
  const auth = { headers: { 'X-Mahjong-Agent-Token': 'test-runtime-token' } }
  await withServer({
    sendChannelEnabled: false,
    autoSendReply: false,
    configOverrides: { agentToken: 'test-runtime-token' },
    clientOverrides: {
      contactPolicy: async () => ({ direct: true, group: false, observed: [], groupPermissions: [],
        recipients: (permitted ? [thread, otherThread] : [otherThread]).map(thread_id => ({ thread_id })) }),
      getMessage: sdk.getMessage.bind(sdk),
      mediaAssets: sdk.mediaAssets.bind(sdk),
      mediaContent: sdk.mediaContent.bind(sdk),
      sendText: async () => assert.fail('Reading media must never send a message'),
    },
  }, async ({ call, fetchControl }) => {
    assert.equal((await call(url())).status, 401)
    assert.equal((await call(url('foreign-account'), auth)).status, 404)
    assert.equal((await call(url('acc', 'wxid_unapproved'), auth)).status, 404)
    assert.equal((await call(url() + '&message_id=duplicate', auth)).status, 400)
    assert.deepEqual(hubRequests, [])
    assert.equal((await call(url('acc', otherThread), auth)).status, 404)
    assert.deepEqual(hubRequests, [`/v1/accounts/acc/messages/${id}`])
    const result = await fetchControl(url(), auth)
    assert.equal(result.status, 200)
    assert.equal(result.headers.get('content-type'), 'image/png')
    assert.equal(result.headers.get('x-mahjong-media-variant'), 'original')
    assert.equal(result.headers.get('x-mahjong-media-representation'), 'original')
    assert.match(result.headers.get('cache-control'), /no-store/)
    assert.deepEqual(Buffer.from(await result.arrayBuffer()), bytes)
    revokeOnContent = true
    const revoked = await call(url(), auth)
    assert.equal(revoked.status, 404)
    assert.equal(revoked.body.code, 'media_source_mismatch')
  })
})

test('候选查询在发送关闭时可用，且只返回名单内的原生备注和名字别名匹配', async () => {
  const directory = new Directory({
    listContacts: async () => ({ items: [
      { username: 'wxid_unapproved', remark: '东区队长', alias: 'LeAd-Other' },
      { username: 'wxid_li', remark: '东区队长' },
    ] }),
    listGroups: async () => ({ items: [] }),
  })
  let recipients = [
    { thread_id: 'wxid_li', label: '常客', aliases: ['LeAd-East'] },
    { thread_id: '200@chatroom', label: '联赛联络群', aliases: ['LEAD-WEST'] },
  ]
  await withServer({ directory, sendChannelEnabled: false, autoSendReply: false,
    configOverrides: { outboundAllowlistPath: 'configured', controlToken: 'resolve-secret' },
    clientOverrides: { outboundRecipients: async () => recipients },
  }, async ({ call }) => {
    const lookup = (query, kind = '') => call(`/resolve?${new URLSearchParams({ query, kind })}`, {
      headers: { 'X-Mahjong-Agent-Token': 'resolve-secret' },
    })
    assert.equal((await call('/resolve?query=ead')).status, 401)
    const aliases = await lookup('eAd')
    assert.equal(aliases.status, 200)
    assert.deepEqual(new Set(aliases.body.candidates.map(candidate => candidate.id)), new Set(['wxid_li', '200@chatroom']))
    assert.equal(aliases.body.recommendedId, null)
    assert.deepEqual((await lookup('联络', 'group')).body.candidates.map(candidate => candidate.id), ['200@chatroom'])
    assert.deepEqual((await lookup('队长')).body.candidates.map(candidate => candidate.id), ['wxid_li'])
    assert.deepEqual((await lookup('wxid_unapproved')).body.candidates, [])
    recipients = []
    assert.deepEqual((await lookup('队长')).body.candidates, [])
  })
})

test('发送关闭时按精确联系人刷新，核对落盘后的备注并用于后续搜索', async () => {
  const paths = []
  const target = 'known@openim'
  const previous = { username: target, nickname: '李', alias: 'ali' }
  const operation = { id: 'op-1', accountId: 'acc', kind: 'metadata' }
  let saved = false
  const sdk = new AgentWechatClient({
    baseUrl: 'http://127.0.0.1:1', token: 'native-secret', accountId: 'acc',
    fetchImpl: async (url, init) => {
      const path = new URL(url).pathname
      paths.push([init.method, path])
      assert.equal(init.headers.Authorization, 'Bearer native-secret')
      if (path === '/v1/accounts/acc/contacts/known%40openim') {
        return Response.json(saved ? { ...previous, remark: '新备注', internal: 'hidden' } : previous)
      }
      if (path === '/v1/accounts/acc/contacts/known%40openim/refresh') {
        assert.deepEqual(JSON.parse(init.body), {})
        return Response.json({ ...operation, state: 'queued' }, { status: 202 })
      }
      if (path === '/v1/accounts/acc/operations/op-1') {
        saved = true
        return Response.json({ ...operation, state: 'succeeded', result: { profileProjection: 'complete' } })
      }
      if (path.endsWith('/contacts')) {
        const contact = saved ? { ...previous, remark: '新备注' } : previous
        const query = new URL(url).searchParams.get('search') || ''
        const matches = Object.values(contact).some(value => value.includes(query))
        return Response.json({ items: matches ? [contact] : [], nextCursor: null })
      }
      if (path.endsWith('/groups')) return Response.json({ items: [] })
      assert.fail(`unexpected native path: ${path}`)
    },
  })
  const directory = new Directory(sdk)
  const other = { username: 'wxid_other', remark: '其他联系人' }
  directory.contacts.push(previous, other)
  directory.contactsLoadedAt = directory.groupsLoadedAt = Date.now()
  await withServer({
    directory, sendChannelEnabled: false, autoSendReply: false,
    configOverrides: { controlToken: 'control-secret', outboundAllowlistPath: 'configured' },
    clientOverrides: {
      outboundRecipients: async () => [{ thread_id: target, label: '定向候选', aliases: ['李姐'] }],
      sendText: async () => assert.fail('刷新联系人不能发送消息'),
    },
  }, async ({ call, switches }) => {
    const refresh = (recipientId, token = 'control-secret') => call('/contacts/refresh', {
      method: 'POST', body: JSON.stringify({ recipient_id: recipientId }),
      headers: { 'X-Mahjong-Agent-Token': token },
    })
    assert.equal((await refresh(target, '')).status, 401)
    for (const id of ['100@chatroom', '100@im.chatroom', '李 姐', '..', '/wrong']) {
      assert.equal((await refresh(id)).status, 400)
    }
    assert.equal((await refresh('李姐')).status, 403)
    assert.equal((await refresh('wxid_other')).status, 403)
    assert.deepEqual(paths, [])
    assert.deepEqual(await refresh(target), {
      status: 200, body: {
        ok: true, projection: 'complete',
        recipient: { id: target, kind: 'direct', displayName: '新备注', remark: '新备注', nickname: '李', alias: 'ali' },
      },
    })
    assert.deepEqual(paths, [
      ['GET', '/v1/accounts/acc/contacts/known%40openim'],
      ['POST', '/v1/accounts/acc/contacts/known%40openim/refresh'],
      ['GET', '/v1/accounts/acc/operations/op-1'],
      ['GET', '/v1/accounts/acc/contacts/known%40openim'],
    ])
    for (const query of ['新备注', previous.nickname]) {
      const result = await call(`/resolve?${new URLSearchParams({ query, kind: 'direct' })}`, {
        headers: { 'X-Mahjong-Agent-Token': 'control-secret' },
      })
      assert.deepEqual(result.body.candidates.map(item => [item.id, item.displayName]), [[target, '新备注']])
    }
    assert.equal(directory.contact(other.username), other)
    assert.equal(switches.sendChannelEnabled, false)
    assert.equal(switches.autoSendReply, false)
  })
})

test('刷新未知联系人返回资料未找到，等待超时返回待核对', async () => {
  const directory = makeDirectory({
    refreshContact: async id => {
      if (id === 'wxid_pending') return { pending: true }
      throw Object.assign(new Error('native response withheld'), { code: 'refresh_contact_not_found' })
    },
  })
  await withServer({ directory, configOverrides: { agentToken: 'runtime-secret' } }, async ({ call }) => {
    const refresh = (id, token = 'runtime-secret') => call('/contacts/refresh', {
      method: 'POST', body: JSON.stringify({ recipient_id: id }),
      headers: { 'X-Mahjong-Agent-Token': token },
    })
    assert.equal((await refresh('wxid_pending', '')).status, 401)
    assert.deepEqual(await refresh('wxid_pending'), { status: 202, body: { ok: false, pending: true } })
    const missing = await refresh('wxid_unknown')
    assert.equal(missing.status, 404)
    assert.deepEqual(missing.body, { ok: false, error: '未找到可刷新的微信联系人。' })
  })
})

test('店主会话只查本机精确白名单，过滤无关账号和已撤销权限', async () => {
  const inspected = []
  let direct = true
  let group = false
  const policy = {
    direct, group, recipients: [
      { thread_id: 'wxid_li', label: '保存的备注' },
      { thread_id: '100@chatroom', label: '旧白名单群' },
      { thread_id: '200@chatroom', label: '已改为观察的群' },
    ], observed: [{ thread_id: 'wxid_wang', label: '观察' }],
    managedOnly: [{ thread_id: 'wxid_qin', label: '仅托管' }], groupPermissions: [
      { thread_id: '200@chatroom', mode: 'observe', label: '观察群' },
      { thread_id: '300@chatroom', mode: 'allowlist', label: '人工白名单群' },
    ],
  }
  await withServer({
    configOverrides: { agentToken: 'runtime-secret' },
    clientOverrides: {
      contactPolicy: async () => ({ ...policy, direct, group }),
      getThread: async id => {
        inspected.push(id)
        return { id, accountId: id === '300@chatroom' ? 'wrong' : 'acc',
          kind: id.startsWith('wxid') ? 'direct' : 'group', displayName: '当前保存资料',
          lastMessage: { createdAt: '2026-01-02T12:00:00+08:00', direction: 'outgoing' } }
      },
      listThreadMessages: async () => {
        direct = false
        return { items: [{ id: 'private-message', accountId: 'acc', threadId: 'wxid_li', direction: 'incoming',
          createdAt: '2026-01-02T04:00:00Z', body: '撤销后不应返回这里的内容' }], nextCursor: null }
      },
    },
  }, async ({ call }) => {
    const headers = { 'X-Mahjong-Agent-Token': 'runtime-secret' }
    assert.equal((await call('/console-conversations')).status, 401)
    const first = (await call('/console-conversations', { headers })).body
    assert.deepEqual(first, { ok: true, account_id: 'acc', partial: true, items: [
      { id: 'wxid_li', kind: 'direct', displayName: '当前保存资料', last_at: '2026-01-02T04:00:00.000Z' },
      { id: '300@chatroom', kind: 'group', displayName: '人工白名单群', last_at: null },
    ] })
    assert.deepEqual(inspected, ['wxid_li', '300@chatroom'])
    group = true
    const scoped = (await call('/console-conversations', { headers })).body
    assert.deepEqual(scoped.items.map(row => row.id), ['wxid_li', '100@chatroom', '300@chatroom'])
    for (const [id, kind] of [['wxid_wang', 'direct'], ['wxid_qin', 'direct'], ['200@chatroom', 'group']]) {
      assert.equal((await call(`/console-conversation?thread_id=${encodeURIComponent(id)}&kind=${kind}`, { headers })).status, 404)
    }
    assert.equal((await call('/console-conversation?thread_id=wxid_li&kind=group', { headers })).status, 400)
    assert.equal((await call('/console-conversation?thread_id=wxid_li&kind=direct', { headers })).status, 404)
  })
})

test('店主会话按原生游标读取，同账号同会话才显示，真实发送回执可标明店主', async () => {
  const temporary = await mkdtemp(join(tmpdir(), 'wechat-console-history-'))
  const ledgerPath = join(temporary, 'delivery.jsonl')
  const id = 'wxid_li'
  const seed = { accountId: 'acc', threadId: id, direction: 'incoming', createdAt: '2026-01-02T04:00:00Z', type: 1 }
  const receipt = { id: 'fixture-send', accountId: 'acc', destination: id, state: 'accepted', localMessageId: 'owner-message' }
  try {
    await withServer({
      configOverrides: { agentToken: 'runtime-secret', deliveryLedgerPath: ledgerPath },
      clientOverrides: {
        contactPolicy: async () => ({ direct: true, group: false,
          recipients: [{ thread_id: id, label: '李姐' }], observed: [], groupPermissions: [] }),
        sendText: async () => ({ id: receipt.id }),
        waitForSend: async () => receipt,
        listThreadMessages: async (threadId, { cursor, limit }) => {
          assert.equal(threadId, id)
          assert.equal(limit, 14)
          assert.equal(cursor, 'previous+cursor=')
          return { items: [
            { ...seed, id: 'newer-message', body: '客户内容', raw: 'internal' },
            { ...seed, id: 'owner-message', direction: 'outgoing', body: '店主内容' },
            { ...seed, id: 'unmarked', direction: 'outgoing', body: '发送者不明' },
            { ...seed, id: 'image', type: 3, body: '<msg>internal file data</msg>' },
            { ...seed, id: 'sticker', type: 49, appMessageType: 8, media: [{ url: 'https://private.invalid/file' }] },
            { ...seed, id: 'recalled-sticker', type: 47, recall: { recalledAt: 'known-time' } },
            { ...seed, id: 'quote', type: 49, appMessageType: 57, body: '这是可读的引用回复' },
            { ...seed, id: 'quote-xml', type: 49, appMessageType: 57, body: '<msg>internal quote data</msg>' },
            { ...seed, id: 'long-emoji', body: '字'.repeat(9_999) + '😀后续' },
            { ...seed, id: 'should-filter-account', accountId: 'other', body: '跨账号禁止' },
            { ...seed, id: 'should-filter-thread', threadId: 'wxid_other', body: '串话禁止' },
            { ...seed, id: 'should-filter-date', createdAt: '2026-01-02T04:00:00', body: '时间未核对' },
            { ...seed, id: 'should-filter-system', type: 10000, body: '系统通知' },
          ], nextCursor: 'next+page=' }
        },
      },
    }, async ({ call }) => {
      const sent = await call('/send', { method: 'POST', body: JSON.stringify({ to: id,
        target_type: 'direct', exact_target: true, text: '店主内容' }) })
      assert.equal(sent.status, 200)
      const auth = { 'X-Mahjong-Agent-Token': 'runtime-secret' }
      const query = new URLSearchParams({ thread_id: id, kind: 'direct', cursor: 'previous+cursor=', limit: '14' })
      assert.equal((await call(`/console-conversation?${query}`)).status, 401)
      const read = await call(`/console-conversation?${query}`, { headers: auth })
      assert.equal(read.status, 200)
      assert.equal(read.body.account_id, 'acc')
      assert.deepEqual(read.body.items.filter(row => row.media).map(row => [row.id, row.media]),
        [['image', { kind: 'image' }], ['sticker', { kind: 'sticker' }]])
      assert.deepEqual(read.body.items.map(row => [row.id, row.direction, row.author, row.text]), [
        ['newer-message', 'inbound', 'contact', '客户内容'], ['owner-message', 'outbound', 'owner', '店主内容'],
        ['unmarked', 'outbound', 'outbound', '发送者不明'], ['image', 'inbound', 'contact', '[图片]'],
        ['sticker', 'inbound', 'contact', '[表情]'], ['recalled-sticker', 'inbound', 'contact', '消息已撤回'],
        ['quote', 'inbound', 'contact', '这是可读的引用回复'], ['quote-xml', 'inbound', 'contact', '[引用回复]'],
        ['long-emoji', 'inbound', 'contact', '字'.repeat(9_999) + '…'],
      ])
      assert.equal(read.body.nextCursor, 'next+page=')
      assert.equal(read.body.items[0].raw, undefined)
      assert.equal(read.body.items[0].accountId, undefined)
    })
  } finally {
    await rm(temporary, { recursive: true, force: true })
  }
})

test('群历史按个人备注、同群名片、个人昵称排序，仅给当前白名单好友精确私聊入口', async () => {
  const group = '100@chatroom'
  const contactCalls = []
  const memberCalls = []
  const ids = ['wxid_note', 'wxid_room', 'wxid_nick', 'wxid_removed', 'wxid_mismatch']
  const [note, room, nick, removed, mismatch] = ids
  const allowedDirect = new Set([note, room, removed, mismatch])
  const names = {
    [note]: { remark: '个人备注', roomDisplayName: '群名片被备注覆盖', nickname: '昵称', relation: 'friend' },
    [room]: { roomDisplayName: '这个群的名片', nickname: '另有个人昵称', relation: 'non_contact' },
    [nick]: { nickname: '单独的昵称', relation: 'friend' },
    [removed]: { roomDisplayName: '异群名片不能展示', nickname: '撤销后的昵称', relation: 'friend' },
    [mismatch]: { remark: '错人备注不能展示', nickname: '群资料的个人昵称', relation: 'friend' },
  }
  const seed = { threadId: group, accountId: 'acc', type: 1, direction: 'incoming', createdAt: '2026-01-02T08:00:00Z', body: '测试' }
  await withServer({
    configOverrides: { agentToken: 'runtime-secret' },
    clientOverrides: {
      contactPolicy: async () => ({ direct: true, group: false, observed: [],
        recipients: [...allowedDirect].map(thread_id => ({ thread_id, label: '权限记录' })),
        groupPermissions: [{ thread_id: group, mode: 'allowlist', label: '测试群' }] }),
      listThreadMessages: async () => ({ items: [
        ...ids.map((senderId, index) => ({ ...seed, id: `member-${index}`, senderId })),
        { ...seed, id: 'repeat', senderId: note },
        { ...seed, id: 'outgoing', senderId: note, direction: 'outgoing' },
      ], nextCursor: null }),
      getContact: async id => {
        contactCalls.push(id)
        const name = names[id]
        return { id: id === mismatch ? 'wxid_other' : id, username: id,
          remark: name.remark, nickname: name.nickname, relation: name.relation, alias: '个人微信号' }
      },
      getGroupMember: async (groupId, id) => {
        memberCalls.push([groupId, id])
        if (id === removed) allowedDirect.delete(removed)
        return { groupId: id === removed ? '200@chatroom' : groupId, username: id, state: 'present',
          member: { id, username: id, nickname: names[id].nickname,
            roomDisplayName: names[id].roomDisplayName, displayName: '不能推断为群名片', alias: '个人微信号' } }
      },
      sendText: async () => assert.fail('读取群历史不能发送消息'),
    },
  }, async ({ call }) => {
    const query = new URLSearchParams({ thread_id: group, kind: 'group' })
    const headers = { 'X-Mahjong-Agent-Token': 'runtime-secret' }
    const read = await call(`/console-conversation?${query}`, { headers })
    assert.equal(read.status, 200)
    assert.deepEqual(read.body.items.map(row => row.speaker), [
      '个人备注', '这个群的名片', '单独的昵称', '撤销后的昵称', '群资料的个人昵称', '个人备注', null,
    ])
    const privateTarget = { id: note, kind: 'direct', displayName: '个人备注' }
    assert.deepEqual(read.body.items.map(row => row.direct_recipient), [
      privateTarget, undefined, undefined, undefined, undefined, privateTarget, undefined,
    ])
    assert.deepEqual(new Set(contactCalls), new Set(ids))
    assert.equal(contactCalls.length, ids.length)
    assert.equal(memberCalls.length, ids.length)
    assert.ok(memberCalls.every(([groupId]) => groupId === group))
    assert.equal(read.body.items[1].senderId, undefined)
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
        if (calls.length === 1) {
          return { id: 'send-1', state: 'accepted', accountId: 'acc', destination: 'wxid_li', serverId: 'server-1' }
        }
        if (calls.length === 2) return { state: 'accepted', accountId: 'other', destination: 'wxid_li', serverId: 'foreign' }
        return null
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
      status: 200, body: { ok: true, delivery_state: 'accepted', message_id: 'server-1' },
    })
    assert.deepEqual(calls, [[payload, result]])
    assert.deepEqual(await deliver({ payload, result }, token), {
      status: 200, body: { ok: true, delivery_state: 'needs_review' },
    })
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
      const send = (to, targetType, source, exactTarget) => call('/send', { method: 'POST', body: JSON.stringify({
        to, target_type: targetType, text: '测试用例', source, exact_target: exactTarget,
      }) })
      assert.equal((await send('老李', 'contact')).status, 200)
      assert.equal((await send('周末麻将群', 'room')).status, 200)
      assert.equal((await send('周末麻将群', 'contact')).status, 403)
      assert.equal((await send('wxid_wang', 'contact')).status, 403)
      assert.equal((await send('101@chatroom', 'room')).status, 403)
      assert.equal((await send('老李', 'contact', 'delayed_input_batch_reply')).status, 409)
      assert.equal((await send('老李', 'direct', undefined, true)).status, 403)
      assert.equal((await send(one.thread_id, 'direct', undefined, true)).status, 200)
      await policy('acc', [group])
      assert.equal((await send(one.thread_id, 'direct', undefined, true)).status, 403)
      await assert.rejects(sdk.sendText({ threadId: one.thread_id, text: '测试用例', idempotencyKey: 'revoked' }),
        { code: 'outbound_recipient_denied' })
      await policy('other-account', [one, group])
      assert.equal((await send('老李', 'contact')).status, 403)
      await rm(path)
      assert.equal((await send('周末麻将群', 'room')).status, 403)
      assert.deepEqual(upstream, ['wxid_li', '100@chatroom', 'wxid_li'])
    })
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('联系人管理用真实临时文件热更新私聊，保存不打开开关或改变群发送规则', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'wechat-contact-policy-'))
  const path = join(dir, 'contact_permissions.local.json')
  const contacts = [
    { username: 'wxid_li', nickname: '小李', remark: '李姐' },
    { username: 'wxid_wang', nickname: '小王', remark: '王老板' },
    { username: 'wxid_qin', nickname: '小秦', remark: '秦老板' },
  ]
  const group = { id: '100@chatroom', displayName: '周末麻将群' }
  const submitted = []
  let catalogReads = 0
  const sdk = new AgentWechatClient({
    baseUrl: 'http://stub', accountId: 'acc', managedContactPolicyPath: path,
    fetchImpl: async (url, init) => {
      if (url.includes('/contacts')) {
        catalogReads += 1
        return Response.json({ items: contacts })
      }
      if (url.includes('/groups')) return Response.json({ items: [group] })
      if (url.endsWith('/sends')) {
        submitted.push(JSON.parse(init.body).threadId)
        return Response.json({ id: 'send-1' })
      }
      assert.fail(`unexpected native request ${url}`)
    },
  })
  try {
    await withServer({
      autoSendReply: false, sendChannelEnabled: false, directory: new Directory(sdk),
      configOverrides: { agentToken: 'runtime-secret' },
      clientOverrides: {
        contactPolicy: sdk.contactPolicy.bind(sdk),
        saveContactPermission: sdk.saveContactPermission.bind(sdk),
        sendText: sdk.sendText.bind(sdk),
      },
    }, async ({ call, switches }) => {
      const auth = { 'X-Mahjong-Agent-Token': 'runtime-secret' }
      const search = (query) => call(`/contact-policy?${new URLSearchParams({ query })}`, { headers: auth })
      const save = (recipient_id, mode, automation = {}) => call('/contact-policy', {
        headers: auth, method: 'POST', body: JSON.stringify({ recipient_id, mode, ...automation }),
      })
      const send = (to, target_type, source) => call('/send', { method: 'POST', body: JSON.stringify({
        to, target_type, source, exact_target: true, text: '测试用例',
      }) })

      assert.equal((await call('/contact-policy')).status, 401)
      assert.equal((await call('/contact-policy', { method: 'POST', body: '{}' })).status, 401)
      assert.equal((await call('/health')).body.direct_contact_policy_enabled, false)
      assert.equal((await search('小李')).body.candidates[0].mode, 'unassigned')
      assert.equal((await save('100@chatroom', 'allowlist')).status, 400)
      assert.equal((await save('wxid_other', 'observe')).status, 404)
      assert.equal((await save('wxid_other', 'managed_only')).status, 404)
      assert.equal((await save('wxid_li', 'observe', { auto_reply: true })).status, 400)
      assert.equal((await save('wxid_li', 'allowlist', { managed: false, auto_reply: true })).status, 400)
      assert.equal((await save('wxid_qin', 'managed_only', { auto_reply: true })).status, 400)
      assert.equal((await save('wxid_qin', 'managed_only', { managed: false })).status, 400)
      const results = await Promise.all([
        save('wxid_li', 'allowlist', { managed: false, auto_reply: false }),
        save('wxid_wang', 'observe'),
      ])
      assert.deepEqual(results.map(result => result.status), [200, 200])
      assert.equal(results[0].body.recipient.mode, 'allowlist')
      assert.deepEqual(results[0].body.recipient.automation, { managed: false, auto_reply: false })
      assert.equal(results[1].body.recipient.mode, 'observe')
      if (process.platform !== 'win32') {
        assert.equal((await stat(path)).mode & 0o777, 0o600)
      }
      const listing = (await search('')).body
      assert.equal(listing.managed, true)
      assert.deepEqual(new Set(listing.entries.map(entry => entry.mode)), new Set(['allowlist', 'observe']))
      assert.equal((await search('王老板')).body.candidates[0].mode, 'observe')
      assert.equal((await call('/resolve?query=wxid_wang')).body.candidates.length, 0)
      const groupSearch = new URLSearchParams({ query: '麻将群' })
      assert.equal((await call(`/resolve?${groupSearch}`)).body.candidates[0].id, group.id)
      assert.equal((await send('wxid_li', 'direct')).status, 409)
      assert.deepEqual(submitted, [])

      switches.sendChannelEnabled = true
      assert.equal((await save('wxid_qin', 'allowlist')).status, 200)
      const managedOnly = await save('wxid_qin', 'managed_only', { managed: true, auto_reply: false })
      assert.equal(managedOnly.status, 200)
      assert.equal(managedOnly.body.recipient.mode, 'managed_only')
      assert.deepEqual(managedOnly.body.recipient.automation, { managed: true, auto_reply: false })
      assert.equal((await search('秦老板')).body.candidates[0].mode, 'managed_only')
      const roster = await call('/permission-roster?view=policy', { headers: auth })
      assert.equal(roster.body.items.find(row => row.id === 'wxid_qin').mode, 'managed_only')
      assert.equal((await call('/resolve?query=wxid_qin')).body.candidates.length, 0)
      assert.equal((await send('wxid_qin', 'direct')).status, 403)
      assert.equal((await send('wxid_li', 'direct')).status, 200)
      assert.equal((await send('wxid_li', 'direct', 'delayed_input_batch_reply')).status, 409)
      assert.equal((await send('wxid_wang', 'direct')).status, 403)
      assert.equal((await send(group.id, 'group')).status, 409)
      switches.autoSendReply = true
      assert.equal((await send('wxid_qin', 'direct', 'delayed_input_batch_reply')).status, 403)
      for (const manual of [true, false]) {
        await assert.rejects(sdk.sendText({ threadId: 'wxid_qin', text: '测试', idempotencyKey: 'managed-only', manual }),
          { code: 'outbound_recipient_denied' })
      }
      const allowManual = await save('wxid_qin', 'allowlist')
      assert.deepEqual(allowManual.body.recipient.automation, { managed: true, auto_reply: false })
      assert.equal((await send('wxid_qin', 'direct', 'delayed_input_batch_reply')).status, 403)
      assert.equal((await save('wxid_qin', 'managed_only')).status, 200)
      assert.equal((await send('wxid_li', 'direct', 'delayed_input_batch_reply')).status, 403)
      assert.equal((await send(group.id, 'group')).status, 200)
      // 联系人已从通讯录消失时仍可撤销，重新授权必须读取当前通讯录。
      contacts.shift()
      const beforeRevoke = catalogReads
      assert.equal((await save('wxid_li', 'observe')).status, 200)
      assert.equal(catalogReads, beforeRevoke)
      assert.equal((await save('wxid_li', 'allowlist')).status, 404)
      assert.ok(catalogReads > beforeRevoke)
      contacts.push({ id: 'wxid_li', deleted: true })
      assert.equal((await save('wxid_li', 'allowlist')).status, 404)
      assert.equal((await send('wxid_li', 'direct')).status, 403)
      await assert.rejects(sdk.sendText({ threadId: 'wxid_li', text: '测试', idempotencyKey: 'blocked' }),
        { code: 'outbound_recipient_denied' })
      assert.deepEqual(submitted, ['wxid_li', group.id])
      const reopened = new AgentWechatClient({ baseUrl: 'http://stub', accountId: 'acc', managedContactPolicyPath: path })
      assert.equal((await reopened.contactPolicy()).observed.length, 2)
      assert.deepEqual((await reopened.contactPolicy()).managedOnly, [{ thread_id: 'wxid_qin', label: '秦老板' }])
      // 仅托管是更窄的历史授权，重复编辑文件也不会把它放回外发名单。
      const document = JSON.parse(await readFile(path, 'utf-8'))
      document.recipients.push({ thread_id: 'wxid_qin', label: '秦老板' })
      await writeFile(path, JSON.stringify(document))
      assert.equal((await reopened.contactPolicy()).recipients.length, 0)
      document.observed.push({ thread_id: 'wxid_qin', label: '秦老板' })
      await writeFile(path, JSON.stringify(document))
      assert.equal((await search('秦老板')).body.candidates[0].mode, 'observe')
      assert.deepEqual((await call('/resolve?query=wxid_qin')).body.candidates, [])
    })
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('编辑既有名单保留群别名和其他字段，损坏及账号不符的文件不会被替换', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'wechat-contact-policy-scoped-'))
  const path = join(dir, 'configured.json')
  const group = { thread_id: '100@chatroom', label: '联络群', aliases: ['周末群'] }
  const previous = { thread_id: 'wxid_li', label: '旧标签', aliases: ['老李'] }
  const sdk = new AgentWechatClient({ baseUrl: 'http://stub', accountId: 'acc', outboundAllowlistPath: path,
    fetchImpl: async () => assert.fail('不存在允许发送的委托') })
  try {
    await writeFile(path, JSON.stringify({ account_id: 'acc', recipients: [group, previous], extra: 'keep' }))
    const written = await sdk.saveContactPermission('wxid_li', '李姐', 'allowlist')
    assert.equal(written.group, true)
    assert.deepEqual(written.recipients[0], group)
    assert.deepEqual(written.recipients[1], previous)
    assert.equal(written.document.extra, 'keep')
    sdk.checkForOwnerReply = async () => {
      await sdk.saveContactPermission('wxid_li', '李姐', 'observe')
      return false
    }
    await assert.rejects(sdk.sendText({ threadId: 'wxid_li', text: '测试', idempotencyKey: 'verify-revoke', replySource: {} }),
      { code: 'outbound_recipient_denied' })
    assert.deepEqual((await sdk.contactPolicy()).observed, [previous])
    assert.deepEqual((await sdk.saveContactPermission('wxid_li', '新备注', 'allowlist')).recipients, [group, previous])
    for (const invalid of ['{broken', JSON.stringify({ account_id: 'other', recipients: [group] })]) {
      await writeFile(path, invalid)
      assert.equal(await sdk.saveContactPermission('wxid_li', '李姐', 'allowlist'), null)
      assert.equal(await readFile(path, 'utf8'), invalid)
      assert.deepEqual(await sdk.outboundRecipients(), [])
    }
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('权限名单列出现有私聊和群，浏览通讯录只读一页且排除删除标记', async () => {
  const readPages = []
  await withServer({
    configOverrides: { agentToken: 'runtime-secret', dryRun: true },
    directoryOverrides: {
      contacts: [
        { username: 'wxid_li', remark: '新备注', nickname: '小李', alias: 'simple-li', privateField: 'hidden' },
        { username: 'wxid_wang', remark: '已删除的备注', deleted: true },
        { username: 'wxid_not_scoped', remark: '新备注' },
      ],
      refreshContact: async () => assert.fail('浏览权限名单不能发起主动刷新'),
    },
    clientOverrides: {
      contactPolicy: async () => ({
        direct: true, group: true, managed: true,
        recipients: [{ thread_id: 'wxid_li', label: '李姐' }, { thread_id: '100@chatroom', label: '原有群' }],
        observed: [{ thread_id: 'wxid_wang', label: '老王' }],
        groupPermissions: [{ thread_id: '200@chatroom', label: '观察群', mode: 'observe' }],
      }),
      listContacts: async (options) => {
        readPages.push(['direct', options])
        return { items: [{ username: 'wxid_li', nickname: '小李', remark: '李姐', privacy: '不进入回执' },
          { id: 'wxid_deleted', deleted: true }], nextCursor: 'opaque+next=' }
      },
      listGroups: async (options) => {
        readPages.push(['group', options])
        return { items: [{ roomId: '200@chatroom', displayName: '观察群', ownerId: '不进入回执' }], nextCursor: null }
      },
      saveGroupPermission: async () => assert.fail('只读演练不能调用写入'),
    },
  }, async ({ call }) => {
    const headers = { 'X-Mahjong-Agent-Token': 'runtime-secret' }
    const roster = (args) => call(`/permission-roster?${new URLSearchParams(args)}`, { headers })
    assert.equal((await call('/permission-roster')).status, 401)
    assert.equal((await roster({ view: 'invalid' })).status, 400)
    assert.equal((await roster({ view: 'contacts', limit: 101 })).status, 400)
    const policy = (await roster({ view: 'policy', limit: 2 })).body
    assert.equal(policy.account_id, 'acc')
    assert.deepEqual(policy.items.map(row => [row.kind, row.mode]), [['direct', 'allowlist'], ['direct', 'observe']])
    assert.deepEqual(policy.items.map(row => [row.displayName, row.remark, row.nickname, row.alias]),
      [['新备注', '新备注', '小李', 'simple-li'], ['老王', null, null, null]])
    assert.equal(policy.items[0].privateField, undefined)
    for (const query of ['新备注', '小李', 'simple-LI', '李姐']) {
      const found = (await roster({ view: 'policy', query, limit: 1 })).body
      assert.deepEqual(found.items.map(row => row.id), ['wxid_li'])
      assert.equal(found.nextCursor, null)
    }
    assert.deepEqual((await roster({ view: 'policy', query: '已删除的备注' })).body.items, [])
    assert.equal(policy.nextCursor, '2')
    const inherited = (await roster({ view: 'policy', cursor: policy.nextCursor })).body
    assert.deepEqual(inherited.items.map(row => [row.displayName, row.mode, row.override]),
      [['原有群', 'allowlist', null], ['观察群', 'observe', 'observe']])
    assert.deepEqual(readPages, [])
    const contacts = (await roster({ view: 'contacts', query: '小李', cursor: 'first+page=', limit: 2 })).body
    assert.equal(contacts.items[0].displayName, '李姐')
    assert.equal(contacts.items.length, 1)
    assert.equal(contacts.items[0].privacy, undefined)
    assert.equal(contacts.nextCursor, 'opaque+next=')
    const groups = (await roster({ view: 'groups', query: '观察' })).body
    assert.equal(groups.items[0].mode, 'observe')
    assert.equal(groups.items[0].ownerId, undefined)
    assert.deepEqual(readPages, [
      ['direct', { search: '小李', cursor: 'first+page=', limit: 2 }],
      ['group', { search: '观察', cursor: null, limit: 40 }],
    ])
    const save = (auth) => call('/group-permission', {
      method: 'POST', headers: auth, body: JSON.stringify({ recipient_id: '200@chatroom', mode: 'allowlist' }),
    })
    assert.equal((await save()).status, 401)
    assert.equal((await save(headers)).status, 409)
    assert.equal((await call('/contact-policy/refresh', {
      method: 'POST', headers, body: JSON.stringify({ recipient_id: 'wxid_li' }),
    })).status, 409)
  })
})

test('个人联系人列表和发送候选排除公众号、服务号和系统账号，旧名单仍可撤销', async () => {
  const legacy = { username: 'gh_4b91e624fd20', nickname: '旧版公众号' }
  const personal = { username: 'wxid_li', nickname: '小李', alias: 'gh_personal', classification: { category: 'unknown' } }
  const wecom = { username: 'seller@openim', nickname: '店员', classification: { category: 'openim' } }
  const accounts = [
    { username: 'wxid_official', nickname: '公众号', classification: { category: 'business', businessKind: 'official_account' } },
    { username: 'wxid_service', nickname: '服务号', classification: { category: 'business', businessKind: 'service_account' } },
    { username: 'filehelper', nickname: '文件传输助手', classification: { category: 'system' } },
    legacy,
  ]
  const contacts = [...accounts, personal, wecom]
  let revoked = false
  const existing = { thread_id: legacy.username, label: '历史登记' }
  const otherEntries = [accounts[0], accounts[1], accounts[2], personal]
    .map(item => ({ thread_id: item.username, label: item.nickname }))
  const client = {
    listContacts: async ({ search, cursor }) => ({
      items: cursor ? [] : contacts.filter(item => !search || item.username === search),
      nextCursor: search || cursor ? null : 'contacts=next+',
    }),
    listGroups: async () => ({ items: [], nextCursor: null }),
    contactPolicy: async () => ({
      managed: true, direct: true, group: false,
      recipients: revoked ? otherEntries : [existing, ...otherEntries],
      observed: revoked ? [existing] : [], groupPermissions: [],
    }),
    saveContactPermission: async (id, _label, mode) => {
      assert.equal(id, legacy.username)
      assert.equal(mode, 'observe')
      revoked = true
      return client.contactPolicy()
    },
  }
  await withServer({
    configOverrides: { agentToken: 'runtime-secret' },
    directory: new Directory(client), clientOverrides: client,
  }, async ({ call }) => {
    const headers = { 'X-Mahjong-Agent-Token': 'runtime-secret' }
    const roster = view => call(`/permission-roster?${new URLSearchParams({ view })}`, { headers })
    const expected = [personal.username, wecom.username]
    const listed = await roster('contacts')
    assert.equal(listed.status, 200)
    assert.deepEqual(listed.body.items.map(row => row.id), expected)
    assert.equal(listed.body.nextCursor, 'contacts=next+')
    assert.deepEqual((await call('/contacts')).body.contacts.map(row => row.id), expected)
    for (const query of [...accounts.map(item => item.username), existing.label]) {
      assert.deepEqual((await call(`/resolve?${new URLSearchParams({ query })}`)).body.candidates, [])
    }
    const resolved = await call(`/resolve?${new URLSearchParams({ query: personal.username })}`)
    assert.deepEqual(resolved.body.candidates.map(row => row.id), [personal.username])
    const legacyMode = async () => (await roster('policy')).body.items.find(row => row.id === legacy.username)?.mode
    assert.equal(await legacyMode(), 'allowlist')
    const saved = await call('/contact-policy', { method: 'POST', headers, body: JSON.stringify({
      recipient_id: legacy.username, mode: 'observe',
    }) })
    assert.equal(saved.status, 200)
    assert.equal(saved.body.recipient.mode, 'observe')
    assert.equal(await legacyMode(), 'observe')
  })
})

test('群目录按明确成员状态过滤并保留群权限记录和上游游标', async () => {
  const current = { roomId: '100@chatroom', displayName: '在群', isInGroup: true }
  const excluded = { roomId: '200@chatroom', displayName: '已离开', isInGroup: false }
  const detailExcluded = { roomId: '300@chatroom', displayName: '详情确认已离开' }
  const unknown = { roomId: '400@chatroom', displayName: '成员状态未知' }
  const mismatched = { roomId: '500@chatroom', displayName: '详情标识不一致' }
  const reads = []
  let mode = 'allowlist'
  const client = {
    contactPolicy: async () => ({
      managed: false, direct: false, group: false, recipients: [], observed: [],
      groupPermissions: [{ thread_id: detailExcluded.roomId, label: detailExcluded.displayName, mode }],
    }),
    listGroups: async () => ({ items: [current, excluded, detailExcluded, unknown, mismatched], nextCursor: 'groups=next+' }),
    getGroup: async id => {
      reads.push(id)
      if (id === detailExcluded.roomId) return { roomId: id, isInGroup: false }
      if (id === unknown.roomId) return { roomId: id, isInGroup: null }
      assert.equal(id, mismatched.roomId)
      return { roomId: excluded.roomId, isInGroup: false }
    },
    saveGroupPermission: async (id, _name, requestedMode) => {
      assert.equal(id, detailExcluded.roomId)
      mode = requestedMode
      return client.contactPolicy()
    },
  }
  await withServer({
    configOverrides: { agentToken: 'runtime-secret' },
    directoryOverrides: { findExactGroup: async () => assert.fail('撤销旧权限无需重新核对群目录') },
    clientOverrides: client,
  }, async ({ call }) => {
    const headers = { 'X-Mahjong-Agent-Token': 'runtime-secret' }
    const roster = view => call(`/permission-roster?${new URLSearchParams({ view })}`, { headers })
    const groups = await roster('groups')
    assert.equal(groups.status, 200)
    assert.deepEqual(groups.body.items.map(row => row.id), [current.roomId, unknown.roomId, mismatched.roomId])
    assert.equal(groups.body.nextCursor, 'groups=next+')
    assert.deepEqual(reads.sort(), [detailExcluded.roomId, unknown.roomId, mismatched.roomId])
    assert.deepEqual((await roster('policy')).body.items.map(row => [row.id, row.mode]), [[detailExcluded.roomId, 'allowlist']])
    const saved = await call('/group-permission', { method: 'POST', headers, body: JSON.stringify({
      recipient_id: detailExcluded.roomId, mode: 'observe',
    }) })
    assert.equal(saved.status, 200)
    assert.equal(saved.body.item.mode, 'observe')
  })
})

test('权限页刷新允许名单内仅托管和观察，拒绝未登记联系人且不赋予发送资格', async () => {
  let invalid = false
  const refreshed = []
  await withServer({
    configOverrides: { agentToken: 'runtime-secret' },
    sendChannelEnabled: false,
    autoSendReply: false,
    clientOverrides: {
      contactPolicy: async () => ({
        direct: true, invalid, recipients: [],
        observed: [{ thread_id: 'wxid_wang', label: '王' }],
        managedOnly: [{ thread_id: 'wxid_li', label: '李' }],
        groupPermissions: [],
      }),
      sendText: async () => assert.fail('刷新不能外发'),
    },
    directoryOverrides: {
      refreshContact: async id => {
        refreshed.push(id)
        return { contact: { username: id, remark: '更新的备注' }, projection: 'partial' }
      },
    },
  }, async ({ call, switches }) => {
    const headers = { 'X-Mahjong-Agent-Token': 'runtime-secret' }
    const refresh = (id, auth = headers, path = '/contact-policy/refresh') => call(path, {
      method: 'POST', headers: auth, body: JSON.stringify({ recipient_id: id }),
    })
    assert.equal((await refresh('wxid_li', {})).status, 401)
    assert.equal((await refresh('100@chatroom')).status, 400)
    assert.equal((await refresh('wxid_unlisted')).status, 404)
    assert.equal((await refresh('wxid_li', headers, '/contacts/refresh')).status, 403)
    for (const id of ['wxid_li', 'wxid_wang']) {
      const result = await refresh(id)
      assert.equal(result.status, 200)
      assert.equal(result.body.recipient.remark, '更新的备注')
      assert.equal(result.body.projection, 'partial')
    }
    invalid = true
    assert.equal((await refresh('wxid_li')).status, 409)
    assert.deepEqual(refreshed, ['wxid_li', 'wxid_wang'])
    assert.equal(switches.sendChannelEnabled, false)
    assert.equal(switches.autoSendReply, false)
  })
})

test('群覆盖可热更新且不启用私聊管理，观察拒绝所有发送，继承恢复原规则', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'wechat-group-policy-'))
  const path = join(dir, 'contact_permissions.local.json')
  const target = { roomId: '100@chatroom', displayName: '候选群' }
  const groups = [target, { roomId: '200@chatroom', displayName: '普通群' }]
  let nativeReads = 0
  const sent = []
  const sdk = new AgentWechatClient({
    baseUrl: 'http://stub', accountId: 'acc', managedContactPolicyPath: path,
    fetchImpl: async (url, init) => {
      if (url.includes('/contacts')) return Response.json({ items: [{ username: 'wxid_li', remark: '李姐' }] })
      if (url.includes('/groups')) {
        nativeReads += 1
        return Response.json({ items: groups })
      }
      if (url.endsWith('/sends')) {
        sent.push(JSON.parse(init.body).threadId)
        return Response.json({ id: 'send-1' })
      }
      assert.fail(`unexpected native request ${url}`)
    },
  })
  try {
    await withServer({
      directory: new Directory(sdk), autoSendReply: false,
      configOverrides: { agentToken: 'runtime-secret' },
      clientOverrides: {
        contactPolicy: sdk.contactPolicy.bind(sdk),
        saveGroupPermission: sdk.saveGroupPermission.bind(sdk),
        sendText: sdk.sendText.bind(sdk),
      },
    }, async ({ call, switches }) => {
      const headers = { 'X-Mahjong-Agent-Token': 'runtime-secret' }
      const save = (recipient_id, mode, automation = {}) => call('/group-permission', {
        method: 'POST', headers, body: JSON.stringify({ recipient_id, mode, ...automation }),
      })
      const send = (to, source) => call('/send', { method: 'POST', body: JSON.stringify({
        to, target_type: 'group', exact_target: true, source, text: '测试用例',
      }) })
      assert.equal((await send(target.roomId)).status, 409)
      assert.equal((await save('wxid_li', 'allowlist')).status, 400)
      assert.equal((await save('301@chatroom', 'allowlist')).status, 404)
      const allowed = await save(target.roomId, 'allowlist')
      assert.equal(allowed.status, 200)
      assert.equal(allowed.body.account_id, 'acc')
      assert.equal(allowed.body.item.override, 'allowlist')
      const policy = await sdk.contactPolicy()
      assert.equal(policy.direct, false)
      assert.equal(policy.document.managed_direct, undefined)
      if (process.platform !== 'win32') {
        assert.equal((await stat(path)).mode & 0o777, 0o600)
      }
      assert.equal((await send(target.roomId, 'automatic')).status, 409)
      assert.equal((await send('200@chatroom')).status, 409)
      assert.equal((await send(target.roomId)).status, 200)
      assert.equal((await save(target.roomId, 'observe', { managed: true })).status, 400)
      const automatic = await save(target.roomId, 'inherit', {
        managed: true, auto_reply: true, auto_reply_at: '1900-01-01T00:00:00.000Z',
      })
      assert.deepEqual(automatic.body.item.automation, { managed: true, auto_reply: true })
      assert.equal(automatic.body.item.override, null)
      const [rule] = (await sdk.contactPolicy()).groupPermissions
      assert.ok(Date.parse(rule.managed_at) > 0)
      assert.ok(Date.parse(rule.auto_reply_at) > 0)
      assert.equal((await save(target.roomId, 'allowlist')).body.item.automation.auto_reply, true)
      assert.equal((await sdk.contactPolicy()).groupPermissions[0].auto_reply_at, rule.auto_reply_at)
      groups.shift()
      assert.equal((await send(target.roomId)).status, 409)

      const beforeRevoke = nativeReads
      const observed = await save(target.roomId, 'observe')
      assert.equal(observed.body.item.override, 'observe')
      assert.equal(nativeReads, beforeRevoke)
      assert.equal((await save(target.roomId, 'allowlist')).status, 404)
      switches.autoSendReply = true
      assert.equal((await send(target.roomId)).status, 403)
      assert.equal((await send(target.roomId, 'automatic')).status, 403)
      await assert.rejects(sdk.sendText({ threadId: target.roomId, text: '测试用例', idempotencyKey: 'observe' }),
        { code: 'outbound_recipient_denied' })
      assert.equal((await call(`/resolve?${new URLSearchParams({ query: target.roomId })}`)).body.candidates.length, 0)
      const inherited = await save(target.roomId, 'inherit', { managed: false, auto_reply: false })
      assert.equal(inherited.body.item.override, null)
      assert.equal(inherited.body.item.mode, 'unassigned')
      assert.deepEqual(inherited.body.item.automation, { managed: false, auto_reply: false })
      const [overlaid] = (await sdk.contactPolicy()).groupPermissions
      assert.equal(overlaid.mode, 'inherit')
      assert.equal(overlaid.auto_reply_at, undefined)
      await assert.rejects(sdk.sendText({ threadId: target.roomId, text: '测试用例', idempotencyKey: 'race',
        manual: true, requireManualScope: true }), { code: 'auto_send_required' })
      assert.deepEqual(sent, [target.roomId])
    })
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('普通后台群回复不能绕过授权，旧托管群与看板仍走各自的授权', async () => {
  const group = '100@chatroom'
  const sent = []
  let legacy = false
  await withServer({
    clientOverrides: {
      contactPolicy: async () => ({ direct: false, group: legacy,
        recipients: legacy ? [{ thread_id: group, label: '旧群' }] : [], observed: [], groupPermissions: [] }),
      sendText: async request => {
        sent.push(request.threadId)
        return { id: 'fixture-send' }
      },
    },
  }, async ({ call }) => {
    const send = (source, metadata = {}) => call('/send', { method: 'POST', body: JSON.stringify({
      to: group, target_type: 'group', exact_target: true, text: '测试', source, ...metadata,
    }) })
    assert.equal((await send('delayed_input_batch_reply')).status, 403)
    assert.equal((await send('recovered_agent_run_reply')).status, 403)
    assert.equal((await send('managed_group_domain', { source_message_id: 'native-message' })).status, 403)
    assert.deepEqual(sent, [])
    legacy = true
    assert.equal((await send('delayed_input_batch_reply')).status, 403)
    assert.equal((await send('managed_group_domain', { source_message_id: 'native-message' })).status, 200)
    assert.equal((await send('managed_group_domain', { message_type: 'game_board', source_message_id: 'board-source' })).status, 200)
    assert.equal((await send(undefined)).status, 200)
    assert.deepEqual(sent, [group, group, group])
  })
})

test('既有白名单内的群被观察覆盖后停止发送，新增群覆盖仅允许人工发送', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'wechat-group-legacy-policy-'))
  const path = join(dir, 'existing.json')
  const oldGroup = { thread_id: '100@chatroom', label: '原有群', aliases: ['旧群'] }
  const direct = { thread_id: 'wxid_li', label: '李姐' }
  const sent = []
  const sdk = new AgentWechatClient({
    baseUrl: 'http://stub', accountId: 'acc', outboundAllowlistPath: path,
    fetchImpl: async (_url, init) => {
      sent.push(JSON.parse(init.body).threadId)
      return Response.json({ id: 'send-1' })
    },
  })
  const send = (threadId, manual) => sdk.sendText({ threadId, manual, text: '测试用例', idempotencyKey: 'test' })
  try {
    await writeFile(path, JSON.stringify({ account_id: 'acc', recipients: [oldGroup, direct], existing: true }))
    await sdk.saveGroupPermission(oldGroup.thread_id, '测试', 'observe')
    await sdk.saveGroupPermission('200@chatroom', '新群', 'allowlist')
    await assert.rejects(send(oldGroup.thread_id, true), { code: 'outbound_recipient_denied' })
    await assert.rejects(send(oldGroup.thread_id, false), { code: 'outbound_recipient_denied' })
    await assert.rejects(send('200@chatroom', false), { code: 'outbound_recipient_denied' })
    await send('200@chatroom', true)
    const inherited = await sdk.saveGroupPermission(oldGroup.thread_id, '测试', 'inherit')
    assert.deepEqual(inherited.recipients, [oldGroup, direct])
    assert.equal(inherited.document.existing, true)
    assert.equal(inherited.document.managed_direct, undefined)
    await send(oldGroup.thread_id, false)
    assert.deepEqual(sent, ['200@chatroom', oldGroup.thread_id])
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('已选目标只按当前精确私聊标识或正确类型的群 ID 发送', async () => {
  const current = { username: 'wxid_current', nickname: '李姐', remark: 'wxid_stale' }
  const lookups = []
  const directory = new Directory({
    listContacts: async ({ search }) => {
      lookups.push(search)
      return { items: [current] }
    },
  })
  directory.contacts = [{ username: 'wxid_stale' }]
  const sends = []
  await withServer({ directory, clientOverrides: {
    sendText: async request => {
      sends.push(request.threadId)
      return { id: 'send-1' }
    },
  } }, async ({ call }) => {
    const send = (to, target_type, exact_target = true) => call('/send', { method: 'POST', body: JSON.stringify({
      to, target_type, exact_target, text: '测试用例',
    }) })
    assert.equal((await send('李姐', 'direct')).status, 409)
    assert.equal((await send('wxid_stale', 'direct')).status, 409)
    assert.equal((await send('200@chatroom', 'direct')).status, 400)
    assert.equal((await send('wxid_current', 'group')).status, 400)
    assert.equal((await send('wxid_current', '', true)).status, 400)
    assert.equal((await send('李姐', 'direct', 'true')).status, 400)
    assert.deepEqual(sends, [])
    const selected = await send('wxid_current', 'direct')
    assert.equal(selected.status, 200)
    assert.equal(selected.body.local_message_id, 'wx-out-1')
    assert.equal((await send('200@chatroom', 'group')).status, 200)
    assert.deepEqual(lookups, ['李姐', 'wxid_stale', 'wxid_current'])
    assert.deepEqual(sends, ['wxid_current', '200@chatroom'])
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
      assert.equal(body.local_message_id, 'wx-out-1')
      assert.equal(body.state, 'accepted')
      assert.equal(calls.length, 1)
      assert.equal(calls[0].threadId, 'wxid_li')
      assert.equal(calls[0].idempotencyKey, 'mahjong-send:trace-9:wxid_li')
    },
  )
})

test('同一幂等键已 accepted 时直接去重', async () => {
  let sent = 0
  const createdAt = '2026-01-02T11:00:00Z'
  await withServer(
    {
      clientOverrides: {
        getSendByKey: async () => ({ id: 'send-old', state: 'accepted', localMessageId: 'wx-old', createdAt }),
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
      assert.equal(body.local_message_id, 'wx-old')
      assert.equal(body.message_created_at, createdAt)
      assert.equal(sent, 0)
    },
  )
})

test('上游成功回执只有服务端 ID 时不把它当成本地消息 ID', async () => {
  const createdAt = '2026-01-02T11:00:00Z'
  await withServer({ clientOverrides: {
    waitForSend: async () => ({ id: 'send-1', state: 'accepted', serverId: '123', createdAt }),
  } }, async ({ call }) => {
    const { status, body } = await call('/send', { method: 'POST', body: JSON.stringify({
      to: '200@chatroom', target_type: 'group', exact_target: true, text: '测试用例',
    }) })
    assert.equal(status, 200)
    assert.equal(body.message_id, '123')
    assert.equal(body.server_message_id, '123')
    assert.equal(body.local_message_id, undefined)
    assert.equal(body.message_created_at, createdAt)
  })
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
