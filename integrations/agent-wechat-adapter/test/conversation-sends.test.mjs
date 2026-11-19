import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import { deriveConversationAuthors, recordConversationSend } from '../src/conversation-sends.mjs'

const account = 'example-account'
const thread = 'wxid_example'

/** 收据测试使用独立本机账本，避免读写真实聊天。 */
function tempLedger(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'conversation-sends-'))
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  return path.join(dir, 'delivery_ledger.jsonl')
}

/** 按原生消息格式构造可核对的已保存外发。 */
function outgoing(id, extras = {}) {
  return { accountId: account, threadId: thread, direction: 'outgoing', id, ...extras }
}

/** 构造终态收据并允许覆盖字段，验证身份不一致时拒绝署名。 */
function accepted(id, localMessageId, extras = {}) {
  return { accountId: account, destination: thread, state: 'accepted', id, localMessageId, ...extras }
}

test('控制面回执只记录被平台接受的精确目标，重复读取不重复写入', async t => {
  const ledgerPath = tempLedger(t)
  const client = { accountId: account }
  const input = { client, threadId: thread, author: 'owner' }
  assert.equal(recordConversationSend(ledgerPath, {
    ...input, state: accepted('send-draft', 'not-delivered', { state: 'needs_review' }),
  }), false)
  assert.equal(recordConversationSend(ledgerPath, {
    ...input, state: accepted('send-other', 'other-message', { destination: 'wxid_other' }),
  }), false)
  assert.equal(fs.existsSync(ledgerPath), false)

  const state = accepted('send-owner', 'local-owner', { serverId: 'server-owner' })
  assert.equal(recordConversationSend(ledgerPath, { ...input, state }), true)
  assert.equal(recordConversationSend(ledgerPath, { ...input, state }), true)
  const rows = fs.readFileSync(ledgerPath, 'utf8').trim().split('\n').map(JSON.parse)
  assert.equal(rows.length, 1)
  assert.equal(Object.hasOwn(rows[0], 'source_message_id'), false)
  assert.deepEqual(await deriveConversationAuthors({
    client, ledgerPath, threadId: thread, messages: [outgoing('local-owner', { serverId: 'server-owner' })],
  }), new Map([['local-owner', 'owner']]))
})

test('只按账本和原生消息的同一账号会话投影发送者，冲突或草稿保持未知', async t => {
  const ledgerPath = tempLedger(t)
  const record = { account_id: account, thread_id: thread, author: 'agent', state: 'accepted' }
  const rows = [
    { ...record, local_message_id: 'local-agent', server_message_id: 'server-agent' },
    { ...record, local_message_id: 'unlisted-old-id', server_message_id: 'known-server' },
    { ...record, account_id: 'different-account', local_message_id: 'wrong-account' },
    { ...record, thread_id: 'wxid_different', local_message_id: 'wrong-thread' },
    { ...record, local_message_id: 'wrong-server', server_message_id: 'different-server' },
    { ...record, local_message_id: 'incoming' },
    { ...record, local_message_id: 'conflicting-authors' },
    { ...record, local_message_id: 'conflicting-authors', author: 'owner' },
    { ...record, local_message_id: 'draft', state: 'failed' },
  ]
  fs.writeFileSync(ledgerPath, `${rows.map((row, index) => JSON.stringify({ send_id: `send-${index}`, ...row })).join('\n')}\n{`)
  const messages = [
    outgoing('local-agent', { serverId: 'server-agent' }),
    outgoing('now-synced-id', { serverId: 'known-server' }),
    outgoing('wrong-account'), outgoing('wrong-thread'),
    outgoing('wrong-server', { serverId: 'native-server' }),
    outgoing('incoming', { direction: 'incoming' }),
    outgoing('conflicting-authors'), outgoing('draft'), outgoing('native-unknown'),
  ]
  const authors = await deriveConversationAuthors({ client: { accountId: account }, ledgerPath, threadId: thread, messages })
  assert.deepEqual(authors, new Map([['local-agent', 'agent'], ['now-synced-id', 'agent']]))
})

test('旧自动回复账本必须复核当前账号的原生接受回执才能确认归属', async t => {
  const ledgerPath = tempLedger(t)
  const legacy = { conversation_id: `wechaty:contact:${thread}`, source_message_id: 'incoming-example', state: 'accepted' }
  const rows = ['confirmed', 'wrong-owner', 'wrong-thread', 'unknown', 'wrong-receipt'].map(id => ({
    ...legacy, send_id: `send-${id}`, message_id: id,
  }))
  rows.push({ ...legacy, conversation_id: 'wechaty:contact:wxid_other', send_id: 'foreign', message_id: 'unknown' })
  fs.writeFileSync(ledgerPath, rows.map(row => JSON.stringify(row)).join('\n'))
  const asked = []
  const receipts = {
    'send-confirmed': accepted('send-confirmed', 'confirmed'),
    'send-wrong-owner': accepted('send-wrong-owner', 'wrong-owner', { accountId: 'previous-account' }),
    'send-wrong-thread': accepted('send-wrong-thread', 'wrong-thread', { destination: 'wxid_other' }),
    'send-unknown': accepted('send-unknown', 'unknown', { state: 'needs_review' }),
    'send-wrong-receipt': accepted('returned-another-send', 'wrong-receipt'),
  }
  const client = { accountId: account, getSend: async id => { asked.push(id); return receipts[id] } }
  const authors = await deriveConversationAuthors({ client, ledgerPath, threadId: thread,
    messages: rows.map(row => outgoing(row.message_id)) })
  assert.deepEqual(authors, new Map([['confirmed', 'agent']]))
  assert.equal(asked.length, 5)
  assert.equal(asked.includes('foreign'), false)
})
