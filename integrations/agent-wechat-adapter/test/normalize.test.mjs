import test from 'node:test'
import assert from 'node:assert/strict'
import {
  buildChannelPayload,
  conversationIdForThread,
  replyIdempotencyKey,
  sendIdempotencyKey,
  shouldForwardMessage,
  toWechatyType,
} from '../src/normalize.mjs'

test('原生类型映射到 Wechaty 枚举', () => {
  assert.equal(toWechatyType({ type: 1 }), 7)
  assert.equal(toWechatyType({ type: 3 }), 6)
  assert.equal(toWechatyType({ type: 10000 }), 0)
  assert.equal(toWechatyType({ type: 49, appMessageType: 57 }), 7)
  assert.equal(toWechatyType({ type: 49, appMessageType: 6 }), 1)
  assert.equal(toWechatyType({ type: 999 }), null)
})

test('会话 id 复用 wechaty 命名空间', () => {
  assert.equal(conversationIdForThread('wxid_abc'), 'wechaty:contact:wxid_abc')
  assert.equal(conversationIdForThread('123@chatroom'), 'wechaty:room:123@chatroom')
})

test('默认过滤自消息与系统消息', () => {
  const base = { id: 'm1', threadId: 'wxid_a', type: 1, direction: 'incoming' }
  assert.equal(shouldForwardMessage(base), true)
  assert.equal(shouldForwardMessage({ ...base, direction: 'outgoing' }), false)
  assert.equal(shouldForwardMessage({ ...base, type: 10000 }), false)
  assert.equal(shouldForwardMessage({ ...base, direction: 'outgoing' }, { forwardSelfMessages: true }), true)
  assert.equal(shouldForwardMessage({ ...base, id: '' }), false)
})

test('群消息载荷保留 topic 与发送者 id', () => {
  const payload = buildChannelPayload({
    message: {
      id: 'm2',
      threadId: '123@chatroom',
      senderId: 'wxid_sender',
      body: '晚上三缺一',
      type: 1,
      direction: 'incoming',
      appMessageType: null,
    },
    sender: { username: 'wxid_sender', remark: '老王', nickname: '王哥', displayName: '老王', alias: 'owner-lookup-id' },
    thread: { displayName: '周末局' },
    accountId: 'acc-1',
  })
  assert.equal(payload.channel, 'agent-wechat')
  assert.equal(payload.conversation_id, 'wechaty:room:123@chatroom')
  assert.equal(payload.is_room, true)
  assert.equal(payload.room.topic, '周末局')
  assert.equal(payload.sender_name, '王哥')
  assert.equal(payload.talker.name, '王哥')
  assert.equal(payload.talker.alias, '')
  assert.deepEqual(payload.talker.payload, { username: 'wxid_sender', remark: null, nickname: '王哥', displayName: null })
  assert.equal(payload.message_type, 7)
  assert.equal(payload.payload.native_type, 1)
  assert.equal(payload.raw_observation.thread_id, '123@chatroom')
})

test('群名片缺少可信个人昵称时用群成员，私聊仍保留自己的备注', () => {
  const sender = { username: 'wxid_sender', remark: '内部备注', displayName: '内部展示名' }
  const group = buildChannelPayload({ message: { id: 'm1', threadId: '123@chatroom', senderId: sender.username }, sender })
  assert.equal(group.sender_name, '群成员')
  assert.equal(group.talker.payload.nickname, null)
  assert.equal(group.talker.payload.remark, null)
  const mismatch = buildChannelPayload({ message: { id: 'm2', threadId: '123@chatroom', senderId: 'wxid_other' },
    sender: { ...sender, nickname: '不能借用其他人的昵称' } })
  assert.equal(mismatch.sender_name, '群成员')
  assert.equal(mismatch.talker.payload.username, 'wxid_other')
  const direct = buildChannelPayload({ message: { id: 'm3', threadId: sender.username, senderId: sender.username }, sender })
  assert.equal(direct.sender_name, '内部备注')
  assert.equal(direct.talker.payload.remark, '内部备注')
})

test('外发幂等键有锚点时稳定、无锚点时唯一', () => {
  const first = sendIdempotencyKey({ sourceTraceId: 'trace-1', target: 'wxid_a' })
  const second = sendIdempotencyKey({ sourceTraceId: 'trace-1', target: 'wxid_a' })
  const other = sendIdempotencyKey({ sourceTraceId: 'trace-2', target: 'wxid_a' })
  assert.equal(first, second)
  assert.notEqual(first, other)
  assert.notEqual(
    sendIdempotencyKey({ target: 'wxid_a' }),
    sendIdempotencyKey({ target: 'wxid_a' }),
  )
})

test('自动回复幂等键不超过上游 200 字符', () => {
  const key = replyIdempotencyKey('x'.repeat(400))
  assert.ok(key.length <= 200)
})
