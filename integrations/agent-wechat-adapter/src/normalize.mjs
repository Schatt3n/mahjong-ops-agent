/**
 * agent-wechat 原生消息 -> Mahjong Runtime raw 通道合同。
 *
 * Runtime 的 wechaty 入口读取的是 Wechaty 的 `message_type` 枚举，而 agent-wechat
 * 暴露的是微信原生 type。两者不同，必须在这里显式映射，不能让下游猜。
 * 会话 id 仍使用 `wechaty:contact:` / `wechaty:room:` 命名空间，因为两条通道操作
 * 同一个微信号和同一批 wxid，历史看板与群聚合不能因为换通道而断开。
 */
import { createHash, randomUUID } from 'node:crypto'

/** 微信原生 type -> Wechaty 枚举。 */
export const WECHATY_TYPE_BY_NATIVE = new Map([
  [1, 7], // 文本
  [3, 6], // 图片
  [34, 2], // 语音
  [43, 15], // 视频
  [47, 5], // 表情
  [48, 8], // 位置
  [49, 14], // 应用消息：链接、文件、引用等
  [10000, 0], // 系统提示
])

/** type 49 的 appMessageType -> Wechaty 枚举。 */
export const WECHATY_TYPE_BY_APP_MESSAGE = new Map([
  [5, 14], // 链接
  [6, 1], // 文件
  [8, 5], // 收藏表情
  [19, 14], // 合并转发
  [57, 7], // 引用回复，正文仍是文本
])

export const SYSTEM_MESSAGE_TYPE = 10000

/** 区分其他设备同步的本人消息与 daemon 本地发送回声。 */
export function isOwnerMessageEvent(event, threadId) {
  const data = event?.data
  return event?.kind === 'message.committed' && data?.direction === 'outgoing' &&
    (data.origin === 'live' || data.origin === 'catchup') && data.threadId === threadId
}

/** 群会话 id 以 @chatroom 结尾，其余按私聊处理。 */
export function isChatroomId(threadId) {
  return String(threadId || '').trim().endsWith('@chatroom')
}

/** 把原生消息类型折算成 Runtime 认识的 Wechaty 枚举，未知类型返回 null。 */
export function toWechatyType(message) {
  const nativeType = Number(message?.type)
  if (nativeType === 49) {
    const mapped = WECHATY_TYPE_BY_APP_MESSAGE.get(Number(message?.appMessageType))
    if (mapped !== undefined) return mapped
  }
  const mapped = WECHATY_TYPE_BY_NATIVE.get(nativeType)
  return mapped === undefined ? null : mapped
}

/** 构造与 wechaty 桥接一致的会话 id，保证历史聚合连续。 */
export function conversationIdForThread(threadId) {
  const id = String(threadId || '').trim()
  return isChatroomId(id) ? `wechaty:room:${id}` : `wechaty:contact:${id}`
}

/** 按备注 > 昵称 > 显示名 > id 的顺序挑选展示名，仅用于记录，不作为寻址键。 */
export function displayNameOf(contact, fallback = '') {
  if (!contact) return fallback
  return (
    String(contact.remark || '').trim() ||
    String(contact.nickname || '').trim() ||
    String(contact.displayName || '').trim() ||
    String(contact.username || contact.id || '').trim() ||
    fallback
  )
}

/**
 * 判断一条消息是否应该交给业务系统。
 *
 * 通过事件确认的本人跨设备消息由适配器交给独立上下文入口；这里保留诊断开关。
 * 媒体消息即使没有正文也保留，由 Runtime 判断是否需要 OCR/转写。
 */
export function shouldForwardMessage(message, { forwardSelfMessages = false, forwardSystemMessages = false } = {}) {
  if (!message || !message.id || !message.threadId) return false
  if (Number(message.type) === SYSTEM_MESSAGE_TYPE && !forwardSystemMessages) return false
  if (String(message.direction || '') === 'outgoing' && !forwardSelfMessages) return false
  return true
}

/**
 * 构建 raw 入站载荷。
 *
 * `sender` 与 `thread` 只是本次采集到的展示信息；寻址仍使用 senderId / threadId。
 * 群聊的主题来自 thread.displayName，缺失时不编造。
 */
export function buildChannelPayload({ message, sender = null, thread = null, accountId = '' }) {
  const threadId = String(message?.threadId || '').trim()
  const isRoom = isChatroomId(threadId)
  const senderId = String(message?.senderId || '').trim()
  // 群消息可进入公开业务上下文，店主的私有备注只在工作台显示。
  const exactSender = senderId && (sender?.username || sender?.id) === senderId
  const nickname = exactSender && typeof sender?.nickname === 'string' ? sender.nickname.trim() : ''
  const senderName = isRoom ? nickname || '群成员' : displayNameOf(sender, senderId)
  const body = typeof message?.body === 'string' ? message.body : ''
  const wechatyType = toWechatyType(message)

  const room = isRoom
    ? {
        id: threadId,
        topic: String(thread?.displayName || '').trim(),
        payload: { kind: 'group', id: threadId, displayName: thread?.displayName ?? null },
      }
    : null

  const talker = {
    id: senderId,
    name: senderName,
    alias: isRoom ? '' : String(sender?.alias || '').trim(),
    payload: {
      username: isRoom ? senderId : String(sender?.username || senderId),
      remark: isRoom ? null : sender?.remark ?? null,
      nickname: isRoom ? nickname || null : sender?.nickname ?? null,
      displayName: isRoom ? null : sender?.displayName ?? null,
    },
  }

  return {
    captured_at: new Date().toISOString(),
    channel: 'agent-wechat',
    platform_name: 'agent-wechat',
    conversation_id: conversationIdForThread(threadId),
    message_id: String(message.id),
    source_message_id: String(message.id),
    message_type: wechatyType === null ? '' : wechatyType,
    is_room: isRoom,
    room,
    sender_id: senderId,
    sender_name: senderName,
    talker,
    listener: { id: accountId, name: '', payload: { account_id: accountId } },
    text: body,
    raw_text: body,
    self_message: String(message.direction || '') === 'outgoing',
    payload: {
      ...message,
      native_type: message.type,
      wechaty_type: wechatyType,
      account_id: accountId,
    },
    raw_observation: {
      source: 'agent-wechat',
      message_id: String(message.id),
      thread_id: threadId,
      native_type: message.type,
      app_message_type: message.appMessageType ?? null,
      wechaty_type: wechatyType,
      direction: message.direction ?? null,
      has_body: Boolean(body.trim()),
      server_id: message.serverId ?? null,
      quote_message_id: message.quote?.messageId ?? message.quote?.id ?? null,
      mention_count: Array.isArray(message.mentions?.members) ? message.mentions.members.length : 0,
      media_present: Boolean(message.media?.length || message.image || message.file),
      text_empty: !body.trim(),
    },
  }
}

/** 生成稳定外发幂等键；超长时截断并附哈希后缀，保证不超过上游 200 字符限制。 */
export function replyIdempotencyKey(sourceMessageId) {
  const base = `mahjong-reply:${String(sourceMessageId || '').trim()}`
  if (base.length <= 200) return base
  let hash = 0
  for (const char of base) hash = (hash * 31 + char.codePointAt(0)) >>> 0
  return `${base.slice(0, 180)}:${hash.toString(16)}`
}

/**
 * 生成手动外发幂等键。
 *
 * 有业务锚点（source_trace_id / source_message_id）时跨重试稳定，上游据此返回同一
 * 发送意图；没有锚点时使用一次性键，避免把“相同文本的两次正常发送”误去重。
 */
export function sendIdempotencyKey({ sourceTraceId = '', target = '' } = {}) {
  const anchor = String(sourceTraceId || '').trim()
  if (!anchor) return `mahjong-send:${randomUUID()}`
  const raw = `mahjong-send:${anchor}:${String(target || '').trim()}`
  if (raw.length <= 200) return raw
  const digest = createHash('sha256').update(raw).digest('hex').slice(0, 32)
  return `mahjong-send:${digest}`
}
