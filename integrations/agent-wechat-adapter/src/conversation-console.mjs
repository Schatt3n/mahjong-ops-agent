/** 店主会话读面只投影本机精确白名单里的真实微信记录，不改变账号读状态。 */
import { isDirectContactId, isGroupPermissionId, permitsRecipientSend } from './contact-policy.mjs'
import { deriveConversationAuthors } from './conversation-sends.mjs'
import { displayNameOf, SYSTEM_MESSAGE_TYPE } from './normalize.mjs'

const DIRECTORY_BUDGET_MS = 4000
const HISTORY_BUDGET_MS = 5000
const MEMBER_BUDGET_MS = 2400
const ACTIVE_WINDOW_MS = 2 * 60 * 60_000
const HISTORY_MEDIA_LABELS = new Map([[3, '图片'], [34, '语音'], [43, '视频'], [47, '表情'], [48, '位置'], [49, '分享消息']])

/** 只从显式出站资格派生可浏览范围；未限定的旧手动通道不构成聊天记录授权。 */
function conversationCandidates(policy) {
  const items = new Map()
  if (policy.invalid) return items
  for (const entry of policy.recipients) {
    const id = entry.thread_id
    if (policy.direct && isDirectContactId(id)) items.set(id, { id, kind: 'direct', displayName: entry.label })
    if (policy.group && isGroupPermissionId(id) && permitsRecipientSend(policy, id, true)) {
      items.set(id, { id, kind: 'group', displayName: entry.label })
    }
  }
  for (const entry of policy.groupPermissions) {
    if (entry.mode === 'allowlist' && isGroupPermissionId(entry.thread_id)) {
      const id = entry.thread_id
      items.set(id, { id, kind: 'group', displayName: entry.label })
    }
  }
  return items
}

/** 时间必须带时区且属于已经发生的消息，避免缺省时间或未来时间误标会话活动。 */
function savedTime(value, now = Date.now()) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/u.test(value)) return null
  const milliseconds = Date.parse(value)
  return Number.isFinite(milliseconds) && milliseconds <= now ? new Date(milliseconds).toISOString() : null
}

/** 方向未知的系统通知不能作为客户活动或店主消息。 */
function savedDirection(row) {
  if (Number(row?.type) === SYSTEM_MESSAGE_TYPE) return null
  return row?.direction === 'incoming' ? 'inbound' : row?.direction === 'outgoing' ? 'outbound' : null
}

/** 原生历史的账号、会话和格式都必须匹配，不能借用其他会话的游标读取内容。 */
function isThreadMessage(row, accountId, threadId) {
  return row?.accountId === accountId && row.threadId === threadId && typeof row.id === 'string' &&
    row.id.length > 0 && row.id.length <= 512 && !/\p{Cc}/u.test(row.id)
}

/** 批量读取只扫当前授权 ID；即使详情暂时不可读，也保留候选供任务归并。 */
export async function consoleConversations(client) {
  const accountId = client.accountId
  const policy = await client.contactPolicy()
  const items = [...conversationCandidates(policy).values()].map(item => ({ ...item, last_at: null }))
  if (!accountId || !items.length) {
    return { ok: true, account_id: accountId, items: [], partial: Boolean(!accountId || policy.invalid) }
  }
  const signal = AbortSignal.timeout(DIRECTORY_BUDGET_MS)
  const now = Date.now()
  const checked = new Set()
  let nextIndex = 0
  await Promise.all(Array.from({ length: Math.min(items.length, 4) }, async () => {
    while (nextIndex < items.length && !signal.aborted) {
      const item = items[nextIndex++]
      try {
        const thread = await client.getThread(item.id, { signal })
        if (thread?.accountId !== accountId || thread.id !== item.id || thread.kind !== item.kind) continue
        if (typeof thread.displayName === 'string' && thread.displayName.trim() && thread.displayName !== item.id) {
          item.displayName = thread.displayName.trim()
        }
        const latest = thread.lastMessage
        const lastAt = savedTime(latest?.createdAt, now)
        if (savedDirection(latest)) {
          item.last_at = lastAt
        } else if (lastAt && Date.parse(lastAt) >= now - ACTIVE_WINDOW_MS) {
          // 最新记录可能是系统通知，只为近期会话补查有限的一页真实消息。
          const page = await client.listThreadMessages(item.id, { limit: 20, signal })
          if (!Array.isArray(page?.items)) continue
          item.last_at = page.items.filter(row => isThreadMessage(row, accountId, item.id) && savedDirection(row))
            .map(row => savedTime(row.createdAt, now)).filter(Boolean).sort().at(-1) || null
          if (!item.last_at && page.nextCursor) continue
        }
        checked.add(item.id)
      } catch (error) {
        // 目录尚无此会话属于可确认的空结果；超时和快照错误保留为部分读取。
        if (error.status === 404) checked.add(item.id)
      }
    }
  }))
  const currentPolicy = await client.contactPolicy()
  const current = conversationCandidates(currentPolicy)
  if (client.accountId !== accountId) return { ok: true, account_id: client.accountId, items: [], partial: true }
  const visible = items.filter(item => current.get(item.id)?.kind === item.kind)
  return { ok: true, account_id: accountId, items: visible,
    partial: Boolean(currentPolicy.invalid || visible.some(item => !checked.has(item.id))) }
}

/** 仅输出可展示的正文或附件类型；保留撤回提示，省去媒体和其它原生内部字段。 */
function savedText(row) {
  if (row.recall) return '消息已撤回'
  const type = Number(row.type)
  const quote = type === 49 && Number(row.appMessageType) === 57
  const text = typeof row.body === 'string' ? row.body : ''
  // 引用回复在原生已投影为可读文字时保留；微信消息封装的 XML、媒体与卡片载荷不传给页面。
  const rawEnvelope = /^\s*<(?:\?xml\b|!DOCTYPE\b|msg(?:\s|[>/])|appmsg(?:\s|[>/])|refermsg(?:\s|[>/]))/iu.test(text)
  if ((type === 1 || quote) && text.trim() && !rawEnvelope) {
    // 截断不能把表情的代理对拆成无效的 UTF-8。
    const visible = text.length <= 10_000 ? text : text.slice(0, 10_000).replace(/[\ud800-\udbff]$/u, '') + '…'
    return visible.toWellFormed()
  }
  return `[${quote ? '引用回复' : HISTORY_MEDIA_LABELS.get(type) || '消息'}]`
}

/** 只接受和消息发件人完全相同的联系人记录，显示缓存也不能用昵称或别名猜测身份。 */
function exactContact(row, id, accountId) {
  return row && row.deleted !== true && exactProfileId(row, id) &&
    (row.accountId === undefined || row.accountId === accountId) ? row : null
}

/** 原生资料可包含两种字段表示同一个 ID；同时存在时不能彼此冲突。 */
function exactProfileId(row, id) {
  return Boolean(row) && (row.username === id || row.id === id) &&
    (row.username === undefined || row.username === id) && (row.id === undefined || row.id === id)
}

/** 名字只作工作台显示；去掉空白并限制可展示长度。 */
function speakerName(value) {
  return typeof value === 'string' ? value.trim().slice(0, 256).toWellFormed() : ''
}

/** 同页相同发件人只查一次。当前已存关系是新增私聊入口的唯一好友证据。 */
async function groupSpeakerProfiles({ client, directory, accountId, groupId, rows }) {
  const ids = [...new Set(rows.filter(item => item.direction === 'inbound').map(item => item.row.senderId)
    .filter(isDirectContactId))]
  const profiles = new Map()
  for (const id of ids) {
    const cached = exactContact(directory.contact?.(id), id, accountId)
    profiles.set(id, { speaker: speakerName(cached?.remark) || speakerName(cached?.nickname), friend: null })
  }
  if (!ids.length) return profiles
  const signal = AbortSignal.timeout(MEMBER_BUDGET_MS)
  let nextIndex = 0
  await Promise.all(Array.from({ length: Math.min(ids.length, 4) }, async () => {
    while (nextIndex < ids.length && !signal.aborted) {
      const id = ids[nextIndex++]
      const [current, lookup] = await Promise.allSettled([
        client.getContact(id, { signal }), client.getGroupMember(groupId, id, { signal }),
      ])
      const contact = current.status === 'fulfilled' ? exactContact(current.value, id, accountId) : null
      // 暂时查不到资料可保留显示缓存，但已删除、404 或返回错人的联系人不再展示旧备注。
      const cached = current.status === 'rejected' && current.reason?.status !== 404
        ? exactContact(directory.contact?.(id), id, accountId) : null
      const memberReceipt = lookup.status === 'fulfilled' ? lookup.value : null
      const member = memberReceipt?.groupId === groupId && memberReceipt.username === id &&
        (memberReceipt.accountId === undefined || memberReceipt.accountId === accountId) &&
        memberReceipt.state === 'present' && exactProfileId(memberReceipt.member, id)
        ? memberReceipt.member : null
      const savedContact = contact || cached
      profiles.set(id, {
        speaker: speakerName(savedContact?.remark) || speakerName(member?.roomDisplayName) ||
          speakerName(savedContact?.nickname) || speakerName(member?.nickname),
        friend: contact?.relation === 'friend' ? contact : null,
      })
    }
  }))
  return profiles
}

/** 读取一个已授权会话的一页已保存记录，外发身份只有在真实送达证据支持时才标记。 */
export async function consoleConversation({ client, directory, ledgerPath, threadId, kind, cursor, limit }) {
  const accountId = client.accountId
  const currentScope = async () => {
    const candidates = conversationCandidates(await client.contactPolicy())
    return accountId && client.accountId === accountId ? candidates : new Map()
  }
  if ((await currentScope()).get(threadId)?.kind !== kind) {
    return { status: 404, payload: { ok: false, error: '当前本机白名单中没有这个会话。' } }
  }

  let page
  try {
    page = await client.listThreadMessages(threadId, { cursor: cursor || null, limit, signal: AbortSignal.timeout(HISTORY_BUDGET_MS) })
  } catch (error) {
    if (error.status === 409) return { status: 409, payload: { ok: false, error: '聊天记录已更新，请重新打开会话。' } }
    if (error.status !== 404) return { status: 502, payload: { ok: false, error: '暂时无法读取本机微信聊天记录。' } }
    page = { items: [], nextCursor: null }
  }
  const nextCursor = page?.nextCursor ?? null
  if (!Array.isArray(page?.items) || (nextCursor !== null &&
      (typeof nextCursor !== 'string' || !nextCursor || nextCursor === cursor ||
        Buffer.byteLength(nextCursor) > 8192 || /\p{Cc}/u.test(nextCursor)))) {
    return { status: 502, payload: { ok: false, error: '暂时无法核对本机微信聊天记录。' } }
  }
  const seen = new Set()
  const rows = []
  const now = Date.now()
  for (const row of page.items.slice(0, limit)) {
    if (!isThreadMessage(row, accountId, threadId) || seen.has(row.id)) continue
    const direction = savedDirection(row)
    const at = savedTime(row.createdAt, now)
    if (!at || !direction) continue
    seen.add(row.id)
    rows.push({ row, at, direction })
  }
  const [authors, profiles] = await Promise.all([
    deriveConversationAuthors({ client, ledgerPath, threadId,
      messages: rows.filter(item => item.direction === 'outbound').map(item => item.row) }),
    kind === 'group' ? groupSpeakerProfiles({ client, directory, accountId, groupId: threadId, rows }) : new Map(),
  ])
  const latest = await currentScope()
  if (latest.get(threadId)?.kind !== kind) return { status: 404, payload: { ok: false, error: '当前本机白名单中没有这个会话。' } }
  const items = rows.map(({ row, at, direction }) => {
    const profile = direction === 'inbound' ? profiles.get(row.senderId) : null
    const privateScope = profile?.friend && latest.get(row.senderId)?.kind === 'direct'
    const directRecipient = privateScope ? { id: row.senderId, kind: 'direct',
      displayName: speakerName(displayNameOf(profile.friend, row.senderId)) } : null
    return {
      id: row.id, at, direction, text: savedText(row),
      author: direction === 'inbound' ? 'contact' : authors.get(row.id) || 'outbound',
      speaker: profile?.speaker || null,
      ...(directRecipient ? { direct_recipient: directRecipient } : {}),
    }
  })
  return { status: 200, payload: { ok: true, id: threadId, kind, items, nextCursor } }
}
