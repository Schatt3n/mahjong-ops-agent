/**
 * 联系人 / 群 / 会话目录。
 *
 * 业务系统按名字找人，但微信的名字不唯一、可以随时改，也没有拼音检索。这里只负责
 * 给出候选和置信度，绝不自动替业务系统选定收件人：重名、仅前缀命中和远程搜索命中的
 * 情况必须由上层确认。
 */
import { displayNameOf, isChatroomId } from './normalize.mjs'

const MAX_PAGES = 50

/** 把一页接口结果读成数组，兼容 {items:[...]} 与直接数组两种返回。 */
function itemsOf(payload) {
  if (Array.isArray(payload)) return payload
  return Array.isArray(payload?.items) ? payload.items : []
}

/** 逐页读完一个列表；上限用于防止上游异常时分页死循环。 */
async function readAllPages(fetchPage, limit = 200) {
  const collected = []
  let cursor = null
  for (let page = 0; page < MAX_PAGES; page += 1) {
    const payload = await fetchPage({ cursor, limit })
    collected.push(...itemsOf(payload))
    cursor = payload?.nextCursor ?? null
    if (!cursor) break
  }
  return collected
}

/** 名字匹配打分：完全相等 > 前缀 > 包含；可计入外发名单中的其他别名。 */
export function matchScore(candidate, rawQuery, aliases = []) {
  const query = String(rawQuery || '').trim().toLowerCase()
  if (!query) return 0
  const fields = [
    candidate.remark,
    candidate.displayName,
    candidate.nickname,
    candidate.alias,
    candidate.id,
    ...aliases,
  ]
    .map((value) => String(value || '').trim().toLowerCase())
    .filter(Boolean)

  let best = 0
  for (const field of fields) {
    if (field === query) best = Math.max(best, 100)
    else if (field.startsWith(query)) best = Math.max(best, 70)
    else if (field.includes(query)) best = Math.max(best, 40)
  }
  return best
}

export class Directory {
  constructor(client, { contactTtlMs = 600_000, threadTtlMs = 300_000, logger = console } = {}) {
    this.client = client
    this.contactTtlMs = contactTtlMs
    this.threadTtlMs = threadTtlMs
    this.logger = logger
    this.contacts = []
    this.groups = []
    this.threads = new Map()
    this.contactsLoadedAt = 0
    this.groupsLoadedAt = 0
  }

  /** 计算缓存是否过期；失败时保留旧数据并记日志，避免检索路径被上游抖动打断。 */
  async refresh({ force = false } = {}) {
    const now = Date.now()
    if (!force && now - this.contactsLoadedAt < this.contactTtlMs && now - this.groupsLoadedAt < this.contactTtlMs) {
      return
    }
    try {
      this.contacts = await readAllPages((params) => this.client.listContacts(params))
      this.contactsLoadedAt = Date.now()
    } catch (error) {
      this.logger.warn?.(`[directory] contacts refresh failed: ${error?.message || error}`)
    }
    try {
      this.groups = await readAllPages((params) => this.client.listGroups(params))
      this.groupsLoadedAt = Date.now()
    } catch (error) {
      this.logger.warn?.(`[directory] groups refresh failed: ${error?.message || error}`)
    }
  }

  /** 读取会话展示信息；群消息只需要 topic，私聊只需要备注，均带短 TTL。 */
  async getThread(threadId) {
    const key = String(threadId || '')
    const cached = this.threads.get(key)
    if (cached && Date.now() - cached.loadedAt < this.threadTtlMs) return cached.value
    try {
      const value = await this.client.getThread(key)
      this.threads.set(key, { value, loadedAt: Date.now() })
      return value
    } catch (error) {
      this.logger.warn?.(`[directory] thread ${key} lookup failed: ${error?.message || error}`)
      return cached?.value || null
    }
  }

  /** 优先用本地缓存找联系人；返回 null 时由调用方决定是否回退到 id。 */
  contact(contactId) {
    const id = String(contactId || '')
    return this.contacts.find((item) => String(item.username || item.id) === id) || null
  }

  /** 提交已选择的私聊时重查当前通讯录，搜索命中必须与上游的联系人标识精确相等。 */
  async hasExactContact(id) {
    const contacts = await readAllPages(params => this.client.listContacts({ ...params, search: id }))
    return contacts.some(contact => String(contact.username || contact.id || '') === id)
  }

  /**
   * 按名字检索可发送目标。
   *
   * 返回 `ambiguous` 让上层强制确认；只有在“唯一完全匹配”时才给 recommendedId，
   * 且不会直接消费它。配置外发名单时先按精确会话过滤，再排序和截断，
   * 避免未授权候选占用可选结果或出现在响应中。
   */
  async resolve(query, { kind = '', limit = 10, allowedRecipients = null } = {}) {
    const allowedIds = allowedRecipients && new Set(allowedRecipients.map(recipient => recipient.thread_id))
    const canSearch = !allowedIds || allowedIds.size > 0
    if (canSearch) await this.refresh()

    const local = []
    for (const contact of this.contacts) {
      const id = String(contact.username || contact.id || '')
      if (!id) continue
      local.push({
        id,
        threadId: id,
        kind: 'direct',
        displayName: displayNameOf(contact, id),
        remark: contact.remark ?? null,
        nickname: contact.nickname ?? null,
        alias: contact.alias ?? null,
        source: 'contact_cache',
      })
    }
    for (const group of this.groups) {
      const id = String(group.id || group.roomId || '')
      if (!id) continue
      local.push({
        id,
        threadId: id,
        kind: 'group',
        displayName: displayNameOf(group, id),
        remark: group.remark ?? null,
        nickname: null,
        alias: null,
        source: 'group_cache',
      })
    }

    const remote = []
    if (canSearch && String(query || '').trim()) {
      const searches = []
      if (!kind || kind === 'direct') searches.push(this.client.listContacts({ search: query, limit }))
      if (!kind || kind === 'group') searches.push(this.client.listGroups({ search: query, limit }))
      const settled = await Promise.allSettled(searches)
      for (const result of settled) {
        if (result.status !== 'fulfilled') continue
        for (const item of itemsOf(result.value)) {
          const id = String(item.username || item.id || item.roomId || '')
          if (!id) continue
          remote.push({
            id,
            threadId: id,
            kind: isChatroomId(id) ? 'group' : 'direct',
            displayName: displayNameOf(item, id),
            remark: item.remark ?? null,
            nickname: item.nickname ?? null,
            alias: item.alias ?? null,
            source: 'remote_search',
          })
        }
      }
    }

    const policy = (allowedRecipients || []).map(recipient => ({
      id: recipient.thread_id,
      threadId: recipient.thread_id,
      kind: isChatroomId(recipient.thread_id) ? 'group' : 'direct',
      displayName: recipient.label,
      source: 'outbound_policy',
      aliases: recipient.aliases,
    }))
    const merged = new Map()
    for (const { aliases, ...candidate } of [...local, ...remote, ...policy]) {
      if (kind && candidate.kind !== kind) continue
      if (allowedIds && !allowedIds.has(candidate.id)) continue
      const score = matchScore(candidate, query, aliases)
      if (score === 0) continue
      const previous = merged.get(candidate.id)
      if (!previous || score > previous.score) merged.set(candidate.id, { ...candidate, score })
    }

    const candidates = [...merged.values()]
      .sort((a, b) => b.score - a.score || a.displayName.localeCompare(b.displayName))
      .slice(0, limit)

    const exact = candidates.filter(candidate => candidate.score === 100)
    const recommendedId = exact.length === 1 ? exact[0].id : null
    const allowedGroupCount = allowedIds ? [...allowedIds].filter(isChatroomId).length : 0
    return {
      query: String(query || ''),
      kind: kind || 'any',
      candidates,
      exactCount: exact.length,
      // 只有一个完全匹配、且没有其他候选时才可直接发送；前缀/包含命中必须人工确认。
      ambiguous: recommendedId === null || candidates.length > 1,
      recommendedId,
      cache: allowedIds
        ? { contacts: allowedIds.size - allowedGroupCount, groups: allowedGroupCount }
        : { contacts: this.contacts.length, groups: this.groups.length },
    }
  }
}
