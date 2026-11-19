/**
 * 联系人 / 群 / 会话目录。
 *
 * 业务系统按名字找人，但微信的名字不唯一、可以随时改，也没有拼音检索。这里只负责
 * 给出候选和置信度，绝不自动替业务系统选定收件人：重名、仅前缀命中和远程搜索命中的
 * 情况必须由上层确认。
 */
import { setTimeout as delay } from 'node:timers/promises'
import { UpstreamError } from './client.mjs'
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

/** 单条资料以微信标识覆盖，保留目录内的其他联系人。 */
function upsertContact(contacts, id, contact) {
  const index = contacts.findIndex(item => (item.username || item.id) === id)
  if (index < 0) contacts.push(contact)
  else contacts[index] = contact
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
    this.contactRevision = 0
    this.contactLoads = new Set()
    this.contactRefreshes = new Map()
  }

  /** 守护进程保存联系人变更后使目录过期，合并连续通知，在下次读取时再拉取。 */
  invalidateContacts() {
    this.contactRevision += 1
    this.contactsLoadedAt = 0
  }

  /** 计算缓存是否过期；失败时保留旧数据并记日志，避免检索路径被上游抖动打断。 */
  async refresh({ force = false } = {}) {
    const now = Date.now()
    if (force || now - this.contactsLoadedAt >= this.contactTtlMs) {
      const revision = this.contactRevision
      const updatedContacts = new Map()
      this.contactLoads.add(updatedContacts)
      try {
        const contacts = await readAllPages((params) => this.client.listContacts(params))
        // 同步通知后的读取才可确认缓存有效，旧批次不能覆盖新备注或延长旧缓存。
        if (revision === this.contactRevision) {
          for (const [id, contact] of updatedContacts) upsertContact(contacts, id, contact)
          this.contacts = contacts
          this.contactsLoadedAt = Date.now()
        }
      } catch (error) {
        this.logger.warn?.(`[directory] contacts refresh failed: ${error?.message || error}`)
      } finally {
        this.contactLoads.delete(updatedContacts)
      }
    }
    if (force || now - this.groupsLoadedAt >= this.contactTtlMs) {
      try {
        this.groups = await readAllPages((params) => this.client.listGroups(params))
        this.groupsLoadedAt = Date.now()
      } catch (error) {
        this.logger.warn?.(`[directory] groups refresh failed: ${error?.message || error}`)
      }
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

  /** 合并同一联系人的并发刷新，超时后继续核对已提交的操作。 */
  async refreshContact(id, { timeoutMs = 11_000, intervalMs = 250 } = {}) {
    const refresh = this.contactRefreshes.get(id) || { operationId: '', task: null }
    if (!refresh.task) {
      this.contactRefreshes.set(id, refresh)
      refresh.task = this.#runContactRefresh(id, refresh, { timeoutMs, intervalMs }).finally(() => {
        refresh.task = null
        if (!refresh.operationId) this.contactRefreshes.delete(id)
      })
    }
    return refresh.task
  }

  /** 等原生刷新成功并读回同一个联系人后，才把保存结果写入适配器缓存。 */
  async #runContactRefresh(id, refresh, { timeoutMs, intervalMs }) {
    const signal = AbortSignal.timeout(timeoutMs)
    try {
      if (!refresh.operationId) {
        const current = await this.client.getContact(id, { signal }).catch(error => {
          if (error.status === 404) return null
          throw error
        })
        if ((current?.username || current?.id) !== id) {
          throw new UpstreamError('exact contact not found', { status: 404, code: 'refresh_contact_not_found' })
        }
        const operation = await this.client.startContactRefresh(id, { signal })
        if (typeof operation?.id !== 'string' || !operation.id || operation.kind !== 'metadata' ||
            operation.accountId !== this.client.accountId) {
          throw new UpstreamError('invalid contact refresh receipt', { status: 502 })
        }
        refresh.operationId = operation.id
      }

      let operation
      while (true) {
        operation = await this.client.getOperation(refresh.operationId, { signal }).catch(error => {
          if (error.status === 404) refresh.operationId = ''
          throw error
        })
        if (operation?.id !== refresh.operationId || operation.kind !== 'metadata' ||
            operation.accountId !== this.client.accountId) {
          throw new UpstreamError('invalid contact refresh operation', { status: 502 })
        }
        if (!['queued', 'running'].includes(operation.state)) break
        await delay(intervalMs, undefined, { signal })
      }
      const projection = operation.result?.profileProjection
      if (operation.state !== 'succeeded' || !['complete', 'partial'].includes(projection)) {
        refresh.operationId = ''
        throw new UpstreamError('contact refresh did not complete', { status: 502, code: 'refresh_contact_failed' })
      }
      const contact = await this.client.getContact(id, { signal })
      if ((contact?.username || contact?.id) !== id) {
        throw new UpstreamError('contact refresh returned a different identity', { status: 502 })
      }
      refresh.operationId = ''
      upsertContact(this.contacts, id, contact)
      for (const load of this.contactLoads) load.set(id, contact)
      this.threads.delete(id)
      return { contact, projection }
    } catch (error) {
      if (signal.aborted && refresh.operationId) return { pending: true }
      throw error
    }
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
    const freshContactIds = new Set()
    let contactSearchComplete = false
    if (canSearch && String(query || '').trim()) {
      const [direct, group] = await Promise.allSettled([
        !kind || kind === 'direct' ? this.client.listContacts({ search: query, limit }) : null,
        !kind || kind === 'group' ? this.client.listGroups({ search: query, limit }) : null,
      ])
      const fresh = direct.status === 'fulfilled' ? direct.value : null
      contactSearchComplete = (Array.isArray(fresh) || Array.isArray(fresh?.items)) && !fresh.nextCursor
      for (const result of [direct, group]) {
        if (result.status !== 'fulfilled') continue
        for (const item of itemsOf(result.value)) {
          const id = String(item.username || item.id || item.roomId || '')
          if (!id) continue
          if (result === direct) freshContactIds.add(id)
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
      // 先按精确 ID 选定资料来源，再评分。完整搜索的空结果也能排除只在旧缓存存在的备注。
      if (candidate.source === 'contact_cache' && (contactSearchComplete || freshContactIds.has(candidate.id))) continue
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
