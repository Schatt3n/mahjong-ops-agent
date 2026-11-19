import test from 'node:test'
import assert from 'node:assert/strict'
import { Directory, matchScore } from '../src/directory.mjs'

/** 构造分页客户端替身，覆盖联系人、群和会话读取。 */
function fakeClient() {
  return {
    contacts: [
      { id: 'wxid_wang', username: 'wxid_wang', remark: '王老板', nickname: '老王', displayName: '王老板' },
      { id: 'wxid_wang2', username: 'wxid_wang2', remark: '王老板', nickname: '王二', displayName: '王二' },
      { id: 'wxid_li', username: 'wxid_li', remark: '李姐', nickname: '李', displayName: '李姐' },
    ],
    groups: [{ id: '100@chatroom', roomId: '100@chatroom', displayName: '周末麻将群' }],
    async listContacts() {
      return { items: this.contacts, nextCursor: null }
    },
    async listGroups() {
      return { items: this.groups, nextCursor: null }
    },
    async getThread() {
      return null
    },
  }
}

test('匹配打分：完全相等优先于前缀和包含', () => {
  assert.equal(matchScore({ remark: '王老板' }, '王老板'), 100)
  assert.equal(matchScore({ remark: '王老板' }, '王'), 70)
  assert.equal(matchScore({ remark: '老王老板' }, '老板'), 40)
  assert.equal(matchScore({ remark: '李姐' }, '王'), 0)
})

test('唯一完全匹配才给 recommendedId', async () => {
  const directory = new Directory(fakeClient(), { logger: { warn() {} } })
  const result = await directory.resolve('李姐')
  assert.equal(result.recommendedId, 'wxid_li')
  assert.equal(result.ambiguous, false)
  assert.equal(result.exactCount, 1)
})

test('重名必须标记 ambiguous 且不给推荐', async () => {
  const directory = new Directory(fakeClient(), { logger: { warn() {} } })
  const result = await directory.resolve('王老板')
  assert.equal(result.exactCount, 2)
  assert.equal(result.recommendedId, null)
  assert.equal(result.ambiguous, true)
})

test('仅前缀命中不能自动选人', async () => {
  const directory = new Directory(fakeClient(), { logger: { warn() {} } })
  const result = await directory.resolve('周')
  const group = await directory.resolve('周末')
  assert.equal(group.recommendedId, null)
  assert.equal(group.ambiguous, true)
  // “周”会包含命中“周末麻将群”，但仍不能自动选人。
  assert.equal(result.recommendedId, null)
  assert.equal(result.ambiguous, true)
})

test('kind=group 只返回群', async () => {
  const directory = new Directory(fakeClient(), { logger: { warn() {} } })
  const result = await directory.resolve('周末麻将群', { kind: 'group' })
  assert.equal(result.candidates.length, 1)
  assert.equal(result.candidates[0].kind, 'group')
  assert.equal(result.recommendedId, '100@chatroom')
})

test('外发名单先过滤再限制候选数量，未授权同名者不会占位置', async () => {
  const directory = new Directory(fakeClient(), { logger: { warn() {} } })
  const result = await directory.resolve('王老板', {
    allowedRecipients: [{ thread_id: 'wxid_wang2', label: '熟客' }], limit: 1,
  })
  assert.deepEqual(result.candidates.map(item => item.id), ['wxid_wang2'])
  assert.deepEqual(result.cache, { contacts: 1, groups: 0 })
})

test('并发和等待超时后复用同一个刷新操作，部分写回如实返回', async () => {
  const contact = { username: 'wxid_li', nickname: '李' }
  const operation = { id: 'op-1', kind: 'metadata', accountId: 'acc' }
  let started = 0
  let settled = false
  const directory = new Directory({
    accountId: 'acc',
    getContact: async () => contact,
    startContactRefresh: async () => {
      started += 1
      return { ...operation, state: 'queued' }
    },
    getOperation: async id => {
      assert.equal(id, operation.id)
      return settled
        ? { ...operation, state: 'succeeded', result: { profileProjection: 'partial' } }
        : { ...operation, state: 'running' }
    },
  })
  const refresh = () => directory.refreshContact(contact.username, { timeoutMs: 40, intervalMs: 5 })
  const pending = await Promise.all([refresh(), refresh()])
  assert.deepEqual(pending, [{ pending: true }, { pending: true }])
  assert.equal(started, 1)
  assert.equal(directory.contact(contact.username), null)
  settled = true
  assert.deepEqual(await refresh(), { contact, projection: 'partial' })
  assert.equal(directory.contact(contact.username), contact)
  assert.equal(started, 1)
})

test('失败回执或读回不同联系人时保留当前缓存', async () => {
  const contact = { username: 'wxid_li', remark: '旧备注' }
  const operation = { id: 'op-1', kind: 'metadata', accountId: 'acc' }
  let settled = false
  let receipts = 0
  const directory = new Directory({
    accountId: 'acc',
    getContact: async () => receipts < 2 ? contact : { username: 'wxid_other', remark: '不可串写' },
    startContactRefresh: async () => operation,
    getOperation: async () => {
      receipts += 1
      return settled
        ? { ...operation, state: 'succeeded', result: { profileProjection: 'complete' } }
        : { ...operation, state: 'failed' }
    },
  })
  directory.contacts.push(contact)
  await assert.rejects(directory.refreshContact(contact.username), { code: 'refresh_contact_failed' })
  settled = true
  await assert.rejects(directory.refreshContact(contact.username), /different identity/)
  assert.deepEqual(directory.contacts, [contact])
})

test('定向联系人刷新完成后，较早开始的分页读取不会恢复旧备注', async () => {
  const oldContact = { username: 'wxid_li', remark: '旧备注' }
  const freshContact = { ...oldContact, remark: '新备注' }
  const operation = { id: 'op-1', kind: 'metadata', accountId: 'acc' }
  let finishList
  const list = new Promise(resolve => { finishList = resolve })
  let refreshed = false
  const directory = new Directory({
    accountId: 'acc',
    listContacts: async () => list,
    listGroups: async () => ({ items: [] }),
    getContact: async () => refreshed ? freshContact : oldContact,
    startContactRefresh: async () => operation,
    getOperation: async () => {
      refreshed = true
      return { ...operation, state: 'succeeded', result: { profileProjection: 'complete' } }
    },
  })
  const full = directory.refresh()
  assert.deepEqual(await directory.refreshContact(oldContact.username), { contact: freshContact, projection: 'complete' })
  finishList({ items: [oldContact] })
  await full
  assert.equal(directory.contact(oldContact.username), freshContact)
})
