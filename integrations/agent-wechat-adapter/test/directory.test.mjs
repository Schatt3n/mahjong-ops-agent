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
