import test from 'node:test'
import assert from 'node:assert/strict'
import { AgentWechatClient } from '../src/client.mjs'
import { filterListedGroups } from '../src/group-directory.mjs'

test('群列表和详情使用同一个账号下的 GET 请求与取消时限', async () => {
  const calls = []
  const sdk = new AgentWechatClient({
    baseUrl: 'http://127.0.0.1:8080',
    accountId: 'account/a',
    fetchImpl: async (url, init) => {
      calls.push([url, init.method, init.signal.aborted])
      return Response.json({ items: [] })
    },
  })
  const controller = new AbortController()
  await sdk.listGroups({ cursor: 'opaque+=', limit: 12 }, { signal: controller.signal })
  await sdk.getGroup('12@chatroom', { signal: controller.signal })
  controller.abort()
  await sdk.getGroup('12@chatroom', { signal: controller.signal })
  assert.deepEqual(calls, [
    ['http://127.0.0.1:8080/v1/accounts/account%2Fa/groups?cursor=opaque%2B%3D&limit=12', 'GET', false],
    ['http://127.0.0.1:8080/v1/accounts/account%2Fa/groups/12%40chatroom', 'GET', false],
    ['http://127.0.0.1:8080/v1/accounts/account%2Fa/groups/12%40chatroom', 'GET', true],
  ])
})

test('仅隐藏明确退群的条目，详情未知、出错或身份不符时保留原有顺序', async () => {
  const groups = [
    { id: '1@chatroom', isInGroup: false },
    { id: '2@chatroom', isInGroup: true },
    { roomId: '3@chatroom' },
    { id: '4@chatroom' },
    { id: '5@chatroom' },
    { id: '6@chatroom' },
    { id: '7@chatroom', isInGroup: 'false' },
    { id: '8@chatroom' },
    { id: '9@chatroom', deleted: true },
  ]
  const calls = []
  const client = {
    accountId: 'selected-account',
    async getGroup(id) {
      calls.push(id)
      if (id === '3@chatroom') return { id, isInGroup: false }
      if (id === '4@chatroom') return { id: '1@chatroom', isInGroup: false }
      if (id === '5@chatroom') throw new Error('unavailable')
      if (id === '6@chatroom') return { id, isInGroup: false, accountId: 'another-account' }
      if (id === '8@chatroom') return { id, roomId: '1@chatroom', isInGroup: false }
      return { id }
    },
  }
  const actual = await filterListedGroups(client, groups)
  assert.deepEqual(calls, ['3@chatroom', '4@chatroom', '5@chatroom', '6@chatroom', '7@chatroom', '8@chatroom'])
  assert.deepEqual(actual, [groups[1], ...groups.slice(3, -1)])
})

test('停止核对时取消并发详情请求，保留尚未核实的群', async () => {
  const groups = Array.from({ length: 20 }, (_, index) => ({ id: `${index}@chatroom` }))
  const controller = new AbortController()
  let requested = 0
  const client = {
    getGroup(_id, { signal }) {
      requested++
      assert.equal(signal, controller.signal)
      return new Promise((_, reject) => {
        signal.addEventListener('abort', () => reject(signal.reason), { once: true })
      })
    },
  }
  const reading = filterListedGroups(client, groups, { signal: controller.signal })
  assert.equal(requested, 8)
  controller.abort()
  assert.deepEqual(await reading, groups)
  assert.equal(requested, 8)
})
