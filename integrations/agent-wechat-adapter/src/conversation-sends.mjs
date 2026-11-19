/** 用已接受的原生发送回执确认工作台聊天中的消息归属。 */
import fs from 'node:fs'
import { appendDelivery } from './state.mjs'

const MAX_LEDGER_BYTES = 512 * 1024
const MAX_LEGACY_LOOKUPS = 8
const LEGACY_LOOKUP_TIMEOUT_MS = 1800

/** 账号、会话和回执都使用原生精确标识，空值或显示名不能补足证据。 */
function validId(value) {
  return typeof value === 'string' && value.length > 0 && value.length <= 256 && !/[\s\p{Cc}]/u.test(value)
}

/** 只读固定大小的账本尾部。文件正在写入或某行损坏时，其余完整回执仍有效。 */
function recentLedgerRows(ledgerPath) {
  let file
  try {
    file = fs.openSync(ledgerPath, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK)
    const stats = fs.fstatSync(file)
    if (!stats.isFile() || stats.size === 0) return []
    const offset = Math.max(0, stats.size - MAX_LEDGER_BYTES)
    const buffer = Buffer.alloc(stats.size - offset)
    const length = fs.readSync(file, buffer, 0, buffer.length, offset)
    const lines = buffer.toString('utf8', 0, length).split('\n')
    // 从中间读取时，先丢弃可能被截断的首行；完整尾行仍可用 JSON 独立核对。
    if (offset) lines.shift()
    return lines.flatMap(line => {
      try {
        const row = JSON.parse(line)
        return row?.state === 'accepted' ? [row] : []
      } catch {
        return []
      }
    })
  } catch {
    return []
  } finally {
    if (file !== undefined) fs.closeSync(file)
  }
}

/** 验证原生发送终态绑定当前账号和具体目标；只有可关联聊天消息的回执可用于署名。 */
function acceptedReceipt(state, account, thread) {
  if (state?.state !== 'accepted' || state.accountId !== account || state.destination !== thread || !validId(state.id)) return null
  const local = validId(state.localMessageId) ? state.localMessageId : null
  const server = validId(state.serverId) ? state.serverId : null
  return local || server ? { local_message_id: local, server_message_id: server } : null
}

/** 同时存在两端标识时，冲突的服务端 ID 不能由本地 ID 掩盖。 */
function receiptMatchesMessage(receipt, message) {
  if (validId(message.serverId) && validId(receipt.server_message_id)) return message.serverId === receipt.server_message_id
  return receipt.local_message_id === message.id
}

/** 幂等补记新控制面发送回执，保持自动回复去重使用的入站源 ID 独立。 */
export function recordConversationSend(ledgerPath, { client, threadId, state, author, logger = console }) {
  const account = client?.accountId
  if (!ledgerPath || !validId(account) || !validId(threadId) || !['agent', 'owner'].includes(author)) return false
  const identity = acceptedReceipt(state, account, threadId)
  if (!identity) return false
  if (recentLedgerRows(ledgerPath).some(row => row.account_id === account && row.thread_id === threadId && row.send_id === state.id)) {
    return true
  }
  try {
    appendDelivery(ledgerPath, {
      account_id: account, thread_id: threadId, send_id: state.id, state: 'accepted', author, ...identity,
    })
    return true
  } catch {
    // 平台回执仍有效，账本错误不能使成功发送被重试。
    logger.warn?.('无法保存微信会话发送回执')
    return false
  }
}

/** 旧账本缺少账号字段，需要在本账号的上游发送记录中复核，超时时仍可显示原生消息。 */
async function verifyLegacyAuthor(client, row, thread, message) {
  let timer
  try {
    const state = await Promise.race([
      client.getSend(row.send_id),
      new Promise(resolve => { timer = setTimeout(resolve, LEGACY_LOOKUP_TIMEOUT_MS) }),
    ])
    const receipt = acceptedReceipt(state, client.accountId, thread)
    if (!receipt || state.id !== row.send_id) return false
    return receiptMatchesMessage(receipt, message)
  } catch {
    return false
  } finally {
    clearTimeout(timer)
  }
}

/** 只为归属有回执证据的实际外发消息返回署名，缺少或冲突的证据保留为未知。 */
export async function deriveConversationAuthors({ client, ledgerPath, threadId, messages }) {
  const account = client?.accountId
  const authors = new Map()
  if (!validId(account) || !validId(threadId) || !Array.isArray(messages)) return authors
  const byLocal = new Map()
  const byServer = new Map()
  for (const message of messages) {
    if (message?.accountId !== account || message.threadId !== threadId || message.direction !== 'outgoing' || !validId(message.id)) continue
    byLocal.set(message.id, message)
    if (validId(message.serverId)) {
      byServer.set(message.serverId, byServer.has(message.serverId) ? null : message)
    }
  }
  if (!byLocal.size) return authors
  const rows = recentLedgerRows(ledgerPath)
  for (const row of rows) {
    if (row.account_id !== account || row.thread_id !== threadId || !validId(row.send_id) || !['agent', 'owner'].includes(row.author)) continue
    const local = byLocal.get(row.local_message_id)
    const server = byServer.get(row.server_message_id)
    if (local && server && local.id !== server.id) continue
    const message = local || server
    if (!message || !receiptMatchesMessage(row, message)) continue
    const previous = authors.get(message.id)
    authors.set(message.id, previous === undefined || previous === row.author ? row.author : null)
  }

  const kind = threadId.endsWith('@chatroom') ? 'room' : 'contact'
  const historicalLookups = []
  const checked = new Set()
  for (const row of rows.reverse()) {
    if (historicalLookups.length >= MAX_LEGACY_LOOKUPS) break
    if (row.account_id !== undefined || row.thread_id !== undefined || row.author !== undefined ||
        row.conversation_id !== `wechaty:${kind}:${threadId}` || !validId(row.source_message_id) ||
        !validId(row.send_id) || checked.has(row.send_id)) continue
    const message = byLocal.get(row.message_id) || byServer.get(row.message_id)
    if (!message || authors.has(message.id)) continue
    checked.add(row.send_id)
    historicalLookups.push({ row, message })
  }
  await Promise.all(historicalLookups.map(async ({ row, message }) => {
    if (await verifyLegacyAuthor(client, row, threadId, message)) authors.set(message.id, 'agent')
  }))
  for (const [id, author] of authors) {
    if (author === null) authors.delete(id)
  }
  return authors
}
