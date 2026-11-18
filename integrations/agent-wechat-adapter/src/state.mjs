/**
 * 适配器本地状态：事件游标、外发去重账本、入站重放队列。
 *
 * 这些文件只属于适配器进程，不进入 Python Runtime 的数据库。游标文件用于重启后
 * 续传事件；投递账本用于避免同一源消息重复外发；重放队列用于 Runtime 暂时不可用时
 * 不丢消息。
 */
import fs from 'node:fs'
import path from 'node:path'

/** 创建仅当前用户可访问的状态目录。 */
function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 })
}

/** 读取事件游标；返回 null 表示尚未初始化。 */
export function loadCursor(cursorPath) {
  try {
    const payload = JSON.parse(fs.readFileSync(cursorPath, 'utf8'))
    const cursor = String(payload?.cursor ?? '').trim()
    return cursor && /^\d+$/.test(cursor) ? { ...payload, cursor } : null
  } catch {
    return null
  }
}

/** 原子写入游标；先写临时文件再 rename，避免崩溃留下半个 JSON。 */
export function saveCursor(cursorPath, { cursor, accountId = '', serverId = '' }) {
  ensureDir(path.dirname(cursorPath))
  const temp = `${cursorPath}.tmp`
  const body = JSON.stringify({ cursor: String(cursor), accountId, serverId, updated_at: new Date().toISOString() })
  fs.writeFileSync(temp, `${body}\n`, { mode: 0o600 })
  fs.renameSync(temp, cursorPath)
}

/** 读取已成功外发的源消息 id 集合，用于重启后跳过重复回复。 */
export function loadDeliveryLedger(ledgerPath) {
  const delivered = new Set()
  try {
    for (const line of fs.readFileSync(ledgerPath, 'utf8').split(/\r?\n/)) {
      if (!line.trim()) continue
      try {
        const record = JSON.parse(line)
        if (record?.source_message_id) delivered.add(String(record.source_message_id))
      } catch {
        /* 单行损坏不影响其余记录 */
      }
    }
  } catch {
    /* 首次启动没有账本 */
  }
  return delivered
}

/** 追加一条外发账本；发送成功后调用，失败不写。 */
export function appendDelivery(ledgerPath, record) {
  ensureDir(path.dirname(ledgerPath))
  fs.appendFileSync(ledgerPath, `${JSON.stringify({ ...record, delivered_at: new Date().toISOString() })}\n`, {
    mode: 0o600,
  })
}

/** 把一条待转发载荷落盘；文件名带源消息 id，保证重放可读。 */
export function persistToSpool(spoolDir, payload, error) {
  ensureDir(spoolDir)
  const safeId = String(payload?.source_message_id || payload?.message_id || Date.now()).replace(/[^\w.-]/g, '_')
  const file = path.join(spoolDir, `${Date.now()}_${safeId}.json`)
  fs.writeFileSync(file, JSON.stringify({ payload, error: String(error?.message || error), attempts: 0 }, null, 2), {
    mode: 0o600,
  })
  return file
}

/** 按写入顺序列出待重放文件。 */
export function listSpool(spoolDir) {
  try {
    return fs
      .readdirSync(spoolDir)
      .filter((name) => name.endsWith('.json'))
      .sort()
      .map((name) => path.join(spoolDir, name))
  } catch {
    return []
  }
}

/** 删除已成功重放的文件，不存在时视为已完成。 */
export function removeSpoolFile(file) {
  try {
    fs.unlinkSync(file)
  } catch {
    /* 已被其他重放流程删除 */
  }
}
