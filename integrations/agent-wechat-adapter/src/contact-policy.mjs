/**
 * 本机联系人权限文件。控制台编辑和发送校验共用账号绑定的实时快照，
 * 改动按原子文件替换发布，缺失的可选默认文件沿用原有通道行为。
 */
import { mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { isChatroomId } from './normalize.mjs'

/** 原生标识用精确路径查找，排除控制字符和 URI 路径特例。 */
function isExactPermissionId(id) {
  return typeof id === 'string' && Buffer.byteLength(id) <= 256 && /^[^\s\p{Cc}\p{Cs}/\\]+$/u.test(id) &&
    id !== '.' && id !== '..'
}

/** 只识别可接受精确寻址的私聊联系人。 */
export function isDirectContactId(id) {
  return isExactPermissionId(id) && !isChatroomId(id.toLowerCase()) && !id.toLowerCase().endsWith('@im.chatroom')
}

/** 群 ID 的分类应与现有发送和入站逻辑一致。 */
export function isGroupPermissionId(id) {
  return isExactPermissionId(id) && isChatroomId(id)
}

/** 校验既有外发名单合同，包括控制台编辑时需保留的群标签和别名。 */
function validRecipient(item) {
  return item && typeof item.thread_id === 'string' && /^[^\s\p{Cc}]+$/u.test(item.thread_id) &&
    typeof item.label === 'string' && Boolean(item.label.trim()) &&
    validAutomation(item) &&
    (!('aliases' in item) || (Array.isArray(item.aliases) && item.aliases.every(alias =>
      typeof alias === 'string' && Boolean(alias.trim()))))
}

/** 许可文件中的自动化字段只接受布尔值，未设置时沿用对应旧规则。 */
export function validAutomation(flags) {
  return ['managed', 'auto_reply'].every(key => flags[key] === undefined || typeof flags[key] === 'boolean')
}

/** 新启用的群授权必须以有时区的时间戳绑定入站窗口。 */
function validActivation(value) {
  return typeof value === 'string' && /(?:Z|[+-]\d{2}:\d{2})$/u.test(value) && Number.isFinite(Date.parse(value))
}

/** 从固定文件读取当前账号的权限；不可信文件继续限制发送，并拒绝被界面覆盖。 */
export async function readContactPolicy(path, accountId, configured) {
  const empty = {
    direct: configured, group: configured, managed: false,
    recipients: [], observed: [], managedOnly: [], groupPermissions: [], document: null,
  }
  const invalid = { ...empty, direct: true, group: true, invalid: true }
  if (!path) return empty
  let document
  try {
    document = JSON.parse(await readFile(path, 'utf8'))
  } catch (error) {
    if (error.code === 'ENOENT') return empty
    return invalid
  }
  const recipients = document?.recipients
  const observed = document?.observed === undefined ? [] : document.observed
  const managedOnly = document?.managed_only === undefined ? [] : document.managed_only
  const managed = document?.managed_direct === true
  const groupPermissions = document?.group_permissions === undefined ? [] : document.group_permissions
  if (!accountId || document?.account_id !== accountId ||
      (!managed && (document?.managed_direct !== undefined || (!configured && document?.group_permissions === undefined))) ||
      !Array.isArray(recipients) || !recipients.every(validRecipient) ||
      !Array.isArray(observed) || !observed.every(item => validRecipient(item) && isDirectContactId(item.thread_id)) ||
      !Array.isArray(managedOnly) || (!managed && managedOnly.length !== 0) ||
      !managedOnly.every(item => validRecipient(item) && isDirectContactId(item.thread_id) &&
        item.managed !== false && item.auto_reply !== true) ||
      !Array.isArray(groupPermissions) || !groupPermissions.every(item => validRecipient(item) &&
        isGroupPermissionId(item.thread_id) && ['allowlist', 'observe', 'inherit'].includes(item.mode) &&
        (item.managed !== true || validActivation(item.managed_at)) &&
        (item.auto_reply !== true || validActivation(item.auto_reply_at))) ||
      new Set(groupPermissions.map(item => item.thread_id)).size !== groupPermissions.length) {
    return invalid
  }
  // 重复配置不得放宽任何权限。仅托管比发送白名单更窄，既有明确的托管禁用也继续生效。
  const observedById = new Map(observed.map(item => [item.thread_id, item]))
  const deniedManagement = new Set(recipients.filter(item => item.managed === false).map(item => item.thread_id))
  for (const item of managedOnly) {
    if (deniedManagement.has(item.thread_id) && !observedById.has(item.thread_id)) observedById.set(item.thread_id, item)
  }
  const managedOnlyById = new Map(managedOnly.filter(item => !observedById.has(item.thread_id))
    .map(item => [item.thread_id, item]))
  const allowed = recipients.filter(item => !observedById.has(item.thread_id) && !managedOnlyById.has(item.thread_id) &&
    (configured || isDirectContactId(item.thread_id)))
  return { direct: configured || managed, group: configured, managed, recipients: allowed,
    observed: [...observedById.values()], managedOnly: [...managedOnlyById.values()], groupPermissions, document }
}

/** 读取群手动发送覆盖，inherit 不改动原有群名单。 */
export function groupPermissionMode(policy, id) {
  const mode = groupPermissionRule(policy, id)?.mode
  return mode === 'allowlist' || mode === 'observe' ? mode : null
}

/** 同一账号内的群自动化按精确群标识查询，读取端已拒绝重复条目。 */
export function groupPermissionRule(policy, id) {
  return policy.groupPermissions?.find(item => item.thread_id === id) || null
}

/** 群观察优先于已保存授权；其余群通过 null 表明沿用既有群策略。 */
export function groupAutomation(policy, id) {
  const rule = groupPermissionRule(policy, id)
  if (rule?.mode === 'observe') return { managed: false, auto_reply: false }
  return { managed: rule?.managed ?? null, auto_reply: rule?.auto_reply ?? null }
}

/** 私聊观察和未获准条目没有业务或回复权限；历史白名单条目沿用已授权的默认值。 */
export function directAutomation(policy, id) {
  if (!policy.direct) return { managed: true, auto_reply: true }
  if (policy.managedOnly?.some(item => item.thread_id === id)) return { managed: true, auto_reply: false }
  const rule = policy.recipients.find(item => item.thread_id === id)
  const managed = Boolean(rule && rule.managed !== false)
  return { managed, auto_reply: Boolean(managed && rule.auto_reply !== false) }
}

/** 判断手动发送能否单独使用总闸；群覆盖只为明确选择的群提供这项权限。 */
export function hasManualRecipientScope(policy, id) {
  return isChatroomId(id) ? policy.group || groupPermissionMode(policy, id) === 'allowlist' : policy.direct
}

/** 逐条检查外发资格，群白名单覆盖只允许手动发送，群观察对任何来源生效。 */
export function permitsRecipientSend(policy, id, manual, {
  scopedGroupReply = null, incomingReply = false, legacyManagedGroupReply = false,
} = {}) {
  const group = isChatroomId(id)
  if (group) {
    const override = groupPermissionMode(policy, id)
    if (override === 'observe') return false
    if (manual && override === 'allowlist') return true
    // 只有适配器核实的原生同群回执能用群自动回复覆盖，定时任务和看板不会继承这个许可。
    const rule = groupPermissionRule(policy, id)
    if (scopedGroupReply && !scopedGroupReply.legacy) return rule?.auto_reply === true &&
      Date.parse(scopedGroupReply.receivedAt) >= Date.parse(rule.auto_reply_at)
    if (incomingReply || scopedGroupReply?.legacy) {
      // 旧授权的群回执或托管群回复仍须命中显式群外发名单；普通后台回复不能沿用未限定的范围。
      return (scopedGroupReply?.legacy === true || legacyManagedGroupReply) && rule?.auto_reply === undefined &&
        policy.group && policy.recipients.some(item => item.thread_id === id)
    }
  } else if (!manual && !directAutomation(policy, id).auto_reply) {
    return false
  }
  return !(group ? policy.group : policy.direct) || policy.recipients.some(item => item.thread_id === id)
}

/** 在不改变未选条目和群配置的前提下切换单个联系人的权限，保存后读回同一份语义。 */
export async function saveContactPolicy(path, accountId, configured, id, label, mode, automation = {}) {
  if (!path || !accountId || !isDirectContactId(id) || typeof label !== 'string' || !label.trim() ||
      !['allowlist', 'managed_only', 'observe'].includes(mode) || !validAutomation(automation)) return null
  const previous = await readContactPolicy(path, accountId, configured)
  if (previous.invalid) return null
  const document = previous.document || { account_id: accountId, recipients: [] }
  // 原文件可能给私聊配置了专用标签和别名，切换权限仍保留这些寻址信息。
  const savedManagedOnly = document.managed_only?.find(item => item.thread_id === id)
  const existing = document.recipients.find(item => item.thread_id === id) ||
    document.observed?.find(item => item.thread_id === id) || savedManagedOnly
  const recipients = document.recipients.filter(item => item.thread_id !== id)
  const observed = (document.observed || []).filter(item => item.thread_id !== id)
  const managedOnly = (document.managed_only || []).filter(item => item.thread_id !== id)
  // 单独授予发送资格时，原本仅托管的联系人仍需明确授权才能自动回复。
  const entry = { ...existing, ...(savedManagedOnly ? { auto_reply: false } : {}), ...automation,
    thread_id: id, label: existing?.label || label }
  if ((mode === 'observe' && Object.values(automation).includes(true)) ||
      (mode === 'managed_only' && (automation.managed === false || automation.auto_reply === true)) ||
      (mode === 'allowlist' && automation.auto_reply === true && entry.managed === false)) {
    return { conflict: true }
  }
  if (mode === 'allowlist') recipients.push(entry)
  else if (mode === 'managed_only') {
    delete entry.managed
    delete entry.auto_reply
    managedOnly.push(entry)
  } else observed.push(entry)
  const updated = { ...document, managed_direct: true, recipients, observed }
  if (managedOnly.length || document.managed_only !== undefined) updated.managed_only = managedOnly
  return persistContactPolicy(path, accountId, configured, updated)
}

/** 编辑单独群覆盖，不改变原有群白名单、群托管或私聊规则。 */
export async function saveGroupPermission(path, accountId, configured, id, label, mode, automation = {}) {
  if (!path || !accountId || !isGroupPermissionId(id) || typeof label !== 'string' || !label.trim() ||
      !['allowlist', 'observe', 'inherit'].includes(mode) || !validAutomation(automation)) return null
  const previous = await readContactPolicy(path, accountId, configured)
  if (previous.invalid) return null
  if (mode === 'observe' && Object.values(automation).includes(true)) return { conflict: true }
  const existing = groupPermissionRule(previous, id)
  const next = { ...existing, ...automation, thread_id: id, label, mode }
  for (const key of ['managed', 'auto_reply']) {
    const timestamp = `${key}_at`
    if (automation[key] === true && existing?.[key] !== true) next[timestamp] = new Date().toISOString()
    if (automation[key] === false) delete next[timestamp]
  }
  const groupPermissions = previous.groupPermissions.filter(item => item.thread_id !== id)
  if (mode !== 'inherit' || next.managed !== undefined || next.auto_reply !== undefined) groupPermissions.push(next)
  else if (!existing) return previous
  const document = previous.document || { account_id: accountId, recipients: [] }
  return persistContactPolicy(path, accountId, configured, { ...document, group_permissions: groupPermissions })
}

/** 两类编辑共用原子替换，不让读取方看见尚未写全的授权文件。 */
async function persistContactPolicy(path, accountId, configured, updated) {
  const parent = dirname(path)
  await mkdir(parent, { recursive: true, mode: 0o700 })
  const staging = await mkdtemp(join(parent, '.contact-policy-'))
  try {
    const candidate = join(staging, 'policy.json')
    await writeFile(candidate, `${JSON.stringify(updated, null, 2)}\n`, { mode: 0o600, flag: 'wx' })
    await rename(candidate, path)
  } finally {
    await rm(staging, { recursive: true, force: true })
  }
  return readContactPolicy(path, accountId, configured)
}
