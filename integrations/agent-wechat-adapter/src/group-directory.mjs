/** 群目录只读投影。历史群可能仍留在上游列表，成员状态优先使用已保存的精确详情。 */
import { isGroupPermissionId } from './contact-policy.mjs'

const GROUP_DETAIL_CONCURRENCY = 8
const GROUP_DETAIL_BUDGET_MS = 4000

/** 取原生群列表和详情共用的会话标识，和权限页现有投影保持一致。 */
function listedGroupId(group) {
  return group?.id || group?.roomId || ''
}

/**
 * 仅移除有明确退群证据的条目。详情请求共享剩余时限，查不到或身份不符的群保留在本页；
 * 不改变目录顺序或上游分页游标。
 */
export async function filterListedGroups(client, groups, { signal = AbortSignal.timeout(GROUP_DETAIL_BUDGET_MS) } = {}) {
  const visible = groups.filter(group => group && group.deleted !== true && group.isInGroup !== false)
  const unresolvedIds = [...new Set(visible.filter(group => group?.isInGroup !== true)
    .map(listedGroupId).filter(isGroupPermissionId))]
  if (!unresolvedIds.length || typeof client.getGroup !== 'function') return visible

  const inactive = new Set()
  let next = 0
  const workerCount = Math.min(GROUP_DETAIL_CONCURRENCY, unresolvedIds.length)
  await Promise.all(Array.from({ length: workerCount }, async () => {
    while (!signal.aborted && next < unresolvedIds.length) {
      const id = unresolvedIds[next++]
      try {
        const detail = await client.getGroup(id, { signal })
        if (detail?.isInGroup === false && listedGroupId(detail) === id &&
            (detail.id == null || detail.id === id) && (detail.roomId == null || detail.roomId === id) &&
            (detail.accountId == null || detail.accountId === client.accountId)) {
          inactive.add(id)
        }
      } catch {
        // 上游缺失、超时或暂时不可用只代表无法判断，不据此隐藏群。
      }
    }
  }))
  return visible.filter(group => group?.isInGroup === true || !inactive.has(listedGroupId(group)))
}
