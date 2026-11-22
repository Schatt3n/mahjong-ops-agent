/** 桥接运维状态：只记录链路证据与操作者开关，不参与消息业务判断。 */
import fs from 'node:fs'
import path from 'node:path'
import { createHash, randomUUID } from 'node:crypto'

/** 固定当前进程加载的源码标识，供启动器发现未重启的旧进程。 */
export function sourceBuild(files) {
  return createHash('sha256').update(files.map(file => fs.readFileSync(file)).reduce(
    (all, bytes) => Buffer.concat([all, bytes]), Buffer.alloc(0))).digest('hex')
}

/** 原子保存开关，账号或接入版本不同时不继承已有发送授权。 */
export function outboundState(file, binding, initial) {
  let state = { ...binding, ...initial }
  try {
    const saved = JSON.parse(fs.readFileSync(file, 'utf8'))
    if (saved.account === binding.account && saved.revision === binding.revision && saved.primary === binding.primary &&
        typeof saved.send === 'boolean' && typeof saved.auto === 'boolean') state = saved
    else state = { ...binding, send: false, auto: false }
  } catch (error) {
    if (error.code !== 'ENOENT') throw new Error('outbound_state_invalid')
  }
  return {
    get value() { return { send: state.send, auto: state.auto } },
    save(nextBinding, send, auto) {
      const next = { ...nextBinding, send, auto }
      fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 })
      const temporary = file + '.' + randomUUID() + '.tmp'
      try {
        fs.writeFileSync(temporary, JSON.stringify(next), { flag: 'wx', mode: 0o600 })
        fs.renameSync(temporary, file)
      } finally { fs.rmSync(temporary, { force: true }) }
      state = next
    },
  }
}

/** 失败只暴露固定错误码和计数，不将消息或服务端响应原文写入健康接口。 */
export function listenerStatus() {
  const state = { ready: false, runtime_ready: false, native_ready: false, consecutive_failures: 0, last_error: '', last_error_at: null,
    last_poll_at: null, last_forwarded_at: null, confirmed_cursor: null }
  return {
    value: state,
    observe(kind, data = {}) {
      const now = new Date().toISOString()
      if (kind === 'error') {
        state.ready = false
        if (data.scope === 'runtime' || ['runtime_unavailable', 'runtime_version_mismatch',
          'bridge_primary_account_mismatch', 'outside_active_wechat_setup', 'bridge_authentication_required'].includes(data.code)) state.runtime_ready = false
        else state.native_ready = false
        state.consecutive_failures++
        const allowed = ['bridge_primary_account_mismatch', 'outside_active_wechat_setup', 'bridge_authentication_required',
          'runtime_version_mismatch', 'runtime_unavailable', 'native_unavailable', 'forward_failed']
        state.last_error = allowed.includes(data.code) ? data.code : 'forward_failed'
        state.last_error_at = now
      } else if (kind === 'poll') {
        Object.assign(state, { native_ready: true, ready: state.runtime_ready, last_poll_at: now,
          confirmed_cursor: data.cursor ?? state.confirmed_cursor })
        if (state.ready) Object.assign(state, { consecutive_failures: 0, last_error: '' })
      } else if (kind === 'runtime-ready') {
        state.runtime_ready = true
        state.ready = state.native_ready
        if (state.ready) Object.assign(state, { consecutive_failures: 0, last_error: '' })
      } else if (kind === 'forwarded') state.last_forwarded_at = now
    },
  }
}

/** 只读握手核对身份、协议和实际加载版本；失败不触碰消费游标或发送开关。 */
export async function verifyRuntime({ endpoint, apiToken, controlToken, primary, binding, runtimeBuild }) {
  let response
  try {
    response = await fetch(new URL('/api/channels/wechaty/preflight', endpoint), {
      headers: { Authorization: 'Bearer ' + apiToken, 'X-Mahjong-Bridge-Token': controlToken },
      redirect: 'error', signal: AbortSignal.timeout(4000),
    })
  } catch { throw Object.assign(new Error('runtime_unavailable'), { code: 'runtime_unavailable' }) }
  if (!response.ok) {
    const code = [401, 403].includes(response.status) ? 'bridge_authentication_required'
      : [404, 501].includes(response.status) ? 'runtime_version_mismatch' : 'runtime_unavailable'
    throw Object.assign(new Error(code), { code })
  }
  const status = await response.json()
  const code = status.protocol !== 1 || status.build !== runtimeBuild ? 'runtime_version_mismatch'
    : !primary || status.primary_account !== primary ? 'bridge_primary_account_mismatch'
      : status.account !== binding.agent_wechat_account || status.revision !== binding.agent_wechat_revision ? 'outside_active_wechat_setup' : ''
  if (code) throw Object.assign(new Error(code), { code })
  return status
}
