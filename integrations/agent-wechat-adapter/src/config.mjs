/**
 * 适配器配置。
 *
 * 本进程只做协议翻译和可靠投递，不持有业务状态：会话、画像、记忆都在 Python
 * Runtime 一侧。所有可能改变线上行为的开关默认关闭，避免部署即外发。
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ADAPTER_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
export const REPO_ROOT = path.resolve(ADAPTER_DIR, '../..')

/** 读取布尔环境变量，无法识别时返回默认值而不是抛出。 */
export function envBool(name, fallback = false) {
  const raw = process.env[name]
  if (raw === undefined || raw === '') return fallback
  return ['1', 'true', 'yes', 'on'].includes(String(raw).trim().toLowerCase())
}

/** 读取整数环境变量，非法值回退默认，避免启动时因拼写错误中断。 */
function envInt(name, fallback) {
  const value = Number.parseInt(process.env[name] || '', 10)
  return Number.isFinite(value) ? value : fallback
}

/** 读取带 fallback 的 token 文件，文件缺失时返回空串。 */
function readTokenFile(candidate) {
  if (!candidate) return ''
  try {
    return fs.readFileSync(candidate, 'utf8').trim()
  } catch {
    return ''
  }
}

const defaultAgentTokenPath = path.join(REPO_ROOT, 'data', 'runtime_api_token')
const defaultStateDir = path.join(REPO_ROOT, 'data', 'agent_wechat_adapter')

/**
 * 解析运行配置。
 *
 * `accountId` 允许留空：启动时会调用 /v1/accounts 取唯一账号，多于一个则报错，
 * 避免静默选中错误的微信号。
 */
export function loadConfig(env = process.env) {
  const dataHome = env.XDG_DATA_HOME && path.isAbsolute(env.XDG_DATA_HOME)
    ? env.XDG_DATA_HOME
    : path.join(os.homedir(), '.local', 'share')

  const upstreamBaseUrl = (env.MAHJONG_AGENT_WECHAT_BASE_URL || 'http://127.0.0.1:6174').replace(/\/+$/, '')
  const upstreamToken =
    (env.MAHJONG_AGENT_WECHAT_TOKEN || '').trim() ||
    readTokenFile(env.MAHJONG_AGENT_WECHAT_TOKEN_FILE || path.join(dataHome, 'agent-wechat', 'api-token'))

  const agentToken =
    (env.MAHJONG_AGENT_WECHAT_AGENT_TOKEN || '').trim() ||
    (env.MAHJONG_AGENT_API_TOKEN || '').trim() ||
    readTokenFile(env.MAHJONG_AGENT_API_TOKEN_PATH || defaultAgentTokenPath)

  const stateDir = env.MAHJONG_AGENT_WECHAT_STATE_DIR || defaultStateDir

  return {
    upstreamBaseUrl,
    upstreamToken,
    accountId: (env.MAHJONG_AGENT_WECHAT_ACCOUNT || '').trim(),
    agentEndpoint: (env.MAHJONG_AGENT_WECHAT_RAW_ENDPOINT || 'http://127.0.0.1:8790/api/channels/agent-wechat/raw').trim(),
    agentToken,
    stateDir,
    cursorPath: path.join(stateDir, 'events_cursor.json'),
    deliveryLedgerPath: path.join(stateDir, 'delivery_ledger.jsonl'),
    spoolDir: path.join(stateDir, 'ingress_spool'),
    controlHost: env.MAHJONG_AGENT_WECHAT_CONTROL_HOST || '127.0.0.1',
    controlPort: envInt('MAHJONG_AGENT_WECHAT_CONTROL_PORT', 8792),
    controlToken: (env.MAHJONG_AGENT_WECHAT_CONTROL_TOKEN || '').trim(),
    sendEnabled: envBool('MAHJONG_AGENT_WECHAT_SEND_ENABLED', false),
    autoSendReply: envBool('MAHJONG_AGENT_WECHAT_AUTO_SEND_REPLY', false),
    forwardSelfMessages: envBool('MAHJONG_AGENT_WECHAT_FORWARD_SELF', false),
    forwardSystemMessages: envBool('MAHJONG_AGENT_WECHAT_FORWARD_SYSTEM', false),
    replayHistory: envBool('MAHJONG_AGENT_WECHAT_REPLAY_HISTORY', false),
    dryRun: envBool('MAHJONG_AGENT_WECHAT_DRY_RUN', false),
    requestBodyLimitBytes: envInt('MAHJONG_AGENT_WECHAT_CONTROL_BODY_LIMIT_BYTES', 262_144),
    startCursor: (env.MAHJONG_AGENT_WECHAT_START_CURSOR || '').trim(),
    contactCacheTtlMs: envInt('MAHJONG_AGENT_WECHAT_CONTACT_CACHE_TTL_SECONDS', 600) * 1000,
    threadCacheTtlMs: envInt('MAHJONG_AGENT_WECHAT_THREAD_CACHE_TTL_SECONDS', 300) * 1000,
    sendWaitTimeoutMs: envInt('MAHJONG_AGENT_WECHAT_SEND_WAIT_SECONDS', 60) * 1000,
    logLevel: (env.MAHJONG_AGENT_WECHAT_LOG_LEVEL || 'info').toLowerCase(),
  }
}
