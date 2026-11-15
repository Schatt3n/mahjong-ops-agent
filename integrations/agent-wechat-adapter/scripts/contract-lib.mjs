/**
 * 上游合同锁定与漂移检测。
 *
 * 只关心适配器真正依赖的路由、字段和发送状态，不复制整份 OpenAPI。原作者新增
 * 路由不会报警；删改我们依赖的部分会非零退出。
 */
import { createHash } from 'node:crypto'

/** 适配器当前对齐并验证过的上游发布版本。 */
export const PINNED_UPSTREAM_VERSION = 'v2.5.0'
export const PINNED_UPSTREAM_COMMIT = 'ce5c0687c0e99e3f93fd85266baa3fad26d63a11'
export const PINNED_OPENAPI_URL =
  `https://raw.githubusercontent.com/thisnick/agent-wechat/${PINNED_UPSTREAM_VERSION}/interfaces/http/openapi.json`
export const PINNED_SOURCE_LABEL =
  `thisnick/agent-wechat@${PINNED_UPSTREAM_VERSION} (${PINNED_UPSTREAM_COMMIT.slice(0, 7)}) interfaces/http/openapi.json`

/** 适配器依赖的路由与方法。 */
export const CRITICAL_PATHS = {
  '/v1/accounts': ['get'],
  '/v1/accounts/{a}/messages/{id}': ['get'],
  '/v1/accounts/{a}/events': ['get'],
  '/v1/accounts/{a}/events/stream': ['get'],
  '/v1/accounts/{a}/sends': ['post'],
  '/v1/accounts/{a}/sends/{id}': ['get'],
  '/v1/accounts/{a}/sends/by-idempotency-key': ['get'],
  '/v1/accounts/{a}/contacts': ['get'],
  '/v1/accounts/{a}/groups': ['get'],
  '/v1/accounts/{a}/threads': ['get'],
  '/v1/accounts/{a}/threads/{id}': ['get'],
}

/** 适配器读取的 schema 字段。`required` 变化视为破坏性漂移。 */
export const CRITICAL_SCHEMAS = {
  NativeMessage: ['id', 'threadId', 'senderId', 'body', 'type', 'appMessageType', 'direction', 'createdAt', 'serverId', 'mentions', 'quote', 'media'],
  NativeEvent: ['kind', 'data', 'cursor', 'createdAt'],
  NativeSendIntent: ['id', 'state', 'localMessageId', 'serverId', 'errorCode'],
  NativeContact: ['id', 'username', 'displayName', 'nickname', 'remark', 'alias'],
  NativeThread: ['id', 'kind', 'displayName'],
  NativeTextSendRequest: ['threadId', 'text'],
}

/** 从 state 描述里抽取发送状态集合，上游新增/删除状态都会被发现。 */
export function sendStatesOf(doc) {
  const description = doc?.components?.schemas?.NativeSendIntent?.properties?.state?.description || ''
  const states = new Set()
  for (const match of description.matchAll(/`([a-z_]+)`/g)) states.add(match[1])
  return [...states].sort()
}

/** 生成可提交的合同锁。 */
export function buildLock(doc, source, { upstreamVersion = '', upstreamCommit = '' } = {}) {
  const paths = {}
  for (const [pathname, methods] of Object.entries(CRITICAL_PATHS)) {
    paths[pathname] = {}
    for (const method of methods) {
      const operation = doc?.paths?.[pathname]?.[method] || null
      paths[pathname][method] = operation
        ? {
            operationId: operation.operationId || '',
            idempotencyHeader:
              (operation.parameters || []).some(
                (parameter) => parameter?.in === 'header' && parameter?.name === 'Idempotency-Key',
              ),
          }
        : null
    }
  }

  const schemas = {}
  for (const [name, fields] of Object.entries(CRITICAL_SCHEMAS)) {
    const schema = doc?.components?.schemas?.[name] || null
    schemas[name] = schema
      ? {
          fields,
          required: [...(schema.required || [])].sort(),
          properties: Object.keys(schema.properties || {}).sort(),
        }
      : null
  }

  return {
    generated_at: new Date().toISOString(),
    source,
    upstream_version: upstreamVersion,
    upstream_commit: upstreamCommit,
    sha256: createHash('sha256').update(JSON.stringify(doc)).digest('hex'),
    openapi_info: doc?.info || {},
    paths,
    schemas,
    send_states: sendStatesOf(doc),
  }
}

/** 比较当前文档和锁，返回人类可读的漂移列表。 */
export function diffLock(lock, doc) {
  const issues = []
  const add = (code, message) => issues.push({ code, message })

  for (const [pathname, methods] of Object.entries(CRITICAL_PATHS)) {
    for (const method of methods) {
      const operation = doc?.paths?.[pathname]?.[method]
      const locked = lock?.paths?.[pathname]?.[method]
      if (!operation) add('path_missing', `${method.toUpperCase()} ${pathname} 不存在`)
      else if (!locked) add('lock_missing_path', `锁文件缺少 ${method.toUpperCase()} ${pathname}`)
      else if (method === 'post' && locked.idempotencyHeader) {
        const hasHeader = (operation.parameters || []).some(
          (parameter) => parameter?.in === 'header' && parameter?.name === 'Idempotency-Key',
        )
        if (!hasHeader) add('idempotency_header_missing', `${method.toUpperCase()} ${pathname} 不再要求 Idempotency-Key`)
      }
    }
  }

  for (const [name, fields] of Object.entries(CRITICAL_SCHEMAS)) {
    const schema = doc?.components?.schemas?.[name]
    const locked = lock?.schemas?.[name]
    if (!schema) {
      add('schema_missing', `schema ${name} 不存在`)
      continue
    }
    if (!locked) {
      add('lock_missing_schema', `锁文件缺少 schema ${name}`)
      continue
    }
    const required = [...(schema.required || [])].sort()
    if (JSON.stringify(required) !== JSON.stringify(locked.required)) {
      add('required_changed', `${name}.required 变化: ${locked.required.join(',')} -> ${required.join(',')}`)
    }
    const properties = new Set(Object.keys(schema.properties || {}))
    const missing = fields.filter((field) => !properties.has(field))
    if (missing.length) add('field_missing', `${name} 缺少字段: ${missing.join(', ')}`)
  }

  const states = sendStatesOf(doc)
  if (JSON.stringify(states) !== JSON.stringify(lock?.send_states || [])) {
    add('send_states_changed', `发送状态变化: ${(lock?.send_states || []).join(',')} -> ${states.join(',')}`)
  }

  return issues
}
