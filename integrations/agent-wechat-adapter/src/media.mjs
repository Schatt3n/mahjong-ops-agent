/** 核实 Hub 的图片和表情包与原消息、附件的绑定后，才读取媒体字节。 */
import { setTimeout as pause } from 'node:timers/promises'
import { UpstreamError } from './client.mjs'

const IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/gif', 'image/webp'])
const PREVIEW_VARIANTS = ['standard', 'original', 'thumbnail']
const TERMINAL_MEDIA_STATES = new Set(['recalled', 'expired', 'unsupported'])
const ACTIVE_JOB_STATES = new Set(['queued', 'running'])
const MAX_ORIGINAL_BYTES = 32 * 1024 * 1024
const MAX_PREVIEW_BYTES = 8 * 1024 * 1024

/** 拼接 Hub 路由前拒绝标识中的控制字符及路径别名，避免改变目标资源。 */
function validId(value) {
  return typeof value === 'string' && value.length > 0 && value.length <= 4096 &&
    value === value.trim() && value !== '.' && value !== '..' && !/\p{Cc}/u.test(value)
}

/** 从 Hub 消息类型识别支持的图片或表情包，只投影类别用于历史展示及下载。 */
export function mediaKindOf(message) {
  if (!message || message.recall) return null
  if (message.type === 3) return 'image'
  if (message.type === 47 || (message.type === 49 && message.appMessageType === 8)) return 'sticker'
  return null
}

/** 核对附件的原消息属于指定账号和会话，并拒绝已撤回或不包含受支持媒体的消息。 */
async function requireMessage(client, accountId, threadId, messageId, signal) {
  const message = await client.getMessage(messageId, { signal, redirect: 'error' })
  if (message?.accountId !== accountId || message?.threadId !== threadId || message?.id !== messageId) {
    throw new UpstreamError('media source was not found', { status: 404, code: 'media_source_mismatch' })
  }
  if (message.recall) {
    throw new UpstreamError('message has been recalled', { status: 410, code: 'message_recalled' })
  }
  const kind = mediaKindOf(message)
  if (kind) return kind
  throw new UpstreamError('message does not contain supported visual media', { status: 404, code: 'unsupported_media' })
}

/** 预览按 Hub 实际清晰度选择可展示的附件，原件入口只接受已就绪的原始附件。 */
function selectAvailable(assets, representation, maxBytes) {
  if (representation === 'original') {
    return assets.find(asset => asset.state === 'available' && asset.actualVariant === 'original')
  }
  const ready = PREVIEW_VARIANTS.flatMap(variant => assets.filter(asset => asset.state === 'available' &&
    asset.actualVariant === variant && (IMAGE_TYPES.has(asset.contentType) ||
      (asset.contentType === 'image/x-wxgf' && IMAGE_TYPES.has(asset.preview?.contentType)))))
  return ready.find(asset => (asset.contentType === 'image/x-wxgf' ? asset.preview.byteLength : asset.byteLength) <= maxBytes) ?? ready[0]
}

/** 区分撤回、过期和不可用状态，缺少原件时明确报错以免把缩略图当成原件。 */
function unavailable(assets, representation) {
  if (assets.some(asset => asset.state === 'recalled')) {
    return new UpstreamError('message has been recalled', { status: 410, code: 'message_recalled' })
  }
  if (assets.some(asset => asset.state === 'expired')) {
    return new UpstreamError('media has expired', { status: 410, code: 'media_expired' })
  }
  return new UpstreamError('requested media is unavailable', {
    status: 404, code: representation === 'original' ? 'original_unavailable' : 'media_unavailable',
  })
}

/** 尽量复用同一附件的 Hub 下载任务，校验任务绑定并用统一时限约束轮询。 */
async function downloadAsset(client, accountId, messageId, candidate, signal) {
  let job = candidate.jobId && validId(candidate.jobId)
    ? await client.mediaJob(candidate.jobId, { signal })
    : await client.transferMedia(messageId, candidate.id, candidate.variant, { signal })
  for (;;) {
    if (job?.accountId !== accountId || job?.assetId !== candidate.id || job?.kind !== 'download' || !validId(job.id)) {
      throw new UpstreamError('media download task does not match', { status: 502, code: 'media_source_mismatch' })
    }
    if (!ACTIVE_JOB_STATES.has(job.state)) break
    await pause(300, undefined, { signal })
    job = await client.mediaJob(job.id, { signal })
  }
}

/** 通过 Hub 的附件接口读取与消息绑定的图片或表情包，拒绝解析并访问附件内的 URL。 */
export async function readObservedMedia(client, { accountId, threadId, messageId, representation, signal = null }) {
  if (![accountId, threadId, messageId].every(validId) || !['preview', 'original'].includes(representation)) {
    throw new UpstreamError('exact media reference is required', { status: 400, code: 'invalid_media_reference' })
  }
  if (client.accountId !== accountId) {
    throw new UpstreamError('media source was not found', { status: 404, code: 'media_source_mismatch' })
  }
  const timeout = AbortSignal.timeout(15_000)
  const deadline = signal ? AbortSignal.any([signal, timeout]) : timeout
  const scoped = { signal: deadline }
  const maxBytes = representation === 'original' ? MAX_ORIGINAL_BYTES : MAX_PREVIEW_BYTES
  const kind = await requireMessage(client, accountId, threadId, messageId, deadline)

  /** 每次读取附件列表后先验证其账号、消息和附件标识，再按媒体类型选择。 */
  const list = async () => {
    const result = await client.mediaAssets(messageId, scoped)
    if (!Array.isArray(result?.items) || result.items.some(asset =>
      asset?.accountId !== accountId || asset?.messageId !== messageId || !validId(asset.id))) {
      throw new UpstreamError('media assets do not match their message', { status: 502, code: 'media_source_mismatch' })
    }
    return result.items.filter(asset => asset.kind === kind)
  }

  let assets = await list()
  let available = selectAvailable(assets, representation, maxBytes)
  if (!available) {
    const variants = representation === 'original' ? ['original'] : PREVIEW_VARIANTS
    const candidates = variants.flatMap(variant => assets.filter(asset => asset.variant === variant &&
      !TERMINAL_MEDIA_STATES.has(asset.state) && asset.state !== 'available'))
    const candidate = candidates.find(asset => !(asset.advertisedByteLength > maxBytes))
    if (!candidate && candidates.length) {
      throw new UpstreamError('media exceeds the byte limit', { status: 413, code: 'media_too_large' })
    }
    if (!candidate) throw unavailable(assets.filter(asset => variants.includes(asset.variant)), representation)
    try {
      await downloadAsset(client, accountId, messageId, candidate, deadline)
    } catch (error) {
      // 本次等待超时后，重试可继续查询同一个 Hub 下载任务。
      if (timeout.aborted && !signal?.aborted) {
        throw new UpstreamError('media download is still in progress', { status: 503, code: 'media_pending' })
      }
      throw error
    }
    assets = await list()
    available = selectAvailable(assets, representation, maxBytes)
    if (!available) throw unavailable(assets.filter(asset => variants.includes(asset.variant)), representation)
  }

  // WXGF 预览需核对原件摘要，下载时仍返回微信原始格式。
  const derived = representation === 'preview' && available.contentType === 'image/x-wxgf'
  const expected = derived ? available.preview : available
  const supported = IMAGE_TYPES.has(expected?.contentType) || (!derived && expected?.contentType === 'image/x-wxgf')
  if (!supported) {
    throw new UpstreamError('media has no supported representation', { status: 404, code: 'media_preview_unavailable' })
  }
  if (!/^[a-f0-9]{64}$/.test(expected.sha256 || '') || !Number.isSafeInteger(expected.byteLength) || expected.byteLength <= 0 ||
      (derived && expected.sourceSha256 !== available.sha256)) {
    throw new UpstreamError('media metadata could not be verified', { status: 502, code: 'media_integrity_failed' })
  }
  if (expected.byteLength > maxBytes) {
    throw new UpstreamError('media exceeds the byte limit', { status: 413, code: 'media_too_large' })
  }
  const encoding = derived ? 'preview' : 'original'
  const bytes = await client.mediaContent(available.id, expected, { representation: encoding, maxBytes, signal: deadline })
  await requireMessage(client, accountId, threadId, messageId, deadline)
  return { bytes, contentType: expected.contentType, variant: available.actualVariant, representation: encoding }
}
