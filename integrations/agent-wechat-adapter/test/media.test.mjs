import test from 'node:test'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { AgentWechatClient } from '../src/client.mjs'
import { readObservedMedia } from '../src/media.mjs'

const scope = { accountId: 'acc', threadId: 'wxid_test', messageId: 'sticker-1' }
const sticker = { accountId: scope.accountId, threadId: scope.threadId, id: scope.messageId, type: 49, appMessageType: 8 }
const bytes = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aFfsAAAAASUVORK5CYII=', 'base64')
const hash = createHash('sha256').update(bytes).digest('hex')
const originalHash = 'a'.repeat(64)
const original = { id: 'asset-original', messageId: scope.messageId, accountId: scope.accountId, kind: 'sticker',
  variant: 'original', state: 'available', actualVariant: 'original', contentType: 'image/x-wxgf',
  sha256: originalHash, byteLength: 100, preview: { contentType: 'image/png', byteLength: bytes.length,
    sha256: hash, sourceSha256: originalHash } }

test('derived preview is bound to the native original digest', async () => {
  const seen = []
  const client = new AgentWechatClient({ baseUrl: 'http://hub.test', accountId: scope.accountId,
    fetchImpl: async (url) => {
      const parsed = new URL(url)
      seen.push(`${parsed.pathname}${parsed.search}`)
      if (parsed.pathname.endsWith('/messages/sticker-1/media')) return Response.json({ items: [original] })
      if (parsed.pathname.endsWith('/messages/sticker-1')) return Response.json(sticker)
      assert.equal(parsed.search, '?representation=preview')
      return new Response(bytes, { headers: { 'content-type': 'image/png', 'x-source-sha256': originalHash } })
    },
  })
  const read = await readObservedMedia(client, { ...scope, representation: 'preview' })
  assert.deepEqual(read, { bytes, contentType: 'image/png', variant: 'original', representation: 'preview' })
  assert.deepEqual(seen, ['/v1/accounts/acc/messages/sticker-1', '/v1/accounts/acc/messages/sticker-1/media',
    '/v1/accounts/acc/media/asset-original/content?representation=preview', '/v1/accounts/acc/messages/sticker-1'])
})

test('thumbnail fetch never masquerades as an original and is reused', async () => {
  let assets = [{ ...original, id: 'thumb', variant: 'thumbnail', actualVariant: null, state: 'pending', preview: null }]
  const calls = []
  const client = { accountId: scope.accountId, getMessage: async () => sticker,
    mediaAssets: async () => ({ items: assets }),
    transferMedia: async (messageId, assetId, quality) => {
      calls.push({ messageId, assetId, quality })
      assets = [{ ...assets[0], actualVariant: 'thumbnail', state: 'available', contentType: 'image/png',
        sha256: hash, byteLength: bytes.length }]
      return { id: 'job', accountId: scope.accountId, assetId, kind: 'download', state: 'available' }
    },
    mediaContent: async () => bytes,
  }
  await assert.rejects(readObservedMedia(client, { ...scope, representation: 'original' }), {
    status: 404, code: 'original_unavailable',
  })
  const preview = await readObservedMedia(client, { ...scope, representation: 'preview' })
  assert.equal(preview.variant, 'thumbnail')
  await readObservedMedia(client, { ...scope, representation: 'preview' })
  assert.deepEqual(calls, [{ messageId: scope.messageId, assetId: 'thumb', quality: 'thumbnail' }])
})

test('cached thumbnail remains usable when WXGF has no derived preview', async () => {
  const thumb = { ...original, id: 'thumb', variant: 'thumbnail', actualVariant: 'thumbnail', preview: null,
    contentType: 'image/png', sha256: hash, byteLength: bytes.length }
  const client = { accountId: scope.accountId, getMessage: async () => sticker,
    mediaAssets: async () => ({ items: [{ ...original, preview: null }, thumb] }),
    mediaContent: async (id) => { assert.equal(id, thumb.id); return bytes },
  }
  const preview = await readObservedMedia(client, { ...scope, representation: 'preview' })
  assert.equal(preview.variant, 'thumbnail')
})

test('mismatched attachment or later recall prevents serving bytes', async () => {
  const client = { accountId: scope.accountId, getMessage: async () => sticker,
    mediaAssets: async () => ({ items: [{ ...original, messageId: 'unrelated' }] }),
    mediaContent: async () => assert.fail('must not request mismatched asset'),
  }
  await assert.rejects(readObservedMedia(client, { ...scope, representation: 'preview' }), { code: 'media_source_mismatch' })
  let readContent = false
  await assert.rejects(readObservedMedia({ ...client,
    getMessage: async () => readContent ? { ...sticker, recall: { recalledAt: 'known-time' } } : sticker,
    mediaAssets: async () => ({ items: [original] }),
    mediaContent: async () => { readContent = true; return bytes },
  }, { ...scope, representation: 'preview' }), { status: 410, code: 'message_recalled' })
})

test('rejects digest mismatches and oversized streams', async () => {
  const client = new AgentWechatClient({ baseUrl: 'http://hub.test', accountId: scope.accountId,
    fetchImpl: async () => new Response(Uint8Array.from([1, 2, 3]), { headers: { 'content-type': 'image/png' } }),
  })
  const expected = { sha256: hash, byteLength: 3, contentType: 'image/png' }
  await assert.rejects(client.mediaContent('asset', expected, { representation: 'original', maxBytes: 3 }), {
    status: 502, code: 'media_integrity_failed',
  })
  await assert.rejects(client.mediaContent('asset', expected, { representation: 'original', maxBytes: 2 }), {
    status: 413, code: 'media_too_large',
  })
})
