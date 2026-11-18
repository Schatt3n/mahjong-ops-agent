#!/usr/bin/env node
/** 生成 contract-lock.json。用法：node scripts/write-contract-lock.mjs [--file path|--url url] */
import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  buildLock,
  PINNED_OPENAPI_URL,
  PINNED_SOURCE_LABEL,
  PINNED_UPSTREAM_COMMIT,
  PINNED_UPSTREAM_VERSION,
} from './contract-lib.mjs'

const ADAPTER_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const REPO_ROOT = path.resolve(ADAPTER_DIR, '../..')

/** 解析输入与稳定的来源标签；默认读取固定发布的 v2.5.0 OpenAPI。 */
function parseArgs(argv) {
  const result = { file: '', url: '', sourceLabel: '', version: '', commit: '' }
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--file') result.file = argv[++index]
    else if (argv[index] === '--url') result.url = argv[++index]
    else if (argv[index] === '--source-label') result.sourceLabel = argv[++index]
    else if (argv[index] === '--version') result.version = argv[++index]
    else if (argv[index] === '--commit') result.commit = argv[++index]
  }
  return result
}

/** 把本地文件路径转成仓库相对标签，避免锁文件写入本机绝对路径。 */
function stableFileLabel(file) {
  const absolute = path.resolve(file)
  const relative = path.relative(REPO_ROOT, absolute)
  return !relative.startsWith('..') && !path.isAbsolute(relative) ? relative : path.basename(absolute)
}

/** 读取 URL 或本地 OpenAPI，并返回稳定的来源标签。 */
async function loadDocument(args) {
  const url = args.url || (args.file ? '' : PINNED_OPENAPI_URL)
  if (url) {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`fetch ${url} -> HTTP ${response.status}`)
    const source = args.sourceLabel || (url === PINNED_OPENAPI_URL ? PINNED_SOURCE_LABEL : url)
    return { doc: await response.json(), source, pinned: url === PINNED_OPENAPI_URL }
  }
  const absolute = path.resolve(args.file)
  const source = args.sourceLabel || stableFileLabel(absolute)
  return { doc: JSON.parse(await fs.readFile(absolute, 'utf8')), source, pinned: false }
}

const args = parseArgs(process.argv.slice(2))
const { doc, source, pinned } = await loadDocument(args)
const lock = buildLock(doc, source, {
  upstreamVersion: args.version || (pinned ? PINNED_UPSTREAM_VERSION : ''),
  upstreamCommit: args.commit || (pinned ? PINNED_UPSTREAM_COMMIT : ''),
})
await fs.writeFile(path.join(ADAPTER_DIR, 'contract-lock.json'), `${JSON.stringify(lock, null, 2)}\n`)
console.log(`wrote contract-lock.json from ${source}`)
console.log(`paths=${Object.keys(lock.paths).length} schemas=${Object.keys(lock.schemas).length} states=${lock.send_states.join(',')}`)
