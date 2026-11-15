#!/usr/bin/env node
/**
 * 检查上游合同是否漂移。
 *
 * 默认检查固定发布的 v2.5.0 OpenAPI；要预检新版本时传 `--url` 或先下载后用 `--file`。
 * 有破坏性变化时输出 JSON 并非零退出，供 CI 或人工升级前检查使用。
 */
import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { diffLock, PINNED_OPENAPI_URL, PINNED_SOURCE_LABEL } from './contract-lib.mjs'

const ADAPTER_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const LOCK_PATH = path.join(ADAPTER_DIR, 'contract-lock.json')

/** 解析 `--file` / `--url`，默认使用固定发布版本。 */
function parseArgs(argv) {
  const result = { file: '', url: '' }
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--file') result.file = argv[++index]
    else if (argv[index] === '--url') result.url = argv[++index]
  }
  return result
}

/** 读取上游 OpenAPI 并与锁文件比较，有漂移时非零退出。 */
async function main() {
  const args = parseArgs(process.argv.slice(2))
  const lock = JSON.parse(await fs.readFile(LOCK_PATH, 'utf8'))
  let doc
  let source
  const url = args.url || (args.file ? '' : PINNED_OPENAPI_URL)
  if (url) {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`fetch ${url} -> HTTP ${response.status}`)
    doc = await response.json()
    source = url === PINNED_OPENAPI_URL ? PINNED_SOURCE_LABEL : url
  } else {
    source = path.resolve(args.file)
    doc = JSON.parse(await fs.readFile(source, 'utf8'))
  }

  const issues = diffLock(lock, doc)
  const report = {
    ok: issues.length === 0,
    source,
    upstream_version: lock.upstream_version || '',
    locked_sha256: lock.sha256,
    issues,
  }
  console.log(JSON.stringify(report, null, 2))
  if (issues.length) process.exitCode = 1
}

main().catch((error) => {
  console.error(`contract check failed: ${error?.stack || error}`)
  process.exitCode = 2
})
