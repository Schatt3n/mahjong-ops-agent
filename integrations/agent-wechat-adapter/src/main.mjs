#!/usr/bin/env node
/**
 * agent-wechat 通道适配器入口。
 *
 * 上游 daemon 独立部署、独立升级；本进程只做 HTTP/SSE 协议翻译、可靠投递和本地
 * 外发闸门。业务状态全部留在 Python Runtime。
 */
import { loadConfig } from './config.mjs'
import { AgentWechatClient } from './client.mjs'
import { Directory } from './directory.mjs'
import { AgentWechatAdapter, createLogger } from './adapter.mjs'
import { ControlSwitches, startControlServer } from './control.mjs'

/** 解析启动参数，只接受显式的 dry-run 和帮助开关。 */
function parseArgs(argv) {
  const flags = new Set(argv.filter((item) => item.startsWith('--')))
  return { dryRun: flags.has('--dry-run'), help: flags.has('--help') || flags.has('-h') }
}

/** 打印最小启动说明，不输出 token 或账号信息。 */
function printHelp() {
  console.log(`用法: node src/main.mjs [--dry-run]

环境变量见 README.md。必填：
  MAHJONG_AGENT_WECHAT_TOKEN 或默认 token 文件 ~/.local/share/agent-wechat/api-token
  MAHJONG_AGENT_WECHAT_RAW_ENDPOINT（默认 http://127.0.0.1:8790/api/channels/agent-wechat/raw）

--dry-run 只读演练：不向 Runtime 投递、不发送任何消息。`)
}

/** 启动适配器、控制面和信号处理；异常由最外层统一记录。 */
async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.help) {
    printHelp()
    return
  }

  const config = loadConfig()
  if (args.dryRun) config.dryRun = true
  const logger = createLogger(config.logLevel)

  if (!config.upstreamToken) {
    throw new Error(
      'agent-wechat API token not found; set MAHJONG_AGENT_WECHAT_TOKEN or pass MAHJONG_AGENT_WECHAT_TOKEN_FILE',
    )
  }

  const client = new AgentWechatClient({
    baseUrl: config.upstreamBaseUrl,
    token: config.upstreamToken,
    accountId: config.accountId,
    outboundAllowlistPath: config.outboundAllowlistPath,
  })
  const directory = new Directory(client, {
    contactTtlMs: config.contactCacheTtlMs,
    threadTtlMs: config.threadCacheTtlMs,
    logger,
  })
  const switches = new ControlSwitches({
    sendChannelEnabled: config.sendEnabled,
    autoSendReply: config.autoSendReply,
  })
  const adapter = new AgentWechatAdapter({ client, directory, config, switches, logger })

  // 先确认账号并固定游标，再开放控制面，避免 /send 在账号未解析时接请求。
  await adapter.resolveAccount()
  await adapter.bootstrapCursor()
  await directory.refresh({ force: true })

  const control = await startControlServer({ config, client, directory, switches, adapter, logger })

  const controller = new AbortController()
  let closing = false
  const shutdown = async (signal) => {
    if (closing) return
    closing = true
    logger.info(`received ${signal}; stopping`)
    adapter.stopping = true
    controller.abort()
    await control.close().catch(() => {})
  }
  process.once('SIGINT', () => void shutdown('SIGINT'))
  process.once('SIGTERM', () => void shutdown('SIGTERM'))

  logger.info(
    `starting dry_run=${config.dryRun} send_enabled=${switches.sendChannelEnabled} auto_send=${switches.autoSendReply}`,
  )
  if (config.dryRun) logger.info('dry-run: inbound messages will be logged but not delivered')
  await adapter.run(controller.signal)
  await control.close().catch(() => {})
}

main().catch((error) => {
  console.error('[agent-wechat-adapter] fatal:', error?.stack || error)
  process.exitCode = 1
})
