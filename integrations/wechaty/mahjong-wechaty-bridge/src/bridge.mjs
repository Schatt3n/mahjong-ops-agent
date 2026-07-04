import { WechatyBuilder } from 'wechaty'
import qrcodeTerminal from 'qrcode-terminal'

const DEFAULT_ENDPOINT = 'http://127.0.0.1:8790/api/channels/wechaty/raw'

const endpoint = process.env.MAHJONG_WECHATY_RAW_ENDPOINT || DEFAULT_ENDPOINT
const botName = process.env.MAHJONG_WECHATY_BOT_NAME || 'mahjong-wechaty-bridge'
const forwardSelfMessages = truthy(process.env.MAHJONG_WECHATY_FORWARD_SELF)

function truthy(value) {
  return ['1', 'true', 'yes', 'on'].includes(String(value || '').trim().toLowerCase())
}

function nowText() {
  const date = new Date()
  const pad = (item) => String(item).padStart(2, '0')
  return [
    date.getFullYear(),
    '-',
    pad(date.getMonth() + 1),
    '-',
    pad(date.getDate()),
    ' ',
    pad(date.getHours()),
    ':',
    pad(date.getMinutes()),
    ':',
    pad(date.getSeconds()),
  ].join('')
}

function primitive(value) {
  if (value === null || value === undefined) {
    return value
  }
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value
  }
  return String(value)
}

async function safeCall(label, fn) {
  try {
    const value = await fn()
    return primitive(value)
  } catch (error) {
    return { error: `${label}: ${error?.message || String(error)}` }
  }
}

async function buildPayload(message) {
  const room = await safeCall('room', () => message.room())
  const talker = await safeCall('talker', () => message.talker())
  const listener = await safeCall('listener', () => message.listener())
  const text = await safeCall('text', () => message.text())
  const type = await safeCall('type', () => message.type())
  const id = primitive(message.id || message.payload?.id || message.payload?.filename || '')

  let roomPayload = null
  if (room && typeof room === 'object' && !room.error) {
    roomPayload = {
      id: primitive(room.id),
      topic: await safeCall('room.topic', () => room.topic()),
    }
  }

  let talkerPayload = null
  if (talker && typeof talker === 'object' && !talker.error) {
    talkerPayload = {
      id: primitive(talker.id),
      name: await safeCall('talker.name', () => talker.name()),
      alias: await safeCall('talker.alias', () => talker.alias()),
    }
  }

  let listenerPayload = null
  if (listener && typeof listener === 'object' && !listener.error) {
    listenerPayload = {
      id: primitive(listener.id),
      name: await safeCall('listener.name', () => listener.name()),
    }
  }

  const roomId = roomPayload?.id || ''
  const senderId = talkerPayload?.id || ''
  const conversationId = roomId ? `wechaty:room:${roomId}` : `wechaty:contact:${senderId}`

  return {
    captured_at: nowText(),
    channel: 'wechaty',
    platform_name: 'wechaty',
    puppet: process.env.WECHATY_PUPPET || '',
    conversation_id: conversationId,
    message_id: id,
    source_message_id: id,
    message_type: primitive(type),
    is_room: Boolean(roomId),
    room: roomPayload,
    sender_id: senderId,
    sender_name: talkerPayload?.name || '',
    talker: talkerPayload,
    listener: listenerPayload,
    text: typeof text === 'string' ? text : '',
    raw_text: text,
    self_message: await safeCall('self', () => message.self()),
    payload: message.payload || null,
  }
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
    body: JSON.stringify(payload),
  })
  const body = await response.text()
  let parsed = null
  try {
    parsed = JSON.parse(body)
  } catch {
    parsed = { raw_response: body }
  }
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${body}`)
  }
  return parsed
}

const bot = WechatyBuilder.build({ name: botName })

bot.on('scan', (qrcode, status) => {
  console.log(`[${nowText()}] scan status=${status}`)
  console.log(`https://wechaty.js.org/qrcode/${encodeURIComponent(qrcode)}`)
  qrcodeTerminal.generate(qrcode, { small: true })
})

bot.on('login', (user) => {
  console.log(`[${nowText()}] login: ${user}`)
})

bot.on('logout', (user) => {
  console.log(`[${nowText()}] logout: ${user}`)
})

bot.on('error', (error) => {
  console.error(`[${nowText()}] error:`, error)
})

bot.on('message', async (message) => {
  if (!forwardSelfMessages && message.self()) {
    return
  }
  const payload = await buildPayload(message)
  try {
    const result = await postJson(endpoint, payload)
    console.log(
      `[${nowText()}] forwarded message_id=${payload.message_id || '-'} ` +
        `conversation_id=${payload.conversation_id} trace_id=${result.trace_id || '-'}`
    )
  } catch (error) {
    console.error(`[${nowText()}] forward failed: ${error?.message || String(error)}`)
  }
})

process.once('SIGINT', async () => {
  console.log(`[${nowText()}] stopping...`)
  await bot.stop()
  process.exit(0)
})

console.log(`[${nowText()}] starting ${botName}`)
console.log(`[${nowText()}] endpoint=${endpoint}`)
console.log(`[${nowText()}] WECHATY_PUPPET=${process.env.WECHATY_PUPPET || '(default)'}`)

await bot.start()
