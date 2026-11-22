import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import http from 'node:http'
import test from 'node:test'
import { outboundState, listenerStatus, verifyRuntime } from './src/bridge-operations.mjs'

test('普通重启保留操作者开关，换账号或接入版本不能继承发送授权', t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bridge-state-'))
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const file = path.join(dir, 'state.json')
  const binding = { primary: 'primary', account: 'active', revision: 'one' }
  const first = outboundState(file, binding, { send: false, auto: false })
  first.save(binding, true, true)
  assert.deepEqual(outboundState(file, binding, { send: false, auto: false }).value, { send: true, auto: true })
  first.save(binding, true, false)
  assert.deepEqual(outboundState(file, binding, {}).value, { send: true, auto: false })
  for (const changed of [{account:'other'}, {primary:'other'}, {revision:'two'}]) {
    assert.deepEqual(outboundState(file, {...binding,...changed}, {send:true,auto:true}).value, {send:false,auto:false})
  }
  fs.writeFileSync(file, '{broken')
  assert.throws(() => outboundState(file, binding, {}), /outbound_state_invalid/)
})

test('写入失败不得修改内存授权，链路错误不得被成功轮询掩盖', t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bridge-write-'))
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const state = outboundState(path.join(dir,'missing','state'), {}, {send:false,auto:false})
  fs.writeFileSync(path.join(dir,'missing'), 'blocked')
  assert.throws(() => state.save({},true,true))
  assert.deepEqual(state.value,{send:false,auto:false})
  const status = listenerStatus()
  status.observe('runtime-ready'); status.observe('poll',{cursor:'10'})
  assert.equal(status.value.ready,true)
  status.observe('error',{code:'bridge_primary_account_mismatch'})
  status.observe('poll',{cursor:'10'})
  assert.equal(status.value.ready,false)
  assert.equal(status.value.last_error,'bridge_primary_account_mismatch')
  assert.equal(status.value.confirmed_cursor,'10')
  status.observe('runtime-ready')
  assert.equal(status.value.ready,true)
  assert.equal(status.value.last_error,'')
})

test('只读握手拒绝旧版本、账号差异和旧认证，不发业务消息', async t => {
  let result = {protocol:1,build:'loaded',primary_account:'primary',account:'active',revision:'one'}
  let statusCode = 200
  const paths=[]
  const server=http.createServer((request,response) => {
    paths.push(request.method+' '+request.url)
    response.writeHead(statusCode,{'Content-Type':'application/json'}).end(JSON.stringify(result))
  })
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve))
  t.after(()=>new Promise(resolve=>server.close(resolve)))
  const args = { endpoint:`http://127.0.0.1:${server.address().port}`,apiToken:'fixture',controlToken:'fixture',
    primary:'primary',binding:{agent_wechat_account:'active',agent_wechat_revision:'one'},runtimeBuild:'loaded' }
  await verifyRuntime(args)
  for (const [field,value,code] of [['build','old','runtime_version_mismatch'],
    ['primary_account','wrong','bridge_primary_account_mismatch'],['revision','old','outside_active_wechat_setup']]) {
    const original=result[field];result[field]=value
    await assert.rejects(verifyRuntime(args),{code});result[field]=original
  }
  statusCode=403
  await assert.rejects(verifyRuntime(args),{code:'bridge_authentication_required'})
  assert.ok(paths.every(route=>route==='GET /api/channels/wechaty/preflight'))
})
