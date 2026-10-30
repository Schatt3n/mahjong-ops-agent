"""直接执行生产发送函数，验证 HTTP 首次响应的逆序和重试隔离。"""

from pathlib import Path
import json
import re
import shutil
import subprocess

import pytest


def _production_functions() -> str:
    """只替换 DOM 与网络依赖，发送和结果渲染均取自实际页面源码。"""
    source = (Path(__file__).resolve().parents[1] / "scripts" / "agent_runtime_app.py").read_text()
    functions = []
    for name in ("renderMessageResult", "sendMessage"):
        match = re.search(
            rf"^(?:async )?function {name}\([^\n]*\)\{{\n.*?^\}}",
            source,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match, f"生产页面缺少 {name}"
        functions.append(match.group())
    return "\n".join(functions)


HARNESS = r"""
let latestResult = null, pendingMessageAttempt = null, messagePollGeneration = 0;
const window = {lastTraceId:'initial-trace'};
const output = {textContent:'initial-output'}, wechatText = {value:'initial-draft'};
const conversationId = {value:'conversation-a'}, senderId = {value:'customer-a'};
const senderName = {value:'客户甲'}, text = {value:'第一条'};
const requests = [], polls = [];
let generatedIds = 0, stateRefreshes = 0, liveRefreshes = 0;
const createMessageId = () => `message-${++generatedIds}`;
const loadState = async () => {stateRefreshes++;};
const refreshLive = async () => {liveRefreshes++;};
const pollInputBatch = async (batch, generation) => {polls.push({batch,generation});};
function deferred(){
  let resolve, reject;
  const promise = new Promise((accept, refuse) => {resolve=accept;reject=refuse;});
  return {promise, resolve, reject};
}
function fetch(url, options){
  assert.equal(url, '/api/message', '测试禁止真实网络请求');
  assert.equal(options.method, 'POST');
  const response = deferred(), json = deferred(), jsonStarted = deferred();
  const request = {response,json,jsonStarted,body:JSON.parse(options.body),jsonReads:0};
  requests.push(request);
  return response.promise;
}
function respond(index, {ok=true, body, waitForJson=false}={}){
  const request=requests[index];
  request.response.resolve({ok, status:ok?200:503, json:()=>{
    request.jsonReads++;
    request.jsonStarted.resolve();
    return request.json.promise;
  }});
  if(!waitForJson) request.json.resolve(body);
}
function result(label){
  return {trace_id:`trace-${label}`, final_reply:`reply-${label}`,
          poll_batch_id:`batch-${label}`, background_processing:true};
}
function visibleState(){
  return {latestResult, output:output.textContent, trace:window.lastTraceId,
          draft:wechatText.value, stateRefreshes, liveRefreshes, polls:[...polls]};
}
function assertVisible(label){
  assert.deepEqual(latestResult,result(label));
  assert.deepEqual(JSON.parse(output.textContent),result(label));
  assert.equal(window.lastTraceId,`trace-${label}`);
  assert.equal(wechatText.value,`reply-${label}`);
}
"""


def _run_case(case: str) -> None:
    """等待异步用例明确完成，避免 Node 提前退出被误算为通过。"""
    node = shutil.which("node")
    assert node, "API UI 并发验证需要现有 Node 运行时"
    program = HARNESS + "\n" + _production_functions() + "\n(async()=>{\n" + case + "\n})()"
    launcher = (
        "const vm=require('node:vm'), assert=require('node:assert/strict');\n"
        f"const source={json.dumps(program)};\n"
        "Promise.resolve(vm.runInNewContext(source,{assert},{timeout:1000}))"
        ".then(()=>process.stdout.write('ok\\n'))"
        ".catch(error=>{console.error(error.stack);process.exitCode=1;});\n"
    )
    completed = subprocess.run([node, "-"], input=launcher, text=True, capture_output=True, timeout=10)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout == "ok\n", "异步场景未完成，可能仍有未解决的 Promise"


@pytest.mark.parametrize("stage", ["fetch", "json"])
@pytest.mark.parametrize("old_ok", [True, False])
def test_late_initial_response_cannot_replace_latest_result(stage: str, old_ok: bool) -> None:
    """新输入先完成时，旧成功或 HTTP 错误都不能回退回复、trace 和草稿。"""
    _run_case("const stage=" + json.dumps(stage) + ";const oldOk=" + json.dumps(old_ok) + ";\n" + r"""
 const first=sendMessage();
 if(stage==='json'){
   respond(0,{ok:oldOk,waitForJson:true});
   await requests[0].jsonStarted.promise;
 }
 text.value='第二条';
 const second=sendMessage();
 respond(1,{body:result('new')});
 await second;
 assertVisible('new');
 const current=visibleState();
 if(stage==='fetch') respond(0,{ok:oldOk,body:result('old')});
 else requests[0].json.resolve(result('old'));
 await first;
 assert.deepEqual(visibleState(),current);
 assert.equal(pendingMessageAttempt,null);
 assert.deepEqual(polls,[{batch:'batch-new',generation:2}]);
 assert.equal(stateRefreshes,1);
 assert.equal(liveRefreshes,1);
 if(stage==='fetch') assert.equal(requests[0].jsonReads,0);
 assert.equal(requests[0].body.text,'第一条');
 assert.equal(requests[1].body.text,'第二条');
 assert.notEqual(requests[0].body.message_id,requests[1].body.message_id);
""")


@pytest.mark.parametrize("same_text", [False, True])
@pytest.mark.parametrize("stage", ["fetch", "json"])
def test_old_success_keeps_latest_retry_message_id(same_text: bool, stage: str) -> None:
    """旧成功不能清除新请求失败后保留的幂等键，同文并发重试也受代次保护。"""
    _run_case("const sameText=" + json.dumps(same_text) + ";const stage=" + json.dumps(stage) + ";\n" + r"""
 const first=sendMessage();
 if(stage==='json'){
   respond(0,{waitForJson:true});
   await requests[0].jsonStarted.promise;
 }
 if(!sameText) text.value='第二条';
 const second=sendMessage();
 const latestAttempt=pendingMessageAttempt;
 respond(1,{ok:false,body:{error:'请稍后重试'}});
 await second;
 assert.equal(pendingMessageAttempt,latestAttempt);
 const failedState=visibleState();
 if(stage==='fetch') respond(0,{body:result('old')});
 else requests[0].json.resolve(result('old'));
 await first;
 assert.equal(pendingMessageAttempt,latestAttempt);
 assert.deepEqual(visibleState(),failedState);
 const retry=sendMessage();
 assert.equal(requests[2].body.message_id,requests[1].body.message_id);
 assert.equal(requests[0].body.message_id===requests[1].body.message_id,sameText);
 respond(2,{body:result('retry')});
 await retry;
 assertVisible('retry');
 assert.equal(pendingMessageAttempt,null);
 assert.deepEqual(polls,[{batch:'batch-retry',generation:3}]);
""")


@pytest.mark.parametrize("stage", ["fetch", "json"])
def test_stale_rejection_is_ignored_without_touching_current_attempt(stage: str) -> None:
    """晚到的网络或 JSON 异常属于旧请求，不能打断新请求或丢失其重试信息。"""
    _run_case("const stage=" + json.dumps(stage) + ";\n" + r"""
 const first=sendMessage();
 if(stage==='json'){
   respond(0,{waitForJson:true});
   await requests[0].jsonStarted.promise;
 }
 text.value='第二条';
 const second=sendMessage();
 const latestAttempt=pendingMessageAttempt;
 const current=visibleState();
 requests[0][stage==='fetch'?'response':'json'].reject(new Error('旧请求失败'));
 await first;
 assert.equal(pendingMessageAttempt,latestAttempt);
 assert.deepEqual(visibleState(),current);
 respond(1,{body:result('new')});
 await second;
 assertVisible('new');
""")


@pytest.mark.parametrize("stage", ["fetch", "json"])
def test_current_rejection_keeps_existing_error_and_retry_behavior(stage: str) -> None:
    """当前失败仍向调用方暴露，重试复用原 message_id，不伪造成功结果。"""
    _run_case("const stage=" + json.dumps(stage) + ";\n" + r"""
 const pending=sendMessage();
 const attempt=pendingMessageAttempt;
 const initial=visibleState();
 if(stage==='json'){
   respond(0,{waitForJson:true});
   await requests[0].jsonStarted.promise;
 }
 const rejected=assert.rejects(pending,/当前请求失败/);
 requests[0][stage==='fetch'?'response':'json'].reject(new Error('当前请求失败'));
 await rejected;
 assert.equal(pendingMessageAttempt,attempt);
 assert.deepEqual(visibleState(),initial);
 const retry=sendMessage();
 assert.equal(requests[1].body.message_id,requests[0].body.message_id);
 respond(1,{body:result('retry')});
 await retry;
 assertVisible('retry');
 assert.equal(pendingMessageAttempt,null);
""")
