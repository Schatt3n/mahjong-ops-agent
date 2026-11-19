"""后台轮询只展示经交付接口取得且仍属于本次输入的结果。"""

from pathlib import Path
import json
import re
import shutil
import subprocess

import pytest


STATIC = Path(__file__).resolve().parents[1] / "scripts" / "static"
OWNER_SCRIPT = STATIC / "owner_workspace.js"
DEVELOPER_SCRIPT = STATIC / "developer_workspace.js"


def _production_function(path: Path, name: str) -> str:
    match = re.search(
        rf"^(?:async )?function {name}\([^\n]*\)\{{\n.*?^\}}",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f"{path.name} 缺少 {name}"
    return match.group()


@pytest.mark.parametrize("case", ["completed", "pending_then_completed", "expired", "late_fetch", "late_json"])
def test_api_poll_generation_and_delivery_endpoint(case):
    function = _production_function(DEVELOPER_SCRIPT, "pollInputBatch")
    node = shutil.which("node")
    assert node, "UI concurrency verification requires the existing Node runtime"
    program = """
const assert = require('node:assert/strict');
let messagePollGeneration = 1;
let received = [], requests = [];
const sleep = async () => {};
const renderMessageResult = body => received.push(body);
const loadState = async () => {};
const refreshLive = async () => {};
let fetch;
""" + function + "\nconst scenario = " + json.dumps(case) + ";\n" + """
(async () => {
  if(scenario === 'late_fetch' || scenario === 'late_json'){
    let started, resume;
    const startedPromise = new Promise(resolve => started = resolve);
    fetch = async (url, options) => {
      requests.push({url, options});
      if(scenario === 'late_fetch'){
        started();
        return await new Promise(resolve => resume = resolve);
      }
      return {ok:true, status:200, json:async () => {
        started();
        return await new Promise(resolve => resume = resolve);
      }};
    };
    const pending = pollInputBatch('batch-1', 1);
    await startedPromise;
    messagePollGeneration = 2;
    const body = {final_reply:'outdated', input_status:'completed'};
    resume(scenario === 'late_fetch' ? {ok:true,status:200,json:async()=>body} : body);
    await pending;
    assert.equal(received.length, 0);
  }else{
    fetch = async (url, options) => {
      requests.push({url, options});
      if(scenario === 'expired') return {ok:false,status:404};
      const pending = scenario === 'pending_then_completed' && requests.length === 1;
      return {ok:true,status:200,json:async()=>({final_reply:pending?'':'current',input_status:pending?'processing':'completed'})};
    };
    await pollInputBatch('batch-1', 1);
    if(scenario === 'expired'){
      assert.equal(received[0].status,'superseded');
      assert.equal(received[0].final_reply,'');
    }else{
      assert.equal(received.at(-1).final_reply,'current');
      assert.equal(requests.length,scenario === 'pending_then_completed' ? 2 : 1);
    }
  }
  for(const request of requests){
    assert.equal(request.url,'/api/input-batches/result');
    assert.equal(request.options.method,'POST');
    assert.deepEqual(JSON.parse(request.options.body),{batch_id:'batch-1'});
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run([node, "-e", program], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_console_state_poll_keeps_raw_snapshot_on_demand():
    """运营页只刷新摘要，开发页收起调试区后拒绝迟到的原始快照。"""
    assert "loadStateDebug" not in OWNER_SCRIPT.read_text(encoding="utf-8")
    state_scripts = "\n".join((
        _production_function(OWNER_SCRIPT, "loadState"),
        _production_function(DEVELOPER_SCRIPT, "loadStateDebug"),
    ))
    program = r"""
const assert = require('node:assert/strict');
let stateRefreshGeneration = 0, stateDebugGeneration = 0;
const summaryStatus = {hidden:true}, debug = {open:false}, raw = {textContent:''};
const liveStatus = {textContent:''};
const counter = {dataset:{stateCount:'active_games'},textContent:'—'};
const openSeats = {dataset:{stateCount:'open_seats'},textContent:'—'};
const rendered = [];
const requests = [];
const document = {
  getElementById: id => ({stateSummaryStatus:summaryStatus,stateDebug:debug,stateRaw:raw})[id],
  querySelectorAll: selector => {
    assert.equal(selector,'[data-state-count]');
    return [counter,openSeats];
  },
};
const renderInviteDrafts = drafts => rendered.push(drafts);
const renderGameSchedule = games => rendered.push(games);
function fetch(url, options){
  assert.equal(options.cache,'no-store');
  return new Promise(resolve=>requests.push({url,resolve}));
}
function reply(index, data, status=200){
  requests[index].resolve({status,ok:status===200,json:async()=>data});
}
const summary = {
  summary:{active_games:2}, invite_drafts:[{draft_id:'review'}],
  active_game_schedule:[{seat_summary:{remaining_seats:2}},{seat_summary:{remaining_seats:1}}],
};
""" + "\n" + state_scripts + r"""
(async()=>{
  await loadStateDebug();
  assert.deepEqual(requests, []);
  const initial = loadState();
  assert.equal(requests[0].url,'/api/state?view=console');
  reply(0, summary);
  assert.equal(await initial,true);
  assert.equal(counter.textContent,2);
  assert.equal(openSeats.textContent,3);
  assert.match(liveStatus.textContent,/更新于 /);
  assert.deepEqual(rendered,[summary.invite_drafts,summary.active_game_schedule]);
  assert.equal(raw.textContent,'');

  debug.open = true;
  const stale = loadStateDebug();
  assert.equal(requests[1].url,'/api/state');
  debug.open = false;
  await loadStateDebug();
  reply(1,{runtime_config:{marker:'stale'}});
  await stale;
  assert.equal(raw.textContent,'');

  debug.open = true;
  const fresh = loadStateDebug();
  const snapshot = {games:[{game_id:'sample'}]};
  assert.equal(requests[2].url,'/api/state');
  reply(2,snapshot);
  await fresh;
  assert.deepEqual(JSON.parse(raw.textContent),snapshot);
  debug.open = false;
  await loadStateDebug();
  assert.equal(raw.textContent,'');

  const failed = loadState();
  reply(3,{error:'unauthorized'},401);
  assert.equal(await failed,false);
  assert.equal(counter.textContent,2);
  assert.equal(summaryStatus.hidden,false);
  assert.match(summaryStatus.textContent,/401/);
  process.stdout.write('ok\n');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    node = shutil.which("node")
    assert node, "控制台交互验证需要现有 Node 运行时"
    result = subprocess.run([node, "-"], input=program, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n"
