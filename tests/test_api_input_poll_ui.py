"""后台轮询只展示经交付接口取得且仍属于本次输入的结果。"""

from pathlib import Path
import json
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("case", ["completed", "pending_then_completed", "expired", "late_fetch", "late_json"])
def test_api_poll_generation_and_delivery_endpoint(case):
    source = (Path(__file__).resolve().parents[1] / "scripts" / "agent_runtime_app.py").read_text()
    function = source.split("async function pollInputBatch", 1)[1].split("async function sendMessage", 1)[0]
    function = "async function pollInputBatch" + function
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
