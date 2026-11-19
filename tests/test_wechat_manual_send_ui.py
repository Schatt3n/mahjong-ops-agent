"""离线执行控制台脚本，验证找人、人工确认与安全重试。"""

import ast
from pathlib import Path
import shutil
import subprocess


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "agent_runtime_app.py"


def _console_html():
    module = ast.parse(SOURCE.read_text())
    index = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "index_html")
    return ast.literal_eval(index.body[-1].value)


def test_wechat_sender_starts_empty_with_explicit_recipient_selection():
    html = _console_html()
    markup, script = html.split("<script>", 1)
    assert 'id="wechatTarget" type="search" autocomplete="off"' in markup
    assert 'id="wechatRecipientCandidates"' in markup
    assert 'id="wechatSendButton" type="button" onclick="sendWechatText()" disabled' in markup
    assert "xml31323" not in html
    assert "127.0.0.1:8791" not in script


HARNESS = r"""
const assert = require('node:assert/strict');
const requests = [], confirmations = [], timers = new Map();
let focused, nextTimer = 0, confirmed = false;
class Element extends EventTarget {
  constructor(){
    super();
    this.children = [];
    this.value = '';
    this.textContent = '';
  }
  append(child){this.children.push(child);}
  replaceChildren(){this.children = [];}
  querySelectorAll(selector){
    assert.equal(selector, 'button');
    return this.children.map(row=>row.children[0]);
  }
  focus(){focused = this;}
}
const document = {createElement: () => new Element()};
const wechatTarget = new Element(), wechatText = new Element(), wechatSendButton = new Element();
const wechatRecipientCandidates = new Element(), wechatRecipientStatus = new Element(), wechatSendOutput = new Element();
wechatRecipientCandidates.hidden = true;
wechatSendButton.disabled = true;
function setTimeout(callback, ms){
  assert.equal(ms, 250);
  timers.set(++nextTimer, callback);
  return nextTimer;
}
function clearTimeout(timer){timers.delete(timer);}
function typeWechatSearch(value, isComposing = false){
  wechatTarget.value = value;
  wechatTarget.dispatchEvent(Object.assign(new Event('input'),{isComposing}));
}
function fireTimers(){
  const queued = [...timers.values()];
  timers.clear();
  queued.forEach(callback=>callback());
}
function fetch(url, options){
  assert.match(url, /^\/api\/wechat\//, '测试不允许访问真实网络');
  return new Promise((resolve,reject) => requests.push({url,options,resolve,reject}));
}
function respond(request, body, status=200){
  request.resolve({ok:status < 400, status, json:async()=>body});
}
function confirm(text){confirmations.push(text);return confirmed;}
function settle(){return new Promise(resolve=>setImmediate(resolve));}
const first = {id:'wxid_friend_1',kind:'direct',displayName:'小李',remark:'李师傅',nickname:'小李',alias:'lee-plus'};
const second = {id:'wxid_friend_2',kind:'direct',displayName:'小李',remark:'李店长',nickname:'小李',alias:'lee-other'};
"""


def _run_console_case(case):
    script = _console_html().split("<script>", 1)[1].split("</script>", 1)[0]
    # 避免初始化时请求无关的页面数据，其他交互全部执行实际脚本。
    start_refresh = "\ntoggleAutoRefresh();\nrefreshLive();\n"
    assert script.endswith(start_refresh)
    program = (HARNESS + script.removesuffix(start_refresh) + "\n(async()=>{\n" + case + "\n})()"
               ".then(()=>process.stdout.write('ok\\n'))"
               ".catch(error=>{console.error(error.stack);process.exitCode=1;});")
    node = shutil.which("node")
    assert node, "控制台交互验证需要现有 Node 运行时"
    result = subprocess.run([node, "-"], input=program, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n", "异步交互没有执行完成"


def test_wechat_recipients_search_after_chinese_composition_ends():
    """中文组字结束无需额外输入事件即可发起搜索并展示候选。"""
    _run_console_case(r"""
typeWechatSearch('xiaoli',true);
typeWechatSearch('小李',true);
fireTimers();
assert.equal(requests.length,0);
wechatTarget.dispatchEvent(new Event('compositionend'));
fireTimers();
assert.equal(requests.length,1);
assert.equal(new URL(requests[0].url,'http://localhost').searchParams.get('query'),'小李');
respond(requests[0],{ok:true,candidates:[first]});
await settle();
assert.equal(wechatRecipientCandidates.hidden,false);
assert.equal(wechatRecipientCandidates.children.length,1);
assert.equal(wechatRecipientCandidates.children[0].children[0].textContent,'小李');
assert.equal(wechatSendButton.disabled,true);
""")


def test_wechat_fuzzy_search_leaves_selection_to_operator():
    _run_console_case(r"""
typeWechatSearch('李');
typeWechatSearch('李+');
assert.equal(requests.length,0);
fireTimers();
assert.equal(requests.length,1);
assert.equal(new URL(requests[0].url,'http://localhost').searchParams.get('query'),'李+');
assert.equal(requests[0].options.cache,'no-store');
typeWechatSearch('小李');
assert.equal(requests[0].options.signal.aborted,true);
fireTimers();
respond(requests[1],{ok:true,candidates:[first,second]});
await settle();
respond(requests[0],{ok:true,candidates:[{id:'stale',displayName:'旧候选'}]});
await settle();
assert.equal(wechatSendButton.disabled,true);
assert.equal(wechatRecipientCandidates.hidden,false);
assert.equal(wechatRecipientCandidates.children.length,2);
const firstButton = wechatRecipientCandidates.children[0].children[0];
const secondButton = wechatRecipientCandidates.children[1].children[0];
assert.equal(firstButton.textContent,'小李');
assert.match(firstButton.children[0].textContent,/李师傅.*wxid_friend_1/);
assert.match(secondButton.children[0].textContent,/李店长.*wxid_friend_2/);
navigateWechatRecipients({key:'ArrowDown',target:wechatTarget,preventDefault(){}});
assert.equal(focused,firstButton);
navigateWechatRecipients({key:'ArrowDown',target:firstButton,preventDefault(){}});
assert.equal(focused,secondButton);
secondButton.onclick();
assert.equal(focused,wechatText);
assert.equal(wechatRecipientCandidates.hidden,true);
assert.match(wechatRecipientStatus.textContent,/wxid_friend_2/);
assert.equal(wechatSendButton.disabled,true);
wechatText.value = '晚上好';
updateWechatSendControl();
assert.equal(wechatSendButton.disabled,false);
typeWechatSearch('另一位');
assert.equal(wechatSendButton.disabled,true);
await sendWechatText();
assert.deepEqual(confirmations,[]);
assert.equal(requests.length,2);
""")


def test_wechat_send_confirms_exact_recipient_and_reuses_unconfirmed_request():
    _run_console_case(r"""
wechatText.value = '晚上好';
updateWechatSendControl();
await sendWechatText();
assert.deepEqual(confirmations,[]);
selectWechatRecipient(second);
await sendWechatText();
assert.match(confirmations[0],/小李.*wxid_friend_2/);
assert.match(confirmations[0],/晚上好/);
assert.equal(requests.length,0);
confirmed = true;
const pending = sendWechatText();
assert.equal(wechatSendButton.disabled,true);
await sendWechatText();
assert.equal(requests.length,1);
assert.equal(requests[0].url,'/api/wechat/send');
assert.equal(requests[0].options.method,'POST');
const body = JSON.parse(requests[0].options.body);
assert.deepEqual(body,{recipient_id:'wxid_friend_2',kind:'direct',text:'晚上好',request_id:body.request_id});
assert.ok(body.request_id);
requests[0].reject(new Error('connection closed'));
await pending;
assert.match(wechatSendOutput.textContent,/未收到发送回执/);
assert.equal(wechatSendButton.disabled,false);
assert.equal(requests.length,1);
const retry = sendWechatText();
assert.equal(JSON.parse(requests[1].options.body).request_id,body.request_id);
respond(requests[1],{ok:false,sent:false,delivery_uncertain:true,message:'待核对'},504);
await retry;
assert.match(wechatSendOutput.textContent,/发送结果待核对/);
wechatText.value = '改了正文';
updateWechatSendControl();
const edit = sendWechatText();
assert.notEqual(JSON.parse(requests[2].options.body).request_id,body.request_id);
respond(requests[2],{ok:true,sent:true,context_recorded:true});
await edit;
assert.match(wechatSendOutput.textContent,/已发送给.*会话上下文/);
wechatText.value = '';
updateWechatSendControl();
assert.equal(wechatSendButton.disabled,true);
""")


def test_wechat_health_and_switches_use_console_origin():
    _run_console_case(r"""
const health = loadWechatChannelStatus();
assert.equal(requests[0].url,'/api/wechat/health');
respond(requests[0],{ok:true,send_channel_enabled:false,auto_send_reply:false});
await health;
assert.equal(JSON.parse(wechatSendOutput.textContent).send_channel_enabled,false);
await setWechatSendChannel(true);
assert.equal(requests.length,1);
confirmed = true;
const channel = setWechatSendChannel(false);
assert.equal(requests[1].url,'/api/wechat/send-channel');
assert.deepEqual(JSON.parse(requests[1].options.body),{enabled:false});
respond(requests[1],{ok:true});
await channel;
const autoSend = setWechatAutoSend(false);
assert.equal(requests[2].url,'/api/wechat/auto-send');
assert.deepEqual(JSON.parse(requests[2].options.body),{enabled:false});
respond(requests[2],{ok:false,message:'发送总闸关闭'},409);
await autoSend;
assert.match(wechatSendOutput.textContent,/切换失败.*发送总闸关闭/);
""")
