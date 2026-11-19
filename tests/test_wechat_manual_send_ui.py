"""离线执行控制台脚本，验证找人、人工确认与安全重试。"""

from pathlib import Path
import shutil
import subprocess


STATIC = Path(__file__).resolve().parents[1] / "scripts" / "static"
OWNER_HTML = STATIC / "owner_workspace.html"
OWNER_SCRIPT = STATIC / "owner_workspace.js"


def test_wechat_sender_starts_empty_with_explicit_recipient_selection():
    markup = OWNER_HTML.read_text(encoding="utf-8")
    script = OWNER_SCRIPT.read_text(encoding="utf-8")
    assert 'id="wechatTarget" type="search" autocomplete="off"' in markup
    assert 'id="wechatRecipientCandidates"' in markup
    assert 'id="wechatConversationList"' in markup
    assert 'id="wechatHistoryList"' in markup
    assert 'href="/wechat-permissions"' in markup
    assert "saveWechatContactPolicy" not in script
    assert 'onclick="refreshWechatRecipient()" hidden disabled>刷新联系人资料' in markup
    assert 'id="wechatSendButton" type="button" onclick="sendWechatText()" disabled' in markup
    assert "xml31323" not in markup
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
    this.dataset = {};
    this.attributes = {};
    this.scrollHeight = this.scrollTop = this.clientHeight = 0;
  }
  append(...children){
    for(const child of children) child.parentNode = this;
    this.children.push(...children);
  }
  replaceChildren(){
    for(const child of this.children) child.parentNode = null;
    this.children = [];
  }
  get childElementCount(){return this.children.length;}
  querySelectorAll(selector){
    assert.equal(selector, 'button');
    return this.children.map(row=>row.children[0]);
  }
  focus(){focused = this;}
  setAttribute(name,value){this.attributes[name] = value;}
}
const document = {createElement: () => new Element()};
const wechatTarget = new Element(), wechatText = new Element(), wechatSendButton = new Element();
const wechatRecipientCandidates = new Element(), wechatRecipientStatus = new Element(), wechatSendOutput = new Element();
const wechatRecipientRefreshButton = new Element(), wechatChannelStatus = new Element();
const wechatConversationList = new Element(), wechatListStatus = new Element();
const wechatHistoryTitle = new Element(), wechatHistoryWrap = new Element(), wechatHistoryList = new Element();
const wechatHistoryRefresh = new Element(), wechatHistoryOlder = new Element(), wechatHistoryStatus = new Element();
const stored = new Map();
const sessionStorage = {
  getItem:key=>stored.get(key) ?? null,
  setItem:(key,value)=>stored.set(key,value),
  removeItem:key=>stored.delete(key)
};
wechatRecipientCandidates.hidden = true;
wechatRecipientRefreshButton.hidden = true;
wechatRecipientRefreshButton.disabled = true;
wechatSendButton.disabled = true;
function setTimeout(callback, ms){
  assert.equal(ms, 250);
  timers.set(++nextTimer, callback);
  return nextTimer;
}
function clearTimeout(timer){timers.delete(timer);}
function setInterval(callback,ms){assert.equal(ms,6000);}
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


def _run_console_case(case, *, exercise_conversations=False):
    script = OWNER_SCRIPT.read_text(encoding="utf-8")
    startup = "\ninitOwnerWorkspace();\n"
    assert script.endswith(startup)
    # 保留真实页面的事件安装。后台查询用空函数隔离，避免干扰发送断言。
    bind = """
const productionLoadState = loadState, productionLoadChannel = loadWechatChannelStatus;
const productionLoadList = loadWechatConversations;
loadState = () => {};
loadWechatChannelStatus = () => {};
loadWechatConversations = () => {};
initOwnerWorkspace();
loadState = productionLoadState;
loadWechatChannelStatus = productionLoadChannel;
"""
    if exercise_conversations:
        bind += "loadWechatConversations = productionLoadList;\n"
    else:
        bind += "loadWechatConversation = () => {};\n"
    program = (HARNESS + script.removesuffix(startup) + bind + "\n(async()=>{\n" + case + "\n})()"
               ".then(()=>process.stdout.write('ok\\n'))"
               ".catch(error=>{console.error(error.stack);process.exitCode=1;});")
    node = shutil.which("node")
    assert node, "控制台交互验证需要现有 Node 运行时"
    result = subprocess.run([node, "-"], input=program, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n", "异步交互没有执行完成"


def test_owner_takes_trial_draft_once_without_selecting_or_sending():
    """开发页显式交接的草稿只取用一次，始终等候人工择人和确认。"""
    _run_console_case(r"""
const key = 'mahjong.owner.wechat-draft';
sessionStorage.setItem(key,'试运行建议回复');
restoreTrialDraft();
assert.equal(wechatText.value,'试运行建议回复');
assert.equal(sessionStorage.getItem(key),null);
assert.equal(selectedWechatRecipient,null);
assert.equal(wechatSendButton.disabled,true);
await sendWechatText();
assert.deepEqual(requests,[]);
assert.deepEqual(confirmations,[]);

wechatText.value = '老板已有草稿';
sessionStorage.setItem(key,'另一条试运行草稿');
restoreTrialDraft();
assert.equal(wechatText.value,'老板已有草稿');
assert.equal(sessionStorage.getItem(key),null);
""")


def test_recent_wechat_selects_exact_thread_and_loads_older_confirmed_messages():
    """最近会话可按精确标识切换，跨页消息按时间排列，异步旧会话无法串到新会话。"""
    _run_console_case(r"""
const rows = [{...first,last_at:'2026-11-19T04:01:00Z',pending_tasks:2},{...second,last_at:null,pending_tasks:1}];
const list = loadWechatConversations();
assert.equal(requests[0].url,'/api/wechat/conversations');
respond(requests[0],{ok:true,items:rows,partial:true});
await list;
const buttons = wechatConversationList.querySelectorAll('button');
assert.equal(buttons.length,2);
assert.match(wechatListStatus.textContent,/部分会话暂时无法读取/);
assert.match(buttons[0].children[0].textContent,/2 项待处理/);
assert.match(buttons[1].children[0].textContent,/1 项待处理/);
assert.equal(selectedWechatRecipient,null);
buttons[0].onclick();
buttons[1].onclick();
assert.equal(focused,wechatHistoryTitle);
assert.equal(buttons[0].attributes['aria-pressed'],'false');
assert.equal(buttons[1].attributes['aria-pressed'],'true');
assert.equal(requests[1].options.signal.aborted,true);
const url = new URL(requests[2].url,'http://localhost');
assert.equal(url.pathname,'/api/wechat/conversation');
assert.equal(url.searchParams.get('thread_id'),second.id);
assert.equal(url.searchParams.get('kind'),'direct');
assert.equal(url.searchParams.get('limit'),'40');
const prior = {id:'m1',direction:'inbound',author:'contact',speaker:'李店长',text:'<b>今天营业吗</b>',at:'2026-11-19T04:00:00Z'};
const reply = {id:'m2',direction:'outbound',author:'agent',text:'今天营业。',at:'2026-11-19T04:01:00Z'};
const result = {ok:true,id:second.id,kind:second.kind,items:[reply,prior],nextCursor:'older+1='};
respond(requests[2],result);
await settle();
respond(requests[1],{...result,id:first.id,items:[{...prior,text:'不应出现'}]});
await settle();
assert.deepEqual(wechatHistoryList.children.map(row=>row.children[1].textContent),[prior.text,reply.text]);
assert.equal(wechatHistoryList.children[1].dataset.direction,'outbound');
assert.match(wechatHistoryList.children[1].children[0].textContent,/Agent/);
assert.equal(wechatHistoryOlder.hidden,false);
const older = loadWechatConversation({older:true});
assert.equal(new URL(requests[3].url,'http://localhost').searchParams.get('cursor'),'older+1=');
const firstMessage = {...prior,id:'m0',text:'我来过一次',at:'2026-11-19T03:59:00Z'};
respond(requests[3],{...result,items:[firstMessage,prior],nextCursor:null});
await older;
assert.deepEqual(wechatHistoryList.children.map(row=>row.children[1].textContent),[firstMessage.text,prior.text,reply.text]);
assert.equal(wechatHistoryOlder.hidden,true);
await loadWechatConversation({quiet:true});
assert.equal(requests.length,4);
assert.equal(wechatSendButton.disabled,true);
assert.deepEqual(confirmations,[]);
""", exercise_conversations=True)


def test_wechat_thread_drafts_and_send_receipts_stay_with_exact_recipient():
    """切换会话不带走草稿，确认后的发送回执和重新读取均绑定原会话。"""
    _run_console_case(r"""
const history = row=>({ok:true,id:row.id,kind:row.kind,items:[],nextCursor:null});
selectWechatRecipient(first);
respond(requests[0],history(first));
await settle();
wechatText.value = '只给第一位的正文';
confirmed = true;
const pending = sendWechatText();
assert.equal(requests[1].url,'/api/wechat/send');
assert.equal(JSON.parse(requests[1].options.body).recipient_id,first.id);
selectWechatRecipient(second);
respond(requests[2],history(second));
await settle();
assert.equal(wechatText.value,'');
assert.equal(wechatSendOutput.textContent,'');
wechatText.value = '第二位的草稿';
respond(requests[1],{ok:true,sent:true,context_recorded:true});
await pending;
assert.equal(requests[3].url,'/api/wechat/conversations');
respond(requests[3],{ok:true,items:[]});
await settle();
assert.equal(wechatText.value,'第二位的草稿');
assert.equal(wechatSendOutput.textContent,'');
assert.equal(wechatHistoryList.children.length,0);
selectWechatRecipient(first);
assert.match(wechatSendOutput.textContent,/已发送给/);
assert.equal(wechatText.value,'');
respond(requests[4],{...history(first),items:[{id:'m1',at:'2026-11-19T04:00:00Z',direction:'outbound',author:'owner',text:'只给第一位的正文'}]});
await settle();
assert.equal(wechatHistoryList.children[0].children[1].textContent,'只给第一位的正文');
selectWechatRecipient(second);
assert.equal(wechatText.value,'第二位的草稿');
assert.equal(wechatSendOutput.textContent,'');
assert.equal(wechatHistoryList.children.length,0);
assert.equal(requests.filter(request=>request.options.method==='POST').length,1);
assert.equal(confirmations.length,1);
""", exercise_conversations=True)


def test_group_message_private_action_targets_verified_friend_and_preserves_thread_drafts():
    """只有已核实的群消息作者能进入私聊，正文仍需由老板单独确认。"""
    _run_console_case(r"""
const group = {id:'fixtures@chatroom',kind:'group',displayName:'夹具群'};
const friend = {id:second.id,kind:'direct',displayName:'私聊备注'};
const inbound = {id:'group-1',at:'2026-11-19T04:00:00Z',direction:'inbound',author:'contact',speaker:'群内昵称',
  text:'群消息',direct_recipient:{...friend,composerText:'不要复制'}};
const messages = [
  inbound,
  {...inbound,id:'group-2',direct_recipient:null},
  {...inbound,id:'group-3',direct_recipient:{...friend,kind:'group'}},
  {...inbound,id:'group-4',direction:'outbound',author:'agent'},
];
const history = (recipient,items) => ({ok:true,id:recipient.id,kind:recipient.kind,items,nextCursor:null});
const sourceThread = request => new URL(request.url,'http://localhost').searchParams.get('thread_id');
selectWechatRecipient(group);
respond(requests[0],history(group,messages));
await settle();
const metadata = wechatHistoryList.children.map(row=>row.children[0]);
assert.match(metadata[0].textContent,/群内昵称/);
const privateButton = metadata[0].children[0];
assert.equal(privateButton.textContent,'私聊');
assert.match(privateButton.attributes['aria-label'],/私聊备注.*wxid_friend_2/);
assert.deepEqual(metadata.slice(1).map(info=>info.children.length),[0,0,0]);

wechatText.value = '群聊草稿';
privateButton.onclick();
assert.equal(sourceThread(requests[1]),friend.id);
assert.equal(wechatTarget.value,'私聊备注');
assert.equal(wechatText.value,'');
assert.equal(focused,wechatText);
assert.deepEqual(confirmations,[]);
assert.equal(wechatSendButton.disabled,true);
respond(requests[1],history(friend,[inbound]));
await settle();
assert.equal(wechatHistoryList.children[0].children[0].children.length,0);
privateButton.onclick();
assert.equal(requests.length,2);

wechatText.value = '单独编辑的私聊草稿';
selectWechatRecipient(group);
assert.equal(wechatText.value,'群聊草稿');
privateButton.onclick();
assert.equal(wechatTarget.value,'夹具群');
assert.equal(requests.length,3);
respond(requests[2],history(group,messages));
await settle();
wechatHistoryList.children[0].children[0].children[0].onclick();
assert.equal(wechatText.value,'单独编辑的私聊草稿');
assert.equal(sourceThread(requests[3]),friend.id);
assert.equal(requests.filter(request=>request.options.method==='POST').length,0);
assert.deepEqual(confirmations,[]);
""", exercise_conversations=True)


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


def test_wechat_contact_refresh_updates_exact_selection_and_keeps_draft():
    _run_console_case(r"""
await refreshWechatRecipient();
selectWechatRecipient({id:'club@chatroom',kind:'group',displayName:'牌友群'});
assert.equal(wechatRecipientRefreshButton.hidden,true);
await refreshWechatRecipient();
assert.equal(requests.length,0);
selectWechatRecipient(first);
wechatText.value = '原来的草稿';
assert.equal(wechatRecipientRefreshButton.hidden,false);
const refresh = refreshWechatRecipient();
await refreshWechatRecipient();
assert.equal(requests.length,1);
assert.equal(wechatRecipientRefreshButton.disabled,true);
assert.match(wechatRecipientStatus.textContent,/正在拉取/);
assert.equal(requests[0].url,'/api/wechat/recipients/refresh');
assert.equal(requests[0].options.method,'POST');
assert.deepEqual(JSON.parse(requests[0].options.body),{recipient_id:first.id});
respond(requests[0],{ok:true,recipient:{...first,displayName:'新备注',remark:'新备注'},projection:'complete'});
await refresh;
assert.equal(wechatRecipientRefreshButton.disabled,false);
assert.equal(wechatTarget.value,'新备注');
assert.equal(selectedWechatRecipient.id,first.id);
assert.match(wechatRecipientStatus.textContent,/新备注.*已拉取并保存/);
assert.equal(wechatText.value,'原来的草稿');
assert.equal(wechatSendButton.disabled,false);
assert.deepEqual(confirmations,[]);
assert.equal(requests.length,1);
""")


def test_wechat_contact_refresh_respects_later_selection_and_reports_outcome():
    _run_console_case(r"""
selectWechatRecipient(first);
const previous = refreshWechatRecipient();
typeWechatSearch('另一位');
assert.equal(wechatRecipientRefreshButton.hidden,true);
selectWechatRecipient(second);
const selectionStatus = wechatRecipientStatus.textContent;
respond(requests[0],{ok:true,recipient:{...first,displayName:'旧选择的新备注'},projection:'complete'});
await previous;
assert.equal(wechatTarget.value,second.displayName);
assert.equal(wechatRecipientStatus.textContent,selectionStatus);
assert.equal(wechatRecipientRefreshButton.disabled,false);

const pending = refreshWechatRecipient();
respond(requests[1],{ok:false,pending:true,message:'微信仍在拉取资料。'},202);
await pending;
assert.match(wechatRecipientStatus.textContent,/微信仍在拉取资料/);

const partial = refreshWechatRecipient();
respond(requests[2],{ok:true,recipient:second,projection:'partial'});
await partial;
assert.match(wechatRecipientStatus.textContent,/部分同步.*本机已保存的资料/);

const withoutRemark = refreshWechatRecipient();
respond(requests[3],{ok:true,recipient:{...second,remark:null},projection:'complete'});
await withoutRemark;
assert.match(wechatRecipientStatus.textContent,/没有返回微信备注/);

const mismatched = refreshWechatRecipient();
respond(requests[4],{ok:true,recipient:first,projection:'complete'});
await mismatched;
assert.match(wechatRecipientStatus.textContent,/与所选联系人不一致/);
assert.equal(selectedWechatRecipient.id,second.id);

const unsupported = refreshWechatRecipient();
respond(requests[5],{ok:false,message:'当前微信适配器不支持单独刷新联系人。'},501);
await unsupported;
assert.match(wechatRecipientStatus.textContent,/刷新联系人资料失败.*适配器不支持/);
assert.equal(wechatTarget.value,second.displayName);
assert.deepEqual(confirmations,[]);
assert.ok(requests.every(request => request.url === '/api/wechat/recipients/refresh'));
""")


def test_wechat_send_disabled_explains_manual_switch_without_losing_draft():
    _run_console_case(r"""
selectWechatRecipient(first);
wechatText.value = '晚上好';
confirmed = true;
const rejected = sendWechatText();
respond(requests[0],{
  ok:false, sent:false, error:'wechat_send_channel_disabled',
  message:'微信发送总闸已关闭。请先点击「开启微信发送总闸」，然后重新确认发送。'
},409);
await rejected;
assert.match(wechatSendOutput.textContent,/未发送。微信发送总闸已关闭.*开启微信发送总闸.*重新确认发送/);
assert.equal(wechatText.value,'晚上好');
assert.equal(selectedWechatRecipient.id,first.id);
assert.equal(wechatSendButton.disabled,false);
assert.deepEqual(requests.map(request => request.url),['/api/wechat/send']);
""")


def test_wechat_health_and_switches_use_console_origin():
    _run_console_case(r"""
const health = loadWechatChannelStatus();
assert.equal(requests[0].url,'/api/wechat/health');
respond(requests[0],{ok:true,adapter:'agent-wechat',send_channel_enabled:true,auto_send_reply:false});
await health;
assert.match(wechatChannelStatus.textContent,/发送总闸：开启.*自动外发：暂停/);
assert.match(wechatChannelStatus.textContent,/未启用私聊白名单.*未单独授权的手动发送也需要开启自动外发/);
await setWechatSendChannel(true);
assert.equal(requests.length,1);
confirmed = true;
const channel = setWechatSendChannel(false);
assert.equal(requests[1].url,'/api/wechat/send-channel');
assert.deepEqual(JSON.parse(requests[1].options.body),{enabled:false});
respond(requests[1],{ok:true,send_channel_enabled:false});
await channel;
assert.match(wechatChannelStatus.textContent,/发送总闸：暂停.*自动外发：暂停/);
const autoSend = setWechatAutoSend(false);
assert.equal(requests[2].url,'/api/wechat/auto-send');
assert.deepEqual(JSON.parse(requests[2].options.body),{enabled:false});
respond(requests[2],{ok:false,message:'发送总闸关闭'},409);
await autoSend;
assert.match(wechatChannelStatus.textContent,/切换失败.*发送总闸关闭/);
""")
