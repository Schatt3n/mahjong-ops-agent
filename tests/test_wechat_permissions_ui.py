"""离线执行权限页面脚本，验证名单浏览和精确权限保存。"""

from pathlib import Path
import shutil
import subprocess


HTML = Path(__file__).resolve().parents[1] / "scripts" / "static" / "wechat_permissions.html"


HARNESS = r"""
const assert = require('node:assert/strict');
const requests = [], timers = new Map();
let nextTimer = 0, focused;
class Element extends EventTarget {
  constructor(){
    super();
    this.value = '';
    this.textContent = '';
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.children = [];
    this.dataset = {};
    this.attributes = {};
  }
  append(...children){this.children.push(...children);}
  replaceChildren(){this.children = [];}
  querySelectorAll(selector){
    assert.equal(selector,'button');
    return this.children.map(row=>row.children[0]);
  }
  focus(){focused = this;}
  setAttribute(name,value){this.attributes[name] = value;}
}
const [policyTab,contactsTab,groupsTab] = ['policy','contacts','groups'].map(view=>{
  const button = new Element();
  button.dataset.view = view;
  return button;
});
const document = {
  querySelectorAll(selector){
    assert.equal(selector,'[data-view]');
    return [policyTab,contactsTab,groupsTab];
  },
  createElement:()=>new Element()
};
const permissionSearch = new Element(), scopeStatus = new Element(), permissionEditor = new Element();
const permissionName = new Element(), permissionGuidance = new Element(), permissionMode = new Element();
const permissionAllow = new Element(), permissionInherit = new Element(), savePermission = new Element();
const permissionManagedOnly = new Element();
const permissionManaged = new Element(), permissionAutoReply = new Element();
const contactProfile = new Element(), refreshContactProfile = new Element(), profileStatus = new Element();
const saveStatus = new Element(), rosterStatus = new Element(), permissionRows = new Element();
const previousPage = new Element(), pageStatus = new Element(), nextPage = new Element();
function fetch(url,options){
  assert.match(url,/^\/api\/wechat\/(permission-roster\?|contact-policy(?:\/refresh)?$|group-permission$)/);
  return new Promise((resolve,reject)=>requests.push({url,options,resolve,reject}));
}
function respond(request,body,status=200){
  request.resolve({ok:status<400,status,json:async()=>body});
}
function setTimeout(callback,ms){
  assert.equal(ms,250);
  timers.set(++nextTimer,callback);
  return nextTimer;
}
function clearTimeout(timer){timers.delete(timer);}
function fireTimers(){
  for(const timer of [...timers.values()]) timer();
}
function typeSearch(text,isComposing=false){
  permissionSearch.value = text;
  permissionSearch.dispatchEvent(Object.assign(new Event('input'),{isComposing}));
}
function params(item){return new URL(item.url,'http://localhost').searchParams;}
function rowButton(index){return permissionRows.children[index].children[0];}
const settle = ()=>new Promise(resolve=>setImmediate(resolve));
const inactive = {managed:false,auto_reply:false};
const contact = {id:'wxid_li',kind:'direct',displayName:'李师傅',remark:'老李',nickname:'李',mode:'unassigned',
  automation:inactive};
const group = {id:'club@chatroom',kind:'group',displayName:'牌友群',mode:'unassigned',override:null,
  automation:inactive,group_rules:{managed:true,auto_reply:true,observe:false}};
"""


def _run_permissions_case(case):
    markup, script = HTML.read_text().split("<script>", 1)
    assert "确认发送" not in markup
    script = script.split("</script>", 1)[0]
    assert script.endswith("\nvoid loadRoster();\n")
    program = (HARNESS + script + "\n(async()=>{\n" + case + "\n})()"
               ".then(()=>process.stdout.write('ok\\n'))"
               ".catch(error=>{console.error(error.stack);process.exitCode=1;});")
    node = shutil.which("node")
    assert node, "权限页面验证需要现有 Node 运行时"
    result = subprocess.run([node, "-"], input=program, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n", "异步交互没有执行完成"


def test_roster_switches_views_and_preserves_query_across_cursor_pages():
    """三类列表可分别查看，分页使用搜索范围对应的服务端游标。"""
    _run_permissions_case(r"""
assert.equal(params(requests[0]).get('view'),'policy');
assert.equal(params(requests[0]).get('limit'),'40');
respond(requests[0],{ok:true,managed:true,items:[{...contact,mode:'allowlist'}],nextCursor:null});
await settle();
assert.equal(rowButton(0).children[1].textContent,'白名单');
contactsTab.onclick();
assert.equal(params(requests[1]).get('view'),'contacts');
assert.equal(contactsTab.attributes['aria-pressed'],'true');
respond(requests[1],{ok:true,managed:true,items:[contact],nextCursor:'cursor-2'});
await settle();
assert.equal(rowButton(0).children[1].textContent,'仅观察（未分配）');
assert.equal(nextPage.disabled,false);
nextPage.onclick();
assert.equal(params(requests[2]).get('cursor'),'cursor-2');
respond(requests[2],{ok:true,managed:true,items:[],nextCursor:null});
await settle();
assert.equal(previousPage.disabled,false);
assert.equal(pageStatus.textContent,'第 2 页');
previousPage.onclick();
assert.equal(params(requests[3]).get('cursor'),null);

typeSearch('小li',true);
fireTimers();
assert.equal(requests.length,4);
assert.equal(requests[3].options.signal.aborted,true);
typeSearch('李师傅',true);
permissionSearch.dispatchEvent(new Event('compositionend'));
fireTimers();
assert.equal(params(requests[4]).get('query'),'李师傅');
assert.equal(params(requests[4]).get('cursor'),null);
respond(requests[4],{ok:true,managed:true,items:[contact],nextCursor:null});
respond(requests[3],{ok:true,managed:true,items:[{...contact,displayName:'旧结果'}],nextCursor:'old'});
await settle();
assert.equal(rowButton(0).children[0].textContent,'李师傅');
assert.equal(nextPage.disabled,true);

groupsTab.onclick();
assert.equal(params(requests[5]).get('view'),'groups');
assert.equal(params(requests[5]).get('query'),'李师傅');
respond(requests[5],{ok:true,managed:true,items:[group],nextCursor:null});
await settle();
assert.equal(rowButton(0).children[1].textContent,'沿用现有外发规则');
assert.match(rowButton(0).children[2].textContent,/club@chatroom.*托管关闭.*自动回复关闭.*已有配置：托管、自动回复/);
""")


def test_contact_and_group_saves_use_exact_ids_and_distinct_rules():
    """私聊自动回复要求托管，群自动化与手动外发规则独立。"""
    _run_permissions_case(r"""
respond(requests[0],{ok:true,managed:false,items:[contact,group],nextCursor:null});
await settle();
assert.match(scopeStatus.textContent,/首次保存.*默认仅观察/);
rowButton(0).onclick();
assert.equal(focused,permissionMode);
assert.equal(permissionInherit.disabled,true);
permissionMode.value = 'allowlist';
permissionMode.onchange();
assert.equal(permissionAutoReply.disabled,true);
permissionManaged.checked = true;
permissionManaged.onchange();
assert.equal(permissionAutoReply.disabled,false);
permissionAutoReply.checked = true;
permissionAutoReply.onchange();
const contactSave = saveSelectedPermission();
assert.equal(requests[1].url,'/api/wechat/contact-policy');
assert.deepEqual(JSON.parse(requests[1].options.body),
  {recipient_id:contact.id,mode:'allowlist',managed:true,auto_reply:true});
const allowed = {...contact,mode:'allowlist',automation:{managed:true,auto_reply:true}};
respond(requests[1],{ok:true,managed:true,recipient:allowed});
await contactSave;
assert.equal(saveStatus.textContent,'权限已保存。');
const automatedGroup = {...group,automation:{managed:true,auto_reply:true}};
respond(requests[2],{ok:true,managed:true,items:[allowed,automatedGroup],nextCursor:null});
await settle();
assert.match(rowButton(0).children[2].textContent,/托管开启.*自动回复开启/);
rowButton(1).onclick();
assert.equal(permissionAllow.textContent,'手动外发白名单');
assert.equal(permissionInherit.disabled,false);
assert.equal(permissionManaged.checked,true);
assert.equal(permissionAutoReply.checked,true);
assert.match(permissionGuidance.textContent,/手动发送资格.*托管和自动回复可单独设置/);
permissionMode.value = 'observe';
permissionMode.onchange();
assert.equal(permissionManaged.checked,false);
assert.equal(permissionManaged.disabled,true);
assert.equal(permissionAutoReply.checked,false);
assert.equal(permissionAutoReply.disabled,true);
const groupSave = saveSelectedPermission();
assert.equal(requests[3].url,'/api/wechat/group-permission');
assert.deepEqual(JSON.parse(requests[3].options.body),
  {recipient_id:group.id,mode:'observe',managed:false,auto_reply:false});
const observed = {...group,mode:'observe',override:'observe'};
respond(requests[3],{ok:true,item:observed});
await groupSave;
respond(requests[4],{ok:true,managed:true,items:[allowed,observed],nextCursor:null});
await settle();
assert.equal(rowButton(1).children[1].textContent,'仅观察');
assert.match(rowButton(1).children[2].textContent,/自动回复关闭.*已有配置：托管、自动回复/);

permissionMode.value = 'inherit';
permissionMode.onchange();
assert.equal(permissionAutoReply.disabled,false);
permissionAutoReply.checked = true;
permissionAutoReply.onchange();
const clearGroup = saveSelectedPermission();
assert.deepEqual(JSON.parse(requests[5].options.body),
  {recipient_id:group.id,mode:'inherit',managed:false,auto_reply:true});
const inherited = {...group,automation:{managed:false,auto_reply:true}};
respond(requests[5],{ok:true,item:inherited});
await clearGroup;
assert.equal(saveStatus.textContent,'权限已保存。');
respond(requests[6],{ok:true,managed:true,items:[allowed,inherited],nextCursor:null});
await settle();
assert.equal(selected.id,group.id);
assert.equal(permissionMode.value,'inherit');
assert.equal(savePermission.disabled,true);
assert.equal(requests.filter(request=>request.options.method==='POST').length,3);
""")


def test_contact_automation_receipt_matches_exact_requested_flags():
    """取消托管会关闭私聊自动回复，回执中的权限必须与请求相同。"""
    _run_permissions_case(r"""
const allowed = {...contact,mode:'allowlist',automation:{managed:true,auto_reply:true}};
respond(requests[0],{ok:true,managed:true,items:[allowed],nextCursor:null});
await settle();
rowButton(0).onclick();
assert.equal(permissionAutoReply.checked,true);
permissionManaged.checked = false;
permissionManaged.onchange();
assert.equal(permissionAutoReply.disabled,true);
assert.equal(permissionAutoReply.checked,false);
const saving = saveSelectedPermission();
assert.deepEqual(JSON.parse(requests[1].options.body),
  {recipient_id:contact.id,mode:'allowlist',managed:false,auto_reply:false});
respond(requests[1],{ok:true,managed:true,recipient:allowed});
await saving;
assert.match(saveStatus.textContent,/无法核对已保存的权限/);
assert.equal(requests.length,2);

permissionMode.value = 'observe';
permissionMode.onchange();
assert.equal(permissionManaged.disabled,true);
assert.equal(permissionAutoReply.disabled,true);
""")


def test_inbound_only_contact_saves_fixed_automation_and_is_unavailable_for_groups():
    """仅托管只适用于联系人，能处理入站消息并阻止手动与自动外发。"""
    _run_permissions_case(r"""
const allowed = {...contact,mode:'allowlist',automation:{managed:true,auto_reply:true}};
const inboundOnly = {...contact,mode:'managed_only',automation:{managed:true,auto_reply:false}};
respond(requests[0],{ok:true,managed:true,items:[allowed,group],nextCursor:null});
await settle();
rowButton(0).onclick();
assert.equal(permissionManagedOnly.hidden,false);
assert.equal(permissionManagedOnly.disabled,false);
assert.match(permissionGuidance.textContent,/处理入站消息，禁止手动和自动外发/);
permissionMode.value = 'managed_only';
permissionMode.onchange();
assert.equal(permissionManaged.checked,true);
assert.equal(permissionManaged.disabled,true);
assert.equal(permissionAutoReply.checked,false);
assert.equal(permissionAutoReply.disabled,true);

permissionManaged.checked = false;
permissionAutoReply.checked = true;
const saving = saveSelectedPermission();
assert.equal(requests[1].url,'/api/wechat/contact-policy');
assert.deepEqual(JSON.parse(requests[1].options.body),
  {recipient_id:contact.id,mode:'managed_only',managed:true,auto_reply:false});
respond(requests[1],{ok:true,managed:true,recipient:inboundOnly});
await saving;
respond(requests[2],{ok:true,managed:true,items:[inboundOnly,group],nextCursor:null});
await settle();
assert.equal(saveStatus.textContent,'权限已保存。');
assert.equal(rowButton(0).children[1].textContent,'仅托管（禁止外发）');
assert.match(rowButton(0).children[2].textContent,/托管开启.*自动回复关闭/);
rowButton(0).onclick();
assert.equal(permissionMode.value,'managed_only');
assert.equal(savePermission.disabled,true);

rowButton(1).onclick();
assert.equal(permissionManagedOnly.hidden,true);
assert.equal(permissionManagedOnly.disabled,true);
permissionMode.value = 'managed_only';
await saveSelectedPermission();
assert.equal(requests.length,3);
""")


def test_saved_receipt_cannot_replace_a_new_contact_or_group_selection():
    """保存期间可浏览其他列表，迟到回执只触发重新读取。"""
    _run_permissions_case(r"""
respond(requests[0],{ok:true,managed:true,items:[group,contact],nextCursor:null});
await settle();
rowButton(0).onclick();
permissionMode.value = 'allowlist';
permissionMode.onchange();
const pending = saveSelectedPermission();
rowButton(1).onclick();
assert.equal(permissionName.textContent,'李师傅 · wxid_li');
assert.equal(savePermission.disabled,true);
respond(requests[1],{ok:true,item:{...group,mode:'allowlist',override:'allowlist'}});
await pending;
respond(requests[2],{ok:true,managed:true,items:[contact,group],nextCursor:null});
await settle();
assert.equal(selected.id,contact.id);
assert.equal(permissionName.textContent,'李师傅 · wxid_li');
assert.equal(saveStatus.textContent,'');
""")


def test_refresh_exact_configured_contact_keeps_permission_draft_and_labels_remark():
    """有权限的私聊可手动刷新，且备注与名称相同时仍明确显示。"""
    _run_permissions_case(r"""
const inboundOnly = {...contact,mode:'managed_only',automation:{managed:true,auto_reply:false}};
respond(requests[0],{ok:true,managed:true,items:[inboundOnly],nextCursor:null});
await settle();
rowButton(0).onclick();
assert.match(contactProfile.textContent,/备注：老李/);
assert.equal(refreshContactProfile.hidden,false);
permissionMode.value = 'allowlist';
permissionMode.onchange();
permissionAutoReply.checked = true;
permissionAutoReply.onchange();

const pending = refreshContactProfile.onclick();
assert.equal(refreshContactProfile.disabled,true);
assert.equal(savePermission.disabled,true);
assert.equal(requests[1].url,'/api/wechat/contact-policy/refresh');
assert.equal(requests[1].options.method,'POST');
assert.deepEqual(JSON.parse(requests[1].options.body),{recipient_id:contact.id});
respond(requests[1],{ok:false,pending:true},202);
await pending;
assert.match(profileStatus.textContent,/仍在同步/);
assert.equal(refreshContactProfile.disabled,false);

const refresh = refreshContactProfile.onclick();
respond(requests[2],{ok:true,projection:'complete',recipient:{id:contact.id,kind:'direct',
  displayName:'新备注',remark:'新备注',nickname:'李',alias:'li123',mode:'observe',automation:inactive}});
await refresh;
assert.equal(requests.length,3);
assert.equal(permissionName.textContent,'新备注 · wxid_li');
assert.equal(contactProfile.textContent,'备注：新备注 · 昵称：李 · 微信号：li123');
assert.match(rowButton(0).children[2].textContent,/备注：新备注.*昵称：李.*微信号：li123/);
assert.equal(rowButton(0).dataset.selected,'true');
assert.equal(selected.mode,'managed_only');
assert.deepEqual(selected.automation,{managed:true,auto_reply:false});
assert.equal(permissionMode.value,'allowlist');
assert.equal(permissionManaged.checked,true);
assert.equal(permissionAutoReply.checked,true);
assert.equal(savePermission.disabled,false);
assert.match(profileStatus.textContent,/已拉取并保存/);
""")


def test_refresh_respects_new_selection_and_reports_missing_remark():
    """旧刷新回执不能覆盖后来选择，群聊及未授权联系人无法请求刷新。"""
    _run_permissions_case(r"""
const observed = {...contact,mode:'observe'};
const allowed = {...contact,id:'wxid_other',displayName:'另一人',mode:'allowlist'};
const unassigned = {...contact,id:'wxid_unassigned'};
respond(requests[0],{ok:true,managed:true,items:[observed,allowed,unassigned,group],nextCursor:null});
await settle();
rowButton(2).onclick();
assert.equal(refreshContactProfile.hidden,true);
await refreshContactProfile.onclick();
rowButton(3).onclick();
assert.equal(refreshContactProfile.hidden,true);
await refreshContactProfile.onclick();
assert.equal(requests.length,1);

rowButton(0).onclick();
const previous = refreshContactProfile.onclick();
rowButton(1).onclick();
respond(requests[1],{ok:true,projection:'complete',recipient:{id:observed.id,kind:'direct',
  displayName:'迟到的新备注',remark:'迟到的新备注'}});
await previous;
assert.equal(permissionName.textContent,'另一人 · wxid_other');
assert.equal(profileStatus.textContent,'');
assert.equal(rowButton(0).children[0].textContent,observed.displayName);

const mismatch = refreshContactProfile.onclick();
respond(requests[2],{ok:true,projection:'complete',recipient:{id:observed.id,kind:'direct',displayName:'其他人'}});
await mismatch;
assert.match(profileStatus.textContent,/无法核对/);
assert.equal(permissionName.textContent,'另一人 · wxid_other');

const refresh = refreshContactProfile.onclick();
respond(requests[3],{ok:true,projection:'partial',recipient:{id:allowed.id,kind:'direct',
  displayName:'另一人',remark:null,nickname:'另一人',alias:null}});
await refresh;
assert.equal(contactProfile.textContent,'昵称：另一人');
assert.match(profileStatus.textContent,/部分同步.*未读取到微信备注/);
""")
