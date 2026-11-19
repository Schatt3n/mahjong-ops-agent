// 老板工作台仅加载运营投影和微信通道状态。调试数据由独立开发者页面按需读取。
let stateRefreshGeneration = 0;
let selectedWechatRecipient = null;
let wechatSearchTimer = null;
let wechatSearchRequest = null;
let wechatRecipientRefreshBusy = false;
let wechatSendBusy = false;
let wechatChannelState = {};
let wechatChannelRevision = 0;
let wechatSwitchBusy = false;
const pendingWechatSends = new Map();

function createMessageId(){
  if(globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function'){
    return `web_${globalThis.crypto.randomUUID()}`;
  }
  return `web_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}
// 定期刷新只加载运营所需字段，迟到的响应不会覆盖较新的汇总和列表。
async function loadState(){
  const generation = ++stateRefreshGeneration;
  const status = document.getElementById('stateSummaryStatus');
  try{
    const res = await fetch('/api/state?view=console', {cache:'no-store'});
    if(!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if(generation !== stateRefreshGeneration) return true;
    renderInviteDrafts(data.invite_drafts);
    renderGameSchedule(data.active_game_schedule);
    for(const count of document.querySelectorAll('[data-state-count]')){
      const key = count.dataset.stateCount;
      count.textContent = key === 'open_seats'
        ? data.active_game_schedule.reduce((total, game) => total + Number(game.seat_summary?.remaining_seats || 0), 0)
        : data.summary[key];
    }
    liveStatus.textContent = '更新于 ' + new Date().toLocaleTimeString('zh-CN');
    status.hidden = true;
    return true;
  }catch(error){
    if(generation !== stateRefreshGeneration) return true;
    status.textContent = `状态刷新失败（${error.message}）`;
    status.hidden = false;
    return false;
  }
}
function localDateKey(date){
  const y=date.getFullYear(); const m=String(date.getMonth()+1).padStart(2,'0'); const d=String(date.getDate()).padStart(2,'0');
  return `${y}-${m}-${d}`;
}
function scheduleDateLabel(key){
  const today=new Date(); const tomorrow=new Date(today); tomorrow.setDate(today.getDate()+1);
  if(key===localDateKey(today)) return '今天';
  if(key===localDateKey(tomorrow)) return '明天';
  return key;
}
function gameTypeLabel(requirement){
  const variant=String(requirement.variant||'').toLowerCase();
  if(variant.includes('caiqiao')||variant==='cq') return 'cq';
  const type=String(requirement.game_type||'').toLowerCase();
  if(type.includes('sichuan')) return '川麻';
  if(type.includes('red')||type.includes('hongzhong')) return '红中';
  if(type.includes('hangzhou')) return '杭麻';
  return requirement.game_type||'麻将';
}
function smokeLabel(value){
  const smoke=String(value||'').toLowerCase();
  if(smoke.includes('no_smok')||smoke==='无烟') return '无烟';
  if(smoke) return smoke.includes('allow')||smoke.includes('either') ? '烟都可' : String(value);
  return '';
}
function formatGameScheduleLine(game){
  const req=game.requirement||{}; const seats=game.seat_summary||{};
  const claimed=Number(seats.claimed_seats??0); const remaining=Number(seats.remaining_seats??game.remaining_seats??0);
  const code=remaining>0 ? `${claimed}7${remaining}` : '已满';
  const start=game.planned_start_at ? new Date(game.planned_start_at).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',hour12:false}) : '人齐开';
  const stake=req.stake_label||req.stake||req.base_stake||''; const smoke=smokeLabel(req.smoke_preference);
  return [gameTypeLabel(req)+code,start,smoke,stake?`${stake}块`:null].filter(Boolean).join('  ');
}
function renderGameSchedule(games){
  gameSchedule.replaceChildren();
  if(!games.length){
    const empty=document.createElement('div'); empty.className='status'; empty.textContent='暂无待组局。'; gameSchedule.appendChild(empty); return;
  }
  const groups=new Map();
  for(const game of games){
    const key=game.planned_start_at ? localDateKey(new Date(game.planned_start_at)) : localDateKey(new Date());
    if(!groups.has(key)) groups.set(key,[]); groups.get(key).push(game);
  }
  for(const [key,items] of groups){
    const section=document.createElement('section'); section.className='schedule-day';
    const title=document.createElement('h3'); title.textContent=scheduleDateLabel(key); section.appendChild(title);
    for(const game of items){
      const line=document.createElement('div'); line.className='schedule-line'; line.textContent=formatGameScheduleLine(game);
      const status=document.createElement('span'); status.className='schedule-status';
      status.textContent=game.recruitment_status==='scheduled' ? `将在 ${new Date(game.recruitment_opens_at).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',hour12:false})} 开始找人` : '可开始找人';
      line.appendChild(status); section.appendChild(line);
    }
    gameSchedule.appendChild(section);
  }
}
function renderInviteDrafts(drafts){
  inviteDrafts.replaceChildren();
  const pending = drafts.filter(item => item.status === 'pending_approval');
  if(!pending.length){
    const empty = document.createElement('div');
    empty.className = 'status';
    empty.textContent = '暂无待审批邀约。';
    inviteDrafts.appendChild(empty);
    return;
  }
  for(const draft of pending){
    const item = document.createElement('div');
    item.className = 'draft-item';
    const meta = document.createElement('div');
    meta.className = 'draft-meta';
    meta.textContent = `${draft.display_name || draft.customer_id} · ${draft.game_id} · ${draft.metadata?.content_review_approved ? '已过对外内容审查' : '未过对外内容审查'}`;
    const message = document.createElement('div');
    message.className = 'draft-text';
    message.textContent = draft.message_text;
    const approve = document.createElement('button');
    approve.textContent = '通过并发送';
    approve.disabled = draft.metadata?.content_review_approved !== true;
    approve.onclick = () => decideInviteDraft(draft.draft_id, 'approve_send');
    const reject = document.createElement('button');
    reject.className = 'danger';
    reject.textContent = '拒绝';
    reject.onclick = () => decideInviteDraft(draft.draft_id, 'reject');
    item.append(meta, message, approve, document.createTextNode(' '), reject);
    inviteDrafts.appendChild(item);
  }
}
async function decideInviteDraft(draftId, action){
  const verb = action === 'approve_send' ? '发送这条邀约' : '拒绝这条邀约';
  if(!confirm(`确认${verb}？`)) return;
  inviteDraftOutput.textContent = '正在处理邀约…';
  try{
    const res = await fetch('/api/invite-drafts/action',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({draft_id:draftId, action})
    });
    const data = await res.json();
    if(res.ok && data.ok && action === 'reject'){
      inviteDraftOutput.textContent = '已拒绝这条邀约。';
    }else if(res.ok && data.ok && data.sent === true){
      inviteDraftOutput.textContent = '邀约已发送。';
    }else{
      inviteDraftOutput.textContent = data.message || data.error || '暂未确认邀约处理结果。';
    }
    await loadState();
  }catch{
    inviteDraftOutput.textContent = '暂未收到处理回执，请刷新邀约。';
  }
}
// 搜索与选择分开管理。改动输入会立即作废已选目标和所有在途候选。
function cancelWechatRecipientSearch(){
  clearTimeout(wechatSearchTimer);
  wechatSearchTimer = null;
  wechatSearchRequest?.abort();
  wechatSearchRequest = null;
  wechatRecipientCandidates.replaceChildren();
  wechatRecipientCandidates.hidden = true;
}
function queueWechatRecipientSearch(event){
  cancelWechatRecipientSearch();
  selectedWechatRecipient = null;
  updateWechatSendControl();
  wechatRecipientStatus.textContent = '';
  const query = wechatTarget.value.trim();
  // 中文输入法仍在组字时没有完整搜索词，提交选字后才查。
  if(!query || event?.isComposing) return;
  wechatRecipientStatus.textContent = '正在查找收件人…';
  const request = new AbortController();
  wechatSearchRequest = request;
  wechatSearchTimer = setTimeout(() => void searchWechatRecipients(query, request.signal), 250);
}
async function searchWechatRecipients(query, signal){
  try{
    const params = new URLSearchParams({query});
    const res = await fetch(`/api/wechat/recipients?${params}`, {cache:'no-store', signal});
    const data = await res.json();
    // 即使取消发生在 JSON 解析期间，迟到的结果也不能重新打开候选。
    if(signal.aborted) return;
    if(!res.ok || data.ok !== true){
      wechatRecipientStatus.textContent = '查找收件人失败。' + wechatError(data, res.status);
      return;
    }
    const candidates = data.candidates || [];
    for(const recipient of candidates){
      const row = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = recipient.displayName || recipient.id;
      const details = document.createElement('span');
      const distinctions = new Set([
        recipient.kind === 'group' ? '群聊' : '联系人',
        recipient.remark, recipient.nickname, recipient.alias, recipient.id
      ].filter(value => value && value !== button.textContent));
      details.textContent = [...distinctions].join(' · ');
      button.append(details);
      button.onclick = () => selectWechatRecipient(recipient);
      row.append(button);
      wechatRecipientCandidates.append(row);
    }
    wechatRecipientCandidates.hidden = candidates.length === 0;
    wechatRecipientStatus.textContent = candidates.length ? '请选择收件人。' : '没有找到匹配的收件人。';
  }catch{
    if(!signal.aborted) wechatRecipientStatus.textContent = '无法查找收件人，请稍后再试。';
  }
}
// 候选是普通按钮。Tab 可访问，方向键可从搜索框进入并在结果内切换。
function navigateWechatRecipients(event){
  if(event.isComposing) return;
  if(event.key === 'Escape' && !selectedWechatRecipient){
    event.preventDefault();
    cancelWechatRecipientSearch();
    wechatRecipientStatus.textContent = '';
    wechatTarget.focus();
    return;
  }
  if(wechatRecipientCandidates.hidden || !['ArrowDown','ArrowUp'].includes(event.key)) return;
  const buttons = [...wechatRecipientCandidates.querySelectorAll('button')];
  if(!buttons.length) return;
  event.preventDefault();
  const index = buttons.indexOf(event.target);
  const step = event.key === 'ArrowDown' ? 1 : -1;
  if(index < 0){
    buttons[step === 1 ? 0 : buttons.length - 1].focus();
    return;
  }
  buttons[(index + step + buttons.length) % buttons.length].focus();
}
function dismissWechatRecipients(event){
  // 某些浏览器点击候选时不会转移焦点，保留列表直到点击动作完成。
  if(selectedWechatRecipient || !event.relatedTarget || event.currentTarget.contains(event.relatedTarget)) return;
  cancelWechatRecipientSearch();
  wechatRecipientStatus.textContent = '';
}
function selectWechatRecipient(recipient){
  selectedWechatRecipient = {id:recipient.id, kind:recipient.kind, name:recipient.displayName || recipient.id};
  cancelWechatRecipientSearch();
  wechatTarget.value = selectedWechatRecipient.name;
  showSelectedWechatRecipient();
  wechatText.focus();
}
function showSelectedWechatRecipient(feedback = ''){
  const {name, id} = selectedWechatRecipient;
  const selection = name === id ? `已选中 ${id}` : `已选中 ${name} · ${id}`;
  wechatRecipientStatus.textContent = feedback ? `${selection}。${feedback}` : selection;
  updateWechatSendControl();
}
// 只拉取明确选中的个人联系人。切换选择后守住当前页面，忽略旧选择的回执。
async function refreshWechatRecipient(){
  const selected = selectedWechatRecipient;
  if(wechatRecipientRefreshBusy || selected?.kind !== 'direct' || wechatTarget.value !== selected.name) return;
  wechatRecipientRefreshBusy = true;
  showSelectedWechatRecipient('正在拉取联系人资料…');
  try{
    const res = await fetch('/api/wechat/recipients/refresh', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({recipient_id:selected.id})
    });
    const data = await res.json();
    if(selectedWechatRecipient !== selected || wechatTarget.value !== selected.name) return;
    if(res.status === 202 && data.pending === true){
      showSelectedWechatRecipient(data.message || '微信仍在刷新资料，请稍后重试。');
      return;
    }
    if(!res.ok || data.ok !== true){
      showSelectedWechatRecipient('刷新联系人资料失败。' + wechatError(data, res.status));
      return;
    }
    const recipient = data.recipient;
    if(recipient?.id !== selected.id || recipient.kind !== 'direct'){
      showSelectedWechatRecipient('刷新回执与所选联系人不一致。');
      return;
    }
    selectedWechatRecipient = {...selected, name:recipient.displayName || selected.name};
    wechatTarget.value = selectedWechatRecipient.name;
    const feedback = data.projection === 'partial' ? '微信这次只完成了部分同步，当前显示本机已保存的资料。'
      : recipient.remark ? '已拉取并保存联系人资料。' : '资料已刷新，这次没有返回微信备注。';
    showSelectedWechatRecipient(feedback);
  }catch{
    if(selectedWechatRecipient === selected && wechatTarget.value === selected.name){
      showSelectedWechatRecipient('暂时无法确认联系人资料刷新结果。');
    }
  }finally{
    wechatRecipientRefreshBusy = false;
    updateWechatSendControl();
  }
}
// 修改收件人立即作废原选择。拉取资料与人工确认发送各自等待回执。
function updateWechatSendControl(){
  const recipient = selectedWechatRecipient;
  const selectionMatches = recipient && wechatTarget.value === recipient.name;
  const directSelected = selectionMatches && recipient.kind === 'direct';
  wechatRecipientRefreshButton.hidden = !directSelected;
  wechatRecipientRefreshButton.disabled = !directSelected || wechatRecipientRefreshBusy;
  wechatSendButton.disabled = wechatSendBusy || !selectionMatches || !wechatText.value.trim();
}
function wechatError(data, status){
  return data.message || data.error || `HTTP ${status}`;
}
function wechatSwitchLabel(enabled){
  if(enabled === true) return '开启';
  if(enabled === false) return '暂停';
  return '未知';
}
function renderWechatChannelState(update){
  wechatChannelState = {...wechatChannelState, ...update};
  const state = wechatChannelState;
  const status = [
    `发送总闸：${wechatSwitchLabel(state.send_channel_enabled)}`,
    `自动外发：${wechatSwitchLabel(state.auto_send_reply)}`
  ];
  if(state.direct_contact_policy_enabled){
    status.push('私聊外发按名单管理。白名单内的手动发送需要开启总闸，自动外发还需要开启自动外发开关。');
  }else if(state.outbound_recipient_policy_enabled){
    status.push('已启用外发白名单。名单内的手动发送只需开启总闸。');
  }else if(state.adapter === 'agent-wechat'){
    status.push('当前未启用私聊白名单。未单独授权的手动发送也需要开启自动外发。');
  }
  wechatChannelStatus.textContent = status.join(' · ');
}
async function loadWechatChannelStatus(){
  if(wechatSwitchBusy) return;
  const revision = ++wechatChannelRevision;
  try{
    const res = await fetch('/api/wechat/health', {cache:'no-store'});
    const data = await res.json();
    if(revision !== wechatChannelRevision) return;
    if(!res.ok || !data.ok){
      wechatChannelStatus.textContent = '微信发送通道不可用。' + wechatError(data, res.status);
      return;
    }
    renderWechatChannelState({
      ...data,
      direct_contact_policy_enabled:data.direct_contact_policy_enabled === true,
      outbound_recipient_policy_enabled:data.outbound_recipient_policy_enabled === true
    });
  }catch{
    if(revision === wechatChannelRevision) wechatChannelStatus.textContent = '微信发送通道不可用。';
  }
}
async function setWechatSwitch(path, enabled, question){
  if(wechatSwitchBusy || !confirm(question)) return;
  wechatSwitchBusy = true;
  ++wechatChannelRevision;
  try{
    const res = await fetch(path,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({enabled})
    });
    const data = await res.json();
    if(res.ok && data.ok){
      renderWechatChannelState(data);
    }else{
      wechatChannelStatus.textContent = '切换失败。' + wechatError(data, res.status);
    }
  }catch{
    wechatChannelStatus.textContent = '暂时无法确认开关状态。';
  }finally{
    wechatSwitchBusy = false;
  }
}
function setWechatAutoSend(enabled){
  return setWechatSwitch('/api/wechat/auto-send', enabled, enabled
    ? '确认开启自动外发？消息仍受发送总闸和收件人权限限制。'
    : '确认暂停自动外发？');
}
function setWechatSendChannel(enabled){
  return setWechatSwitch('/api/wechat/send-channel', enabled, enabled
    ? '确认开启微信发送总闸？消息仍受收件人配置限制。'
    : '确认暂停微信发送总闸？暂停后，手动发送和自动外发都会被挡住。');
}
// 确认后绑定精确会话和正文。回执不明时，手动重试同一内容仍沿用原请求编号。
async function sendWechatText(){
  const recipient = selectedWechatRecipient;
  const text = wechatText.value;
  if(wechatSendBusy || !recipient || wechatTarget.value !== recipient.name || !text.trim()) return;
  const kind = recipient.kind === 'group' ? '群聊' : '联系人';
  if(!confirm(`确认发给${kind}「${recipient.name}」（${recipient.id}）？

${text}`)) return;
  const signature = JSON.stringify([recipient.id, recipient.kind, text]);
  if(!pendingWechatSends.has(signature)) pendingWechatSends.set(signature, createMessageId());
  const payload = {
    recipient_id:recipient.id, kind:recipient.kind, text, request_id:pendingWechatSends.get(signature)
  };
  wechatSendBusy = true;
  updateWechatSendControl();
  wechatSendOutput.textContent = '正在等待发送回执…';
  try{
    const res = await fetch('/api/wechat/send',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    const data = await res.json();
    if(res.ok && data.ok === true && data.sent === true){
      pendingWechatSends.delete(signature);
      const context = data.context_recorded ? '已记录到会话上下文。' : '';
      wechatSendOutput.textContent = `已发送给「${recipient.name}」。${context}`;
    }else{
      const heading = data.delivery_uncertain ? '发送结果待核对。' : '未发送。';
      wechatSendOutput.textContent = heading + wechatError(data, res.status);
    }
  }catch{
    wechatSendOutput.textContent = '暂未收到发送回执。请先在微信核对。';
  }finally{
    wechatSendBusy = false;
    updateWechatSendControl();
  }
}

// 开发者仅在明确交接时存下草稿，进入工作台立即取出，不绑定收件人或发送。
function restoreTrialDraft(){
  try{
    const draft = sessionStorage.getItem('mahjong.owner.wechat-draft');
    if(!draft) return;
    sessionStorage.removeItem('mahjong.owner.wechat-draft');
    if(!wechatText.value){
      wechatText.value = draft;
      updateWechatSendControl();
      wechatText.focus({preventScroll:true});
    }
  }catch{
    // 关闭浏览器会话存储时，仍可手动输入正文。
  }
}
function initOwnerWorkspace(){
  // Chromium 的中文选字完成事件通过事件监听触发，保留精确候选确认。
  wechatTarget.addEventListener('input', queueWechatRecipientSearch);
  wechatTarget.addEventListener('compositionend', () => queueWechatRecipientSearch());
  restoreTrialDraft();
  void loadWechatChannelStatus();
  void loadState();
  setInterval(() => {
    if(!document.hidden) void loadState();
  }, 6000);
}
initOwnerWorkspace();
