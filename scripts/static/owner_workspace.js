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
const wechatDrafts = new Map(), wechatSendReceipts = new Map();
let unboundWechatDraft = '';
let recentWechatRequest = null, recentWechatSignature = null;
let wechatHistoryRequest = null, wechatHistoryItems = [], wechatHistoryCursor = null, wechatHistoryExpanded = false;
let wechatMediaRequest = null, wechatMediaObjectUrl = null;
let wechatWorkScope = '', wechatWorkId = '', wechatWorkRequest = null, wechatWorkTimer = null;
let wechatWorkDiagnostics = false, wechatWorkSignature = '', wechatWorkClockOffset = 0;

function wechatRecipientKey(recipient){
  return recipient ? JSON.stringify([recipient.kind,recipient.id,recipient.agent_wechat_account,
    recipient.agent_wechat_revision,recipient.work_scope]) : '';
}
function formatWechatTime(value){
  const date = new Date(value || '');
  if(!Number.isFinite(date.getTime())) return '';
  const time = date.toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'});
  return localDateKey(date) === localDateKey(new Date()) ? time : `${date.toLocaleDateString('zh-CN')} ${time}`;
}

// 只显示后端投影的活跃会话。轮询结果不变时保留列表焦点和滚动位置。
async function loadWechatConversations({quiet = false} = {}){
  if(recentWechatRequest) return;
  const request = new AbortController();
  recentWechatRequest = request;
  if(!quiet) wechatListStatus.textContent = '正在读取最近会话。';
  try{
    const res = await fetch('/api/wechat/conversations',{cache:'no-store',signal:request.signal});
    const data = await res.json();
    if(res.status === 501 && data.error === 'wechat_history_unsupported'){
      recentWechatSignature = null;
      wechatConversationList.replaceChildren();
      wechatListStatus.textContent = '当前使用备用桥接。';
      const link = wechatWorkNode('a','查看微信入站和处理记录','status');
      link.href = '/agent-work';
      wechatListStatus.append(link);
      return;
    }
    if(res.status === 401 || res.status === 403){
      recentWechatSignature = null;
      wechatConversationList.replaceChildren();
    }
    if(!res.ok || data.ok !== true || !Array.isArray(data.items)) throw new Error(wechatError(data,res.status));
    const signature = JSON.stringify(data.items);
    if(signature !== recentWechatSignature){
      recentWechatSignature = signature;
      wechatConversationList.replaceChildren();
      for(const row of data.items){
        const item = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.key = wechatRecipientKey(row);
        button.textContent = row.displayName || row.id;
        const details = document.createElement('span');
        details.textContent = [row.kind === 'group' ? '群聊' : '私聊',formatWechatTime(row.last_at),
          row.pending_tasks > 0 ? `${row.pending_tasks} 项待处理` : '',row.displayName !== row.id && row.id]
          .filter(Boolean).join(' · ');
        button.append(details);
        button.onclick = () => selectWechatRecipient(row,{focusComposer:false});
        item.append(button);
        wechatConversationList.append(item);
      }
      syncWechatConversationSelection();
    }
    wechatListStatus.textContent = data.partial === true ? '部分会话暂时无法读取，正在重试。'
      : data.items.length ? '' : '近两小时暂无活动会话，也没有会话待处理。';
  }catch{
    wechatListStatus.textContent = '暂时无法读取最近会话，请稍后刷新。';
  }finally{
    recentWechatRequest = null;
  }
}
function syncWechatConversationSelection(){
  const key = wechatRecipientKey(selectedWechatRecipient);
  for(const button of wechatConversationList.querySelectorAll('button')){
    button.setAttribute('aria-pressed',String(button.dataset.key === key));
  }
}
function clearWechatHistory(){
  wechatHistoryRequest?.abort();
  wechatHistoryRequest = null;
  wechatHistoryItems = [];
  wechatHistoryCursor = null;
  wechatHistoryExpanded = false;
  wechatWorkScope = '';
  clearWechatWork();
  wechatHistoryList.replaceChildren();
  wechatHistoryTitle.textContent = '聊天记录';
  wechatHistoryRefresh.hidden = true;
  wechatHistoryOlder.hidden = true;
  wechatHistoryStatus.textContent = '选择会话后查看聊天记录。';
}

// 只有服务端核实的群消息作者才能作为精确私聊目标，旧消息按钮随当前群聊失效。
function addWechatPrivateAction(info,row,message,conversation){
  const recipient = message.direct_recipient;
  if(conversation?.kind !== 'group' || message.direction !== 'inbound' || message.author !== 'contact'
      || recipient?.kind !== 'direct' || typeof recipient.id !== 'string' || !recipient.id.trim()) return;
  const groupKey = wechatRecipientKey(conversation);
  const name = recipient.displayName || recipient.id;
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'secondary wechat-private-action';
  button.textContent = '私聊';
  button.setAttribute('aria-label',`私聊 ${name}（${recipient.id}）`);
  button.onclick = () => {
    if(row.parentNode !== wechatHistoryList || wechatRecipientKey(selectedWechatRecipient) !== groupKey) return;
    selectWechatRecipient({...recipient,displayName:name});
  };
  info.append(button);
}
function wechatMediaUrl(conversation, message, variant){
  const params = new URLSearchParams({
    thread_id:conversation.id,kind:conversation.kind,message_id:message.id,variant,
  });
  return `/api/wechat/media?${params}`;
}
function clearWechatMedia(){
  wechatMediaRequest?.abort();
  wechatMediaRequest = null;
  wechatMediaImage.hidden = true;
  wechatMediaImage.onerror = null;
  wechatMediaImage.removeAttribute('src');
  wechatMediaDownload.hidden = true;
  wechatMediaDownload.removeAttribute('href');
  if(wechatMediaObjectUrl) URL.revokeObjectURL(wechatMediaObjectUrl);
  wechatMediaObjectUrl = null;
}
async function openWechatMedia(conversation,message,label){
  clearWechatMedia();
  const request = new AbortController();
  wechatMediaRequest = request;
  wechatMediaTitle.textContent = label + '原件';
  wechatMediaStatus.textContent = '正在加载原件…';
  if(!wechatMediaViewer.open) wechatMediaViewer.showModal();
  try {
    const response = await fetch(wechatMediaUrl(conversation,message,'original'),{cache:'no-store',signal:request.signal});
    if(!response.ok){
      const data = await response.json().catch(() => ({}));
      if(wechatMediaRequest !== request) return;
      wechatMediaStatus.textContent = data.reason === 'sticker_source_not_allowed' ? '这条表情包的来源暂不支持。'
        : response.status === 404 || response.status === 410 ? '原件暂不可用，可能已过期或撤回。'
        : '读取原件失败，请稍后重试。';
      return;
    }
    const blob = await response.blob();
    if(wechatMediaRequest !== request) return;
    const extensions = {'image/png':'png','image/jpeg':'jpg','image/gif':'gif','image/webp':'webp',
      'image/x-wxgf':'wxgf','application/octet-stream':'bin'};
    const extension = extensions[blob.type];
    if(!extension || response.headers.get('X-Mahjong-Media-Variant') !== 'original'){
      wechatMediaStatus.textContent = '微信没有提供可验证的原件。';
      return;
    }
    wechatMediaObjectUrl = URL.createObjectURL(blob);
    wechatMediaDownload.href = wechatMediaObjectUrl;
    wechatMediaDownload.download = `wechat-original.${extension}`;
    wechatMediaDownload.hidden = false;
    if(!['wxgf','bin'].includes(extension)){
      wechatMediaStatus.textContent = '';
      wechatMediaImage.alt = label + '原件';
      wechatMediaImage.onerror = () => {
        if(wechatMediaRequest !== request) return;
        wechatMediaImage.hidden = true;
        wechatMediaStatus.textContent = '浏览器无法显示此原件，可以下载查看。';
      };
      wechatMediaImage.src = wechatMediaObjectUrl;
      wechatMediaImage.hidden = false;
    }else{
      wechatMediaStatus.textContent = '微信原格式，可下载查看。';
    }
  }catch(error){
    if(wechatMediaRequest === request) wechatMediaStatus.textContent = '读取原件失败，请稍后重试。';
  }
}
wechatMediaViewer.addEventListener('close',clearWechatMedia);
function renderWechatMedia(conversation,message){
  const label = {image:'图片',sticker:'表情包'}[message.media?.kind];
  if(!conversation || !label) return null;
  const attachment = document.createElement('div');
  attachment.className = 'wechat-attachment';
  const image = document.createElement('img');
  image.loading = 'lazy';
  image.alt = label + '预览';
  image.onerror = () => { image.hidden = true; };
  image.src = wechatMediaUrl(conversation,message,'preview');
  const open = document.createElement('button');
  open.type = 'button';
  open.className = 'secondary';
  open.textContent = '查看原件';
  open.onclick = () => openWechatMedia(conversation,message,label);
  attachment.append(image,open);
  const vision = message.media.vision;
  if(vision?.description || vision?.visible_text){
    const note = document.createElement('p');
    note.className = 'wechat-visual-note';
    note.textContent = ['Agent 识别',vision.description,vision.visible_text].filter(Boolean).join('\n');
    attachment.append(note);
  }
  return attachment;
}
function renderWechatHistory(items, older){
  const conversation = selectedWechatRecipient;
  const height = wechatHistoryWrap.scrollHeight;
  const offset = wechatHistoryWrap.scrollTop;
  const atBottom = height - offset - wechatHistoryWrap.clientHeight < 48;
  const hadMessages = wechatHistoryList.childElementCount > 0;
  const focusedWork = document.activeElement?.dataset?.workId;
  wechatHistoryList.replaceChildren();
  for(const message of items){
    const row = document.createElement('li');
    row.dataset.direction = message.direction;
    const info = document.createElement('span');
    const author = {agent:'Agent',owner:'老板',outbound:'微信外发'}[message.author];
    info.className = 'wechat-message-meta';
    info.textContent = [author || message.speaker || '对方',formatWechatTime(message.at)].filter(Boolean).join(' · ');
    addWechatPrivateAction(info,row,message,conversation);
    const body = document.createElement('p');
    body.className = 'wechat-message-body';
    body.textContent = message.text;
    const attachment = renderWechatMedia(conversation,message);
    if(attachment) body.append(attachment);
    if(wechatWorkScope && message.direction === 'inbound'){
      const inspect = document.createElement('button');
      inspect.type = 'button';
      inspect.className = 'work-message';
      inspect.dataset.workId = message.id;
      inspect.textContent = '查看处理';
      inspect.setAttribute('aria-label','查看处理：'+(message.text?.slice(0,42) || '微信消息'));
      inspect.setAttribute('aria-pressed',String(wechatWorkId === message.id));
      inspect.addEventListener('click',() => {
        if(row.parentNode === wechatHistoryList && wechatRecipientKey(conversation) === wechatRecipientKey(selectedWechatRecipient)){
          selectWechatWork(message.id);
        }
      });
      body.append(inspect);
    }
    row.append(info,body);
    wechatHistoryList.append(row);
  }
  if(older) wechatHistoryWrap.scrollTop = offset + wechatHistoryWrap.scrollHeight - height;
  else if(atBottom || !hadMessages) wechatHistoryWrap.scrollTop = wechatHistoryWrap.scrollHeight;
  if(focusedWork) Array.from(wechatHistoryList.querySelectorAll('[data-work-id]'))
    .find(button => button.dataset.workId === focusedWork)?.focus({preventScroll:true});
}

// 切换会话或账号立即丢弃旧查询，较慢的回执也不能覆盖当前面板。
function clearWechatWork(){
  if(!wechatWorkId) return;
  wechatWorkRequest?.abort();
  wechatWorkRequest = null;
  clearTimeout(wechatWorkTimer);
  wechatWorkId = '';
  wechatWorkDiagnostics = false;
  wechatWorkSignature = '';
  wechatWorkLive.textContent = '';
  wechatWorkContent.replaceChildren(wechatWorkNode('p','点击一条收到的消息下方的「查看处理」。','status'));
}

function wechatWorkNode(tag,text,className){
  const node = document.createElement(tag);
  node.textContent = text;
  if(className) node.className = className;
  return node;
}

function selectWechatWork(id){
  if(wechatWorkId !== id){
    clearWechatWork();
    wechatWorkId = id;
    wechatWorkContent.replaceChildren(wechatWorkNode('p','正在查询这条消息的处理记录…','status'));
    for(const button of wechatHistoryList.querySelectorAll('[data-work-id]')){
      button.setAttribute('aria-pressed',String(button.dataset.workId === id));
    }
    void refreshWechatWork();
  }
  wechatWorkTitle.focus({preventScroll:true});
  wechatWorkPane.scrollIntoView({block:'nearest'});
}

function renderWechatWork(detail){
  const signature = JSON.stringify([detail,wechatWorkDiagnostics]);
  if(signature === wechatWorkSignature) return;
  wechatWorkSignature = signature;
  if(!detail){
    const link = wechatWorkNode('a','查看微信监听状态','status');
    link.href = '/agent-work';
    wechatWorkContent.replaceChildren(wechatWorkNode('p','暂无 Agent 入站记录。','work-status'),link);
    return;
  }
  const scroll = wechatWorkPane.scrollTop;
  const expanded = new Set(Array.from(wechatWorkContent.querySelectorAll('details[open]'), item => item.dataset.eventKey));
  const status = wechatWorkNode('strong',detail.phase,'work-status');
  status.dataset.tone = detail.tone;
  const source = wechatWorkNode('p',[detail.kind,detail.sender,formatWechatTime(detail.time)].filter(Boolean).join(' · '),'status');
  const reply = detail.deliveries.reply;
  const replySection = wechatWorkNode('section','','work-reply');
  replySection.append(wechatWorkNode('h4',detail.reply ? '业务回复' : detail.draft ? '候选回复' : '业务回复'),
    wechatWorkNode('p',detail.reply || detail.draft || '暂无回复正文。'));
  if(detail.ack) replySection.append(wechatWorkNode('p','即时确认 · '+detail.ack,'status'));
  if(reply){
    const receiptStatus = [reply.label === detail.phase ? '' : reply.label,reply.detail].filter(Boolean).join(' · ');
    if(receiptStatus) replySection.append(wechatWorkNode('p',receiptStatus,'status'));
    if(reply.message_id) replySection.append(wechatWorkNode('p','微信回执 '+reply.message_id,'status'));
    if(reply.status === 'blocked'){
      const permissions = wechatWorkNode('a','查看外发与会话权限','status');
      permissions.href = '#settings';
      replySection.append(permissions);
    }
  }else if(detail.reply){
    replySection.append(wechatWorkNode('p','暂无微信发送回执。','status'));
  }
  if(detail.game_id){
    const game = wechatWorkNode('a','查看关联牌局','status');
    game.href = '/agent-work#game-'+encodeURIComponent(detail.game_id);
    replySection.append(game);
  }
  const timeline = wechatWorkNode('ol','','work-timeline');
  for(const event of detail.timeline){
    const row = wechatWorkNode('li','');
    row.append(wechatWorkNode('strong',event.label),wechatWorkNode('p',
      [formatWechatTime(event.time),event.detail].filter(Boolean).join(' · '),'status'));
    if(event.model_io){
      const panel = wechatWorkNode('details','');
      panel.dataset.eventKey = event.id;
      panel.open = expanded.has(event.id);
      panel.append(wechatWorkNode('summary',event.model_io.kind === 'request' ? '模型输入' : '模型原始返回'),
        wechatWorkNode('pre',event.model_io.body || '（正文为空）'));
      row.append(panel);
    }
    timeline.append(row);
  }
  const diagnostics = wechatWorkNode('button',wechatWorkDiagnostics ? '收起诊断' : '查看模型输入与返回','secondary');
  diagnostics.type = 'button';
  diagnostics.setAttribute('aria-expanded',String(wechatWorkDiagnostics));
  diagnostics.addEventListener('click',() => {
    wechatWorkDiagnostics = !wechatWorkDiagnostics;
    if(!wechatWorkDiagnostics) renderWechatWork({...detail,timeline:detail.timeline.filter(event => !event.model_io)});
    void refreshWechatWork();
  });
  const current = detail.tone === 'working' ? detail.timeline.at(-1)?.detail : '';
  if(current) source.append(wechatWorkNode('span',' · '+current));
  const restoreFocus = document.activeElement?.parentNode === wechatWorkContent &&
    document.activeElement?.tagName === 'BUTTON';
  wechatWorkContent.replaceChildren(status,source,replySection,wechatWorkNode('h4','处理时间线'),
    timeline.childElementCount ? timeline : wechatWorkNode('p','暂无可用的执行事件。','status'),diagnostics);
  if(wechatWorkDiagnostics){
    if(!detail.timeline.some(event => event.model_io)){
      wechatWorkContent.append(wechatWorkNode('p','当前日志窗口暂无模型正文记录。','status'));
    }
    wechatWorkContent.append(wechatWorkNode('p','Trace '+detail.trace_ids.join(' · '),'status'));
  }
  if(restoreFocus) diagnostics.focus({preventScroll:true});
  wechatWorkPane.scrollTop = scroll;
}

// 只轮询当前选中的消息，使用服务端记录的实际 HTTP 单次预算判断等待状态。
async function refreshWechatWork(){
  clearTimeout(wechatWorkTimer);
  wechatWorkRequest?.abort();
  wechatWorkRequest = null;
  const recipient = selectedWechatRecipient;
  if(!recipient || !wechatWorkId || document.hidden || document.getElementById('messages').hidden) return;
  const request = new AbortController();
  wechatWorkRequest = request;
  // 同时限制收到响应和读取正文；切换会话时仍由独立控制器立即取消。
  const timeout = AbortSignal.timeout(6500);
  const signal = AbortSignal.any([request.signal,timeout]);
  const params = new URLSearchParams({thread_id:recipient.id,kind:recipient.kind,
    message_id:wechatWorkId,scope:wechatWorkScope});
  if(wechatWorkDiagnostics) params.set('model_io','1');
  try{
    const response = await fetch('/api/wechat/message-work?'+params,{cache:'no-store',signal});
    if(wechatWorkRequest !== request) return;
    if([401,403,404].includes(response.status)){
      // 处理查询也会复核授权；已经加载过旧页时，历史列表不会自行发现权限变化。
      invalidateWechatRecipient();
      wechatWorkLive.textContent = '当前无法查看处理详情。';
      return;
    }
    if(!response.ok) throw new Error('读取处理记录失败');
    const data = await response.json();
    if(wechatWorkRequest !== request) return;
    const clock = Date.parse(data.server_time);
    if(Number.isFinite(clock)) wechatWorkClockOffset = clock-Date.now();
    const detail = data.detail;
    const started = Date.parse(detail?.call_started_at);
    const elapsed = Math.max(0,Math.floor((Date.now()+wechatWorkClockOffset-started)/1000));
    const late = detail?.attempt_timeout_seconds &&
      Date.now()+wechatWorkClockOffset-started >= detail.attempt_timeout_seconds*1000;
    const timing = !Number.isFinite(elapsed) ? '' : late
      ? `已等待 ${elapsed} 秒，超过本次请求预算，尚无结束记录。`
      : `当前阶段已等待 ${elapsed} 秒`;
    if(wechatWorkLive.textContent !== timing) wechatWorkLive.textContent = timing;
    renderWechatWork(detail);
  }catch{
    if(wechatWorkRequest === request) wechatWorkLive.textContent = timeout.aborted
      ? '处理记录刷新超时，当前内容可能已过期，正在重试。'
      : '处理记录暂时无法刷新，当前内容可能已过期，正在重试。';
  }finally{
    if(wechatWorkRequest === request){
      wechatWorkRequest = null;
      const tone = wechatWorkContent.firstElementChild?.dataset?.tone;
      wechatWorkTimer = setTimeout(refreshWechatWork,!tone || tone === 'working' ? 1600 : 6000);
    }
  }
}

// 按精确会话和请求核对回执。翻页时保留此前消息，自动刷新只更新未翻页的最新窗口。
async function loadWechatConversation({older = false,quiet = false} = {}){
  const recipient = selectedWechatRecipient;
  const cursor = older ? wechatHistoryCursor : null;
  if(!recipient || (older && !cursor) || (quiet && wechatHistoryExpanded)) return;
  if(wechatHistoryRequest && (quiet || older)) return;
  wechatHistoryRequest?.abort();
  const request = new AbortController();
  wechatHistoryRequest = request;
  const sameRequest = () => wechatHistoryRequest === request && wechatRecipientKey(selectedWechatRecipient) === wechatRecipientKey(recipient);
  wechatHistoryOlder.disabled = true;
  if(!quiet) wechatHistoryStatus.textContent = '正在读取聊天记录。';
  const params = new URLSearchParams({thread_id:recipient.id,kind:recipient.kind,limit:'40'});
  if(cursor) params.set('cursor',cursor);
  try{
    const res = await fetch(`/api/wechat/conversation?${params}`,{cache:'no-store',signal:request.signal});
    const data = await res.json();
    if(!sameRequest()) return;
    if(!res.ok && [401,403,404].includes(res.status)){
      clearWechatWork();
      wechatWorkScope = '';
      wechatHistoryItems = [];
      wechatHistoryCursor = null;
      wechatHistoryExpanded = false;
      wechatHistoryList.replaceChildren();
      wechatHistoryOlder.hidden = true;
      wechatHistoryStatus.textContent = '当前没有可查看的聊天记录。';
      return;
    }
    if(!res.ok || data.ok !== true || data.id !== recipient.id || data.kind !== recipient.kind || !Array.isArray(data.items)){
      throw new Error(wechatError(data,res.status));
    }
    const scope = data.work_scope || '';
    if(recipient.work_scope && recipient.work_scope !== scope){
      invalidateWechatRecipient();
      return;
    }
    if(scope && !recipient.work_scope){
      // 初次返回若补齐账号范围，把当前选定会话与已核实的账号绑定。
      recipient.work_scope = scope;
      const key = wechatRecipientKey(recipient);
      if(!wechatText.value) wechatText.value = wechatDrafts.get(key) || '';
      wechatSendOutput.textContent = wechatSendReceipts.get(key) || '';
      updateWechatSendControl();
    }
    const scopeChanged = scope !== wechatWorkScope;
    if(scopeChanged) clearWechatWork();
    wechatWorkScope = scope;
    const mergeOlder = older && !scopeChanged;
    const unique = new Map((mergeOlder ? [...data.items,...wechatHistoryItems] : data.items).map(item => [item.id,item]));
    const items = [...unique.values()].sort((a,b) => Date.parse(a.at) - Date.parse(b.at));
    if(scopeChanged || JSON.stringify(items) !== JSON.stringify(wechatHistoryItems)) renderWechatHistory(items,mergeOlder);
    wechatHistoryItems = items;
    wechatHistoryExpanded = mergeOlder;
    wechatHistoryCursor = data.nextCursor || null;
    wechatHistoryOlder.hidden = !wechatHistoryCursor;
    wechatHistoryStatus.textContent = items.length ? '' : '暂无已记录的聊天消息。';
  }catch{
    if(sameRequest()) wechatHistoryStatus.textContent = '暂时无法读取聊天记录，请稍后刷新。';
  }finally{
    if(sameRequest()){
      wechatHistoryRequest = null;
      wechatHistoryOlder.disabled = false;
    }
  }
}

// 草稿和发送回执按会话保留。开发者交接的未绑定草稿只在人工择人后进入会话编辑框。
function switchWechatRecipient(recipient){
  const before = wechatRecipientKey(selectedWechatRecipient);
  const after = wechatRecipientKey(recipient);
  selectedWechatRecipient = recipient;
  if(before === after) return false;
  if(before){
    if(wechatText.value) wechatDrafts.set(before,wechatText.value);
    else wechatDrafts.delete(before);
  }else{
    unboundWechatDraft = wechatText.value;
  }
  if(after && wechatDrafts.has(after)) wechatText.value = wechatDrafts.get(after);
  else{
    wechatText.value = unboundWechatDraft;
    if(after) unboundWechatDraft = '';
  }
  wechatSendOutput.textContent = wechatSendReceipts.get(after) || '';
  clearWechatHistory();
  syncWechatConversationSelection();
  return true;
}

// 失去当前账号或会话权限时按原账号保存草稿，再解除发送目标。
function invalidateWechatRecipient(){
  switchWechatRecipient(null);
  wechatTarget.value = '';
  wechatRecipientStatus.textContent = '';
  wechatHistoryStatus.textContent = '会话访问权限或账号已变化，请重新选择会话。';
  updateWechatSendControl();
}

function setWechatSendReceipt(key,feedback){
  wechatSendReceipts.set(key,feedback);
  if(wechatRecipientKey(selectedWechatRecipient) === key) wechatSendOutput.textContent = feedback;
}

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
  switchWechatRecipient(null);
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
function selectWechatRecipient(recipient,{focusComposer = true} = {}){
  const target = {id:recipient.id,kind:recipient.kind,name:recipient.displayName || recipient.id};
  if(recipient.agent_wechat_account && recipient.agent_wechat_revision){
    target.agent_wechat_account = recipient.agent_wechat_account;
    target.agent_wechat_revision = recipient.agent_wechat_revision;
  }
  if(recipient.work_scope) target.work_scope = recipient.work_scope;
  const changed = switchWechatRecipient(target);
  cancelWechatRecipientSearch();
  wechatTarget.value = target.name;
  wechatHistoryTitle.textContent = `与${target.name}的聊天记录`;
  const reloadHistory = changed || wechatHistoryRefresh.hidden;
  wechatHistoryRefresh.hidden = false;
  showSelectedWechatRecipient();
  if(reloadHistory) void loadWechatConversation();
  if(focusComposer) wechatText.focus();
  else wechatHistoryTitle.focus();
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
    wechatHistoryTitle.textContent = `与${selectedWechatRecipient.name}的聊天记录`;
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
async function postWechatControl(path, payload){
  const headers = {'Content-Type':'application/json'};
  const body = JSON.stringify(payload);
  const res = await fetch(path,{method:'POST',headers,body});
  const data = await res.json();
  if(res.status !== 403 || data.error !== 'bridge_control_token_required') return {res,data};
  const token = prompt('请输入微信桥接控制密钥');
  if(!token?.trim()) return {res,data};
  const authorized = await fetch(path,{method:'POST',headers:{...headers,'X-Mahjong-Bridge-Token':token.trim()},body});
  return {res:authorized,data:await authorized.json()};
}
async function setWechatSwitch(path, enabled, question){
  if(wechatSwitchBusy || !confirm(question)) return;
  wechatSwitchBusy = true;
  ++wechatChannelRevision;
  try{
    const {res,data} = await postWechatControl(path, {enabled});
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
  const key = wechatRecipientKey(recipient);
  const signature = JSON.stringify([key, text]);
  if(!pendingWechatSends.has(signature)) pendingWechatSends.set(signature, createMessageId());
  const payload = {
    recipient_id:recipient.id, kind:recipient.kind, text, request_id:pendingWechatSends.get(signature)
  };
  if(recipient.agent_wechat_account && recipient.agent_wechat_revision){
    payload.agent_wechat_account = recipient.agent_wechat_account;
    payload.agent_wechat_revision = recipient.agent_wechat_revision;
  }
  if(recipient.work_scope) payload.work_scope = recipient.work_scope;
  wechatSendBusy = true;
  updateWechatSendControl();
  setWechatSendReceipt(key,'正在等待发送回执…');
  try{
    const {res,data} = await postWechatControl('/api/wechat/send', payload);
    if(res.ok && data.ok === true && data.sent === true){
      pendingWechatSends.delete(signature);
      if(wechatDrafts.get(key) === text) wechatDrafts.delete(key);
      const context = data.context_recorded ? '已记录到会话上下文。' : '';
      setWechatSendReceipt(key,`已发送给「${recipient.name}」。${context}`);
      if(wechatRecipientKey(selectedWechatRecipient) === key){
        if(wechatText.value === text) wechatText.value = '';
        void loadWechatConversation();
      }
      void loadWechatConversations({quiet:true});
    }else{
      const heading = data.delivery_uncertain ? '发送结果待核对。' : '未发送。';
      setWechatSendReceipt(key,heading + wechatError(data,res.status));
    }
  }catch{
    setWechatSendReceipt(key,'暂未收到发送回执。请先在微信核对。');
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
// 哈希保留既有手动发消息、牌局和审批链接，切页时不会清空草稿或当前会话。
function showOwnerView(){
  const hash = location.hash.slice(1);
  const active = ['operations','current-games','invite-approvals'].includes(hash) ? 'operations'
    : hash === 'settings' ? 'settings' : 'messages';
  for(const link of ownerNav.querySelectorAll('a')){
    const id = link.hash.slice(1);
    document.getElementById(id).hidden = id !== active;
    if(id === active) link.setAttribute('aria-current','page');
    else link.removeAttribute('aria-current');
  }
  if(active === 'operations') void loadState();
  if(active === 'settings') void loadWechatChannelStatus();
  void refreshWechatWork();
}

function initOwnerWorkspace(){
  // Chromium 的中文选字完成事件通过事件监听触发，保留精确候选确认。
  wechatTarget.addEventListener('input', queueWechatRecipientSearch);
  wechatTarget.addEventListener('compositionend', () => queueWechatRecipientSearch());
  restoreTrialDraft();
  void loadWechatChannelStatus();
  showOwnerView();
  window.addEventListener('hashchange',showOwnerView);
  document.addEventListener('visibilitychange',refreshWechatWork);
  void loadWechatConversations();
  setInterval(() => {
    if(document.hidden) return;
    if(!document.getElementById('operations').hidden) void loadState();
    if(!document.getElementById('messages').hidden){
      void loadWechatConversations({quiet:true});
      void loadWechatConversation({quiet:true});
    }
  }, 6000);
}
initOwnerWorkspace();
