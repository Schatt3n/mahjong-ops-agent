// 开发者工具沿用正式运行接口。原始状态和指定 Trace 只在主动查看时请求。
let liveTimer = null;
let pendingMessageAttempt = null;
let messagePollGeneration = 0;
let stateRefreshGeneration = 0;
let stateDebugGeneration = 0;
let latestTrialReply = "";
let logRevision = 0;
let logRequest = null;
let recentLog = [];
let traceRevision = 0;
function createMessageId(){
  if(globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function'){
    return `web_${globalThis.crypto.randomUUID()}`;
  }
  return `web_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}
function sleep(ms){
  return new Promise(resolve => setTimeout(resolve, ms));
}
function renderMessageResult(body){
  window.lastTraceId = body.trace_id || window.lastTraceId;
  output.textContent = JSON.stringify(body, null, 2);
  latestTrialReply = typeof body.final_reply === 'string' ? body.final_reply.trim() : '';
  document.getElementById('handoffTrialReply').disabled = !latestTrialReply;
}
async function pollInputBatch(batchId, generation){
  const terminalStatuses = new Set(['completed', 'ignored', 'failed', 'superseded']);
  const deadline = Date.now() + 180000;
  while(Date.now() < deadline && generation === messagePollGeneration){
    await sleep(1000);
    if(generation !== messagePollGeneration) return;
    const res = await fetch('/api/input-batches/result', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({batch_id: batchId})
    });
    if(generation !== messagePollGeneration) return;
    if(res.status === 404){
      renderMessageResult({final_reply: '', status: 'superseded', input_status: 'superseded', poll_batch_id: batchId});
      return;
    }
    if(!res.ok) continue;
    const body = await res.json();
    if(generation !== messagePollGeneration) return;
    renderMessageResult(body);
    if(terminalStatuses.has(body.input_status)){
      await loadState();
      await refreshLive();
      return;
    }
  }
}
// 每次发送固定本次幂等尝试；旧代次的 HTTP 响应或异常不得覆盖新输入的显示和重试状态。
async function sendMessage(){
  const generation = ++messagePollGeneration;
  const requestSignature = JSON.stringify([
    conversationId.value,
    senderId.value,
    senderName.value,
    text.value
  ]);
  if(!pendingMessageAttempt || pendingMessageAttempt.signature !== requestSignature){
    pendingMessageAttempt = {signature: requestSignature, messageId: createMessageId()};
  }
  const attempt = pendingMessageAttempt;
  const payload = {
    conversation_id: conversationId.value,
    sender_id: senderId.value,
    sender_name: senderName.value,
    text: text.value,
    message_id: attempt.messageId,
    aggregate_fragments: true
  };
  let res, body;
  try{
    res = await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if(generation !== messagePollGeneration) return;
    body = await res.json();
  }catch(error){
    if(generation !== messagePollGeneration) return;
    throw error;
  }
  if(generation !== messagePollGeneration) return;
  if(res.ok && pendingMessageAttempt === attempt) pendingMessageAttempt = null;
  renderMessageResult(body);
  await loadState();
  await refreshLive();
  if(res.ok && body.poll_batch_id && (body.waiting_for_more_input || body.background_processing)){
    void pollInputBatch(body.poll_batch_id, generation);
  }
}
async function recordBadcase(){
  const payload = {
    source_trace_id: window.lastTraceId || '',
    reason: badcaseReason.value,
    expected: { note: badcaseExpected.value }
  };
  const res = await fetch('/api/badcases',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  badcaseOutput.textContent = JSON.stringify(await res.json(), null, 2);
  await loadState();
}

// 调试页仅需要待处理计数，完整快照须由操作者展开。
async function loadState(){
  const generation = ++stateRefreshGeneration;
  try{
    const res = await fetch('/api/state?view=console',{cache:'no-store'});
    if(!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if(generation !== stateRefreshGeneration) return;
    for(const counter of document.querySelectorAll('[data-state-count]')){
      counter.textContent = data.summary[counter.dataset.stateCount];
    }
  }catch{
    if(generation !== stateRefreshGeneration) return;
    for(const counter of document.querySelectorAll('[data-state-count]')) counter.textContent = '—';
  }
}

// 调试快照仅在展开或手动刷新时读取，收起后清空，忽略仍在途的旧请求。
async function loadStateDebug(){
  const generation = ++stateDebugGeneration;
  const details = document.getElementById('stateDebug');
  const raw = document.getElementById('stateRaw');
  raw.textContent = details.open ? '正在加载调试快照…' : '';
  if(!details.open) return;
  try{
    const res = await fetch('/api/state', {cache:'no-store'});
    if(!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if(details.open && generation === stateDebugGeneration) raw.textContent = JSON.stringify(data, null, 2);
  }catch(error){
    if(details.open && generation === stateDebugGeneration) raw.textContent = `调试快照加载失败（${error.message}）`;
  }
}
function compactTraceLine(line){
  const text = String(line || '');
  const match = text.match(/^(.*?)-(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})-(\w+): (.*)$/);
  if(!match) return text.slice(0, 900);
  const payload = match[4];
  try{
    const json = JSON.parse(payload);
    const step = json.step || json.direction || json.path || '-';
    const reply = json.reply || json.final_reply || json.suggested_safe_text || '';
    const reason = json.reason || json.reasoning_summary || json.error || '';
    const trace = match[1];
    return `${match[2]} ${match[3]} ${trace} ${step}${reply ? ' | reply=' + reply : ''}${reason ? ' | ' + reason : ''}`;
  }catch{
    return text.slice(0, 900);
  }
}

// 匹配已加载的日志内容和摘要。换通道时旧查询不能覆盖新通道。
function renderRecentLog(){
  const query = logFilter.value.trim().toLocaleLowerCase();
  const visible = recentLog.filter(row => !query || row.search.includes(query));
  runtimeLogs.textContent = visible.length ? visible.map(row => row.text).join('\n')
    : query ? '最近的记录中没有匹配项。' : '暂无记录。';
}

async function refreshLive(){
  const revision = ++logRevision;
  logRequest?.abort();
  const request = new AbortController();
  logRequest = request;
  const source = logSource.value;
  const paths = {
    runtime:'/api/logs?limit=60',
    'agent-wechat':'/api/channels/agent-wechat/raw?limit=20',
    wechaty:'/api/channels/wechaty/raw?limit=20'
  };
  const path = paths[source];
  try{
    const res = await fetch(path,{cache:'no-store',signal:request.signal});
    if(!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if(revision !== logRevision) return;
    recentLog = source === 'runtime' ? (data.tail || []).map(line => ({
      text:compactTraceLine(line),search:String(line).toLocaleLowerCase()
    })) : (data.records || []).map(record => {
      const text = JSON.stringify(record,null,2);
      return {text,search:text.toLocaleLowerCase()};
    });
    renderRecentLog();
    liveStatus.textContent = '更新于 ' + new Date().toLocaleTimeString('zh-CN');
  }catch(error){
    if(revision !== logRevision) return;
    liveStatus.textContent = `日志暂不可用（${error.message}）。`;
  }finally{
    if(logRequest === request) logRequest = null;
  }
}

function toggleAutoRefresh(){
  clearInterval(liveTimer);
  liveTimer = autoRefresh.checked ? setInterval(() => {
    if(document.hidden) return;
    if(!logRequest) void refreshLive();
    void loadState();
  },5000) : null;
}

async function loadRuntimeHealth(){
  try{
    const [runtime, bridge] = await Promise.all([
      fetch('/api/health',{cache:'no-store'}),fetch('/api/wechat/health',{cache:'no-store'})
    ]);
    const channel = await bridge.json();
    const runtimeLabel = runtime.ok ? 'Runtime 已连接' : 'Runtime 不可用';
    const channelLabel = bridge.ok && channel.ok ? `微信适配器已连接（${channel.adapter || 'wechat'}）` : '微信适配器不可用';
    runtimeHealth.textContent = [runtimeLabel,channelLabel].join(' · ');
  }catch{
    runtimeHealth.textContent = '暂时无法读取运行状态。';
  }
}

async function lookupTrace(event){
  event.preventDefault();
  const id = traceQuery.value.trim();
  if(!id) return;
  const revision = ++traceRevision;
  traceOutput.textContent = '正在查找 Trace…';
  try{
    const params = new URLSearchParams({trace_id:id});
    const res = await fetch('/api/traces?'+params,{cache:'no-store'});
    if(!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if(revision !== traceRevision) return;
    traceOutput.textContent = data.events?.length ? JSON.stringify(data,null,2) : '未找到该 Trace。';
  }catch(error){
    if(revision === traceRevision) traceOutput.textContent = `Trace 查询失败（${error.message}）。`;
  }
}

// 只有操作者点击交接时才预填老板工作台，收件人与发送继续由人工确认。
function sendTrialDraftToOwner(){
  if(!latestTrialReply) return;
  try{
    sessionStorage.setItem('mahjong.owner.wechat-draft',latestTrialReply);
    location.assign('/#manual-send');
  }catch{
    trialDraftStatus.textContent = '浏览器无法保存草稿，请复制回复正文。';
  }
}

async function resetState(){
  const ok = confirm('确认清空当前测试状态和记忆？会删除局、草稿、对话上下文、checkpoint、幂等缓存和消息结果；默认保留客户画像、badcase/eval 和日志。');
  if(!ok) return;
  const res = await fetch('/api/reset-state',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({reason:'operator clicked reset state and memory'})
  });
  resetOutput.textContent = JSON.stringify(await res.json(), null, 2);
  if(res.ok){
    window.lastTraceId = '';
    latestTrialReply = '';
    document.getElementById('handoffTrialReply').disabled = true;
  }
  await loadState();
  await refreshLive();
}

function initDeveloperWorkspace(){
  logSource.addEventListener('change',() => {
    recentLog = [];
    runtimeLogs.textContent = '正在读取日志…';
    void refreshLive();
  });
  logFilter.addEventListener('input',renderRecentLog);
  traceLookup.addEventListener('submit',lookupTrace);
  toggleAutoRefresh();
  void loadRuntimeHealth();
  void loadState();
  void refreshLive();
}
initDeveloperWorkspace();
