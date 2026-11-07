"use strict";
const $ = s => document.querySelector(s);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const readLocal = key => { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } };
const saveLocal = (key, value) => { try { value === null ? localStorage.removeItem(key) : localStorage.setItem(key, JSON.stringify(value)); } catch {} };
let session = null, serviceReady = false, busy = false, file = null, fileReady = false, selectionRevision = 0, previewURL = null, requestKey = null, historyOffset = 0;
const mode = () => $('input[name="mode"]:checked').value;
async function api(path, options = {}) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 35000);
  try {
    const response = await fetch(path, {...options, headers:{"X-PetRead-Client":"web", ...options.headers}, signal:controller.signal, cache:"no-store"});
    if (!response.ok) {
      let detail = `请求未完成（${response.status}）`;
      try { const j = await response.json(); if (typeof j.detail === "string") detail = j.detail; } catch {}
      throw new Error(detail);
    }
    return response;
  } finally { clearTimeout(timer); }
}
const jsonPost = (path, value) => api(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(value)});
function error(message) { $('#form-error').hidden = !message; $('#form-error').textContent = message || ''; }
function update() {
  const ready = mode() === 'text' ? $('#description').value.trim().length > 0 : !!file && fileReady;
  $('#submit').disabled = !session || !serviceReady || !ready || !$('#consent').checked || busy;
  $('#submit').textContent = busy ? '正在整理，请勿重复提交' : !session ? '连接设备空间…' : !serviceReady ? '分析服务暂未就绪' : '开始这次观察 →';
}
function view(name) {
  document.querySelectorAll('.view').forEach(el => {el.hidden = el.id !== name;});
  document.querySelectorAll('[data-view]').forEach(el => {el.dataset.view === name ? el.setAttribute('aria-current','page') : el.removeAttribute('aria-current');});
  if (name === 'history') loadHistory(true);
}
document.querySelectorAll('[data-view]').forEach(el => el.onclick = () => view(el.dataset.view));
function clearFile() {
  selectionRevision++; fileReady = false;
  if (previewURL) URL.revokeObjectURL(previewURL);
  file = null; previewURL = null; $('#media').value = ''; $('#preview').replaceChildren(); $('#preview').hidden = true;
  $('#remove-file').hidden = true; requestKey = null; update();
}
const modeInfo = {
  quick:['选一张清楚的照片','完整拍到身体和耳朵 · JPEG / PNG / WebP · 最大 10 MB','image/jpeg,image/png,image/webp',10],
  native:['选一段自然的视频','15 秒内 · 最大 10 MB · 1080p / 30fps 以内','video/mp4,video/quicktime,video/webm',10],
  audio:['选一段自然的声音','不超过 30 秒 · WAV / MP3 / M4A · 最大 15 MB','audio/wav,audio/mpeg,audio/mp4,audio/x-m4a',15],
};
document.querySelectorAll('input[name="mode"]').forEach(el => el.onchange = () => {
  clearFile(); error(''); const text = mode() === 'text'; $('#media-group').hidden = text; $('#text-group').hidden = !text;
  if (!text) {const [title,hint,accept] = modeInfo[mode()]; $('#file-title').textContent = title; $('#file-hint').textContent = hint; $('#media').accept = accept;}
  update();
});
$('#media').onchange = () => {
  const selected = $('#media').files[0]; if (!selected) return;
  if (selected.size === 0 || selected.size > modeInfo[mode()][3]*1024*1024) {clearFile(); error('素材为空或超过大小上限，请缩小后再选。');return;}
  clearFile();
  file = selected; previewURL = URL.createObjectURL(file); requestKey = null; error('');
  const revision = selectionRevision;
  const selectedMode = mode(), element = document.createElement(selectedMode === 'quick' ? 'img' : selectedMode === 'audio' ? 'audio' : 'video');
  element.src = previewURL;
  element.onerror = () => { if (revision !== selectionRevision) return; clearFile(); error('浏览器无法读取此素材，请换成受支持的格式后重试。'); };
  if (selectedMode === 'quick') { element.alt = '本次分析的照片预览'; element.onload = () => {if(revision === selectionRevision){fileReady=true;update();}}; }
  else { element.controls = true; element.preload = 'metadata'; element.onloadedmetadata = () => {
    if (revision !== selectionRevision) return;
    const seconds = selectedMode === 'native' ? 15 : 30;
    if (!Number.isFinite(element.duration) || element.duration <= 0 || element.duration > seconds) {clearFile();error(`素材需在 ${seconds} 秒以内，请先裁剪再上传。`);return;}
    if (selectedMode === 'native' && (Math.max(element.videoWidth,element.videoHeight)>1920 || Math.min(element.videoWidth,element.videoHeight)>1080 || !element.videoWidth || !element.videoHeight)) {clearFile();error('视频需不超过 1080p，支持横屏或竖屏；请先缩小分辨率。');return;}
    fileReady=true;update();
  }; }
  $('#preview').replaceChildren(element); $('#preview').hidden = false; $('#remove-file').hidden = false; update();
};
$('#remove-file').onclick = clearFile;
$('#analysis-form').addEventListener('input', () => {if (!busy) requestKey = null; update();});
$('#analysis-form').onsubmit = async event => {
  event.preventDefault(); if (busy || !session || !serviceReady || !$('#consent').checked || (mode() !== 'text' && (!file || !fileReady))) return;
  busy = true; update(); error(''); $('#result').hidden = true;
  try {
    await jsonPost('/consent', {accepted:true, version:session.required_consent_version});
    const data = new FormData(); data.append('species',$('input[name="species"]:checked').value); data.append('mode',mode());
    data.append('context',$('#context').value.trim());
    if (mode() === 'text') data.append('text',$('#description').value.trim()); else data.append('file',file);
    requestKey ||= crypto.randomUUID();
    const response = await (await api('/analyze', {method:'POST', body:data, headers:{'Idempotency-Key':requestKey}})).json();
    const pending = {id:response.job_id, started:Date.now()}; saveLocal('petread_active_v2',pending);
    poll(pending);
  } catch(e) {busy = false; update();error(`未确认提交完成：${e.message}。未自动重试。若网络中断，可保持素材不变再次提交，沿用同一请求标识。`);}
};
async function poll(pending) {
  busy = true; update(); $('#progress').hidden = false;
  const deadline = Date.now() + 300000;
  while (Date.now() < deadline) {
    $('#elapsed').textContent = `${Math.floor((Date.now()-pending.started)/1000)}s`;
    try {
      const state = await (await api(`/jobs/${encodeURIComponent(pending.id)}`)).json();
      const last = state.events.at(-1);
      if (last?.[1]?.message) $('#progress-text').textContent = last[1].message;
      if (state.done) {
        saveLocal('petread_active_v2',null); busy = false; update(); $('#progress').hidden = true;
        requestKey = null;
        if (last?.[0] === 'done') render(last[1]); else error(last?.[1]?.message || '未获得有效结果。没有自动重试。');
        return;
      }
    } catch(e) {$('#progress-text').textContent = `连接暂时中断，正在取回原任务（不会再次调用模型）：${e.message}`;}
    await new Promise(resolve => setTimeout(resolve, document.hidden ? 5000 : 1800));
  }
  $('#progress-text').textContent = '仍未确认任务结果；请稍后刷新页面恢复查询。不要重复提交。';
}
function describe(value) { return typeof value === 'string' ? value : JSON.stringify(value); }
function list(items) { return `<ul>${items.map(x=>`<li>${esc(describe(x))}</li>`).join('')}</ul>`; }
function observationLines(r) {
  const names={ear_position:'耳位',body_posture:'体态',tail:'尾部',pupil:'瞳孔',whisker_position:'胡须',mouth:'口部',eye:'眼部',affiliative_signals:'亲和信号',displacement_signals:'位移信号'};
  const lines=[];
  if (Array.isArray(r.literal_observations)) {
    return r.literal_observations.map(o => `${o.source === 'audio' ? '声音' : '画面'} ${o.t}s · ${names[o.channel] || o.channel}：${o.status === 'observed' ? o.description : '不可观测：'+o.reason}`);
  }
  for(const frame of r.visual_signals || []) {
    if (!frame || typeof frame !== 'object') {lines.push(describe(frame));continue;}
    if (frame.signals && typeof frame.signals === 'object') {
      for(const [key,value] of Object.entries(frame.signals)) {
        const detail=value && typeof value === 'object' ? Object.entries(value).map(([k,v])=>`${k}：${describe(v)}`).join('；') : describe(value);
        lines.push(`旧版编码（可能含解释）：${names[key] || key}：${detail}`);
      }
    } else lines.push(describe(frame));
  }
  for(const event of r.audio_events || []) lines.push(describe(event));
  return lines;
}
function render(payload) {
  view('observe'); const r=payload.result || {}, m=payload.meta || {}, c=r.candidates || [];
  const limitations=[...(r.input_assessment?.limitations || []), ...(r.audio_assessment?.limitations || []), ...(r.limitations || [])];
  if (r.insufficient_reason) limitations.unshift(r.insufficient_reason);
  const observations=observationLines(r);
  const trace=r._meta || {}, calls=trace.provider_calls || [];
  const timing=m.timings || trace.timings || {};
  const apiSeconds=calls.length && calls.every(x=>typeof x.latency_s === 'number') ? calls.reduce((sum,x)=>sum+x.latency_s,0).toFixed(2) : null;
  $('#result').innerHTML = `<article class="card result-card"><p class="eyebrow">02 / REFLECT · AI 生成，需核对</p><h2>${r.insufficient_evidence ? '这一刻，信息还不够。' : '这些信号，可能意味着…'}</h2><p class="result-lead">${esc(r.translation || r.insufficient_reason || '暂无可用解释')}</p>
    ${c.length ? `<div class="inset"><h3>待核对的候选解释</h3>${list(c.map(x=>x.label))}</div>`:''}
    <div class="evidence-grid"><section class="inset"><h3>模型报告的可观察信号</h3>${observations.length ? list(observations) : '<p>没有返回可展示的观察。文字描述不是独立观测证据。</p>'}</section><section class="inset limit"><h3>哪些还不能确定</h3>${limitations.length ? list(limitations) : '<p>有限素材无法直接证明情绪或意图；模型的解释可能有误。</p>'}</section></div>
    <section class="inset"><h3>下一步可以怎么做</h3><p>${esc(r.suggestion || '先观察自然行为；必要时补充更清楚的素材。')}</p></section>
    <details><summary>查看完整分析与未校准分数 ↓</summary><p class="micro">分数是模型自评，不是正确概率；没有经过真实效果校准。</p><pre>${esc(JSON.stringify(r,null,2))}</pre></details>
    <p class="result-meta">${esc(m.model || '模型信息未提供')} · 分析链路 ${m.latency_s == null ? 'UNKNOWN' : esc(m.latency_s)+' 秒'} · 输入 ${esc(m.tokens_in ?? 'UNKNOWN')} / 输出 ${esc(m.tokens_out ?? 'UNKNOWN')} token</p>
    <details><summary>核对本次调用与耗时 ↓</summary><p class="micro">${calls.length ? `${calls.length} 次已记录调用；API 等待 ${esc(apiSeconds ?? 'UNKNOWN')} 秒。` : '没有可核实的模型调用明细；不将缺失数据补为零。'}${timing.server_total_s == null ? '' : `服务器总耗时 ${esc(timing.server_total_s)} 秒（不含客户端上传和轮询等待）。`}</p>${calls.length ? list(calls.map(x=>`${x.model || 'UNKNOWN'} · ${x.status || 'UNKNOWN'} · 响应 ID ${x.response_id || 'UNKNOWN'} · Prompt ${x.prompt_sha256 || 'UNKNOWN'}`)) : ''}<p class="micro">调用记录证明请求来源与过程，不证明判读正确。置信度尚未校准。</p></details>
    <div class="feedback-row"><label for="feedback">后来实际发生了什么？</label><textarea id="feedback" maxlength="4000" rows="2" placeholder="记录后续可见的行为，而不是把自己的猜测当作真值。"></textarea><button class="secondary" id="send-feedback">保存观察反馈</button><span id="feedback-status" role="status"></span></div></article>`;
  $('#result').hidden = false; $('#result').focus({preventScroll:true}); $('#result').scrollIntoView({behavior:'smooth',block:'start'});
  $('#send-feedback').onclick = async () => {
    if (!$('#feedback').value.trim()) return;
    $('#send-feedback').disabled = true;
    try {await jsonPost(`/feedback/${payload.record_id}`,{user_note:$('#feedback').value.trim()}); $('#feedback-status').textContent = '已保存';}
    catch(e) {$('#feedback-status').textContent=e.message;$('#send-feedback').disabled=false;}
  };
}
async function loadHistory(reset=false) {
  if (!session) return;
  if (reset) {historyOffset = 0; $('#history-list').replaceChildren();}
  try {
    const {items} = await (await api(`/history?limit=20&offset=${historyOffset}`)).json();
    if (!items.length && !historyOffset) $('#history-list').innerHTML='<div class="empty">还没有观察记录。<br>从一张照片或一段声音开始，慢慢积累你的日记。</div>';
    for (const item of items) {
      const button=document.createElement('button');button.className='history-item';
      button.innerHTML=`<span>${esc(item.created_at)} · ${item.species === 'cat' ? '猫咪' : '狗狗'}</span><strong>${esc(item.insufficient_evidence ? '信息不足，等待补充' : item.predicted_label || '观察记录')}</strong><span>${esc(item.translation || item.context || '查看详情 →')}</span>`;
      button.onclick=async()=>{try{const r=await(await api(`/record/${item.id}`)).json();render({record_id:r.id,result:r.result,meta:r});}catch(e){button.textContent=e.message;}};
      $('#history-list').append(button);
    }
    historyOffset += items.length; $('#more-history').hidden = items.length < 20;
  } catch(e) {$('#history-list').textContent=e.message;}
}
$('#refresh-history').onclick=()=>loadHistory(true);$('#more-history').onclick=()=>loadHistory();
$('#revoke').onclick=async()=>{try{await jsonPost('/consent',{accepted:false,version:session.required_consent_version});$('#consent').checked=false;update();$('#privacy-status').textContent='已撤回。不会开始新的分析；已发出的请求无法撤回。';}catch(e){$('#privacy-status').textContent=e.message;}};
$('#export').onclick=async()=>{try{const blob=await(await api('/me/export')).blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='petread-data.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);}catch(e){$('#privacy-status').textContent=e.message;}};
$('#delete').onclick=()=>$('#delete-dialog').showModal();$('#cancel-delete').onclick=()=>$('#delete-dialog').close();
$('#confirm-delete').onclick=async()=>{try{await api('/me',{method:'DELETE'});saveLocal('petread_active_v2',null);$('#delete-dialog').close();$('#privacy-status').textContent='当前设备数据已删除，无法恢复。';session=null;$('#consent').checked=false;await bootstrap();}catch(e){$('#delete-dialog').close();$('#privacy-status').textContent=e.message;}};
async function bootstrap() {
  try {session=await(await api('/session',{method:'POST'})).json();
    const health=await(await api('/healthz')).json();serviceReady=health.analysis_ready === true;
    if(health.mock) $('.pill').textContent='离线预览 · 不调用模型';
    else if(!serviceReady) error('分析服务未就绪，暂不接收素材。请稍后刷新重试。');
    update();const pending=readLocal('petread_active_v2');if(pending?.id)poll(pending);}
  catch(e) {error(`设备空间暂时不可用：${e.message}。请刷新重试。`);}
}
bootstrap();
