/* Immutable editing revisions; manual changes lock an entire narration's shot combination. */
(() => {
  const $ = s => document.querySelector(s);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const enc = encodeURIComponent;
  let project, state = {}, tab = 'selection', historyRevision = null, busy = false, generation = 0, timer, activeCrop, useModel = false;
  const motionLabels = {auto:'自动安排',hold:'静止',zoom_in:'推进',zoom_out:'拉远',pan_left:'向左平移',pan_right:'向右平移',pan_up:'向上平移',pan_down:'向下平移'};
  const api = () => `/api/editing/${enc(project)}`;
  const image = pid => `${api()}/evidence/${enc(pid)}?revision=${enc(state.revision)}`;
  const shotImage = (nid, pid) => `${api()}/preview/${enc(nid)}/${enc(pid)}?revision=${enc(state.revision)}`;
  const disabled = () => busy || Boolean(historyRevision);
  const percent = value => `${Math.round((value || 0) * 100)}%`;
  const matchLabels = {direct:'直接展示',transition:'过渡延续',commentary:'吐槽反应',callback:'画面回扣'};
  const textRanges = item => item.text_ranges ? `<details><summary>对应原文（${item.text_ranges.length} 段）</summary>${item.context_only ? `<p>借用相邻镜头：${(item.context_origins || []).map(o=>`${o.direction==='previous'?'上一段':'下一段'} ${esc(o.event_id)}`).join('、')}</p>` : ''}${item.text_ranges.map(r=>`<p><small>[${r.start_char}, ${r.end_char}) · ${matchLabels[r.match_type || 'direct'] || esc(r.match_type)}</small><br>${esc(r.text)}${r.reason?`<br><small>理由：${esc(r.reason)}</small>`:''}</p>`).join('') || '<p>尚未关联原文，请调整文字范围。</p>'}</details>` : '';
  const semanticNotice = () => {
    if (state.sync?.shot_text_alignment?.status === 'aligned') return '<div class="status">图片已按对应原文和实测配音边界排轴，请观看确认内容和节奏。</div>';
    const result = state.sync?.semantic_alignment;
    if (result?.status === 'manual_review') return '<div class="status">已按实际配音时长排轴；请人工确认镜头内容和节奏。</div>';
    return `<div class="status ${result?.status === 'aligned' ? '' : 'error'}">句子与画面对应：${result?.status === 'aligned' ? '来源与实测边界检查通过，仍需观看确认' : '待检查；总时长一致不代表语义匹配'}</div>${(result?.issues || []).map(i => `<p>${esc(i.narration_id)} ${esc(i.detail)}</p>`).join('')}`;
  };
  async function json(url, options) {
    const response = await fetch(url, options), value = await response.json();
    if (!response.ok) throw new Error(value.error || `请求失败 (${response.status})`);
    return value;
  }
  function message(text, error = false) {
    const target = $('#message');
    if (target) {target.textContent = text; target.classList.toggle('error', error);}
  }
  const groupById = nid => state.groups.find(g => g.narration_id === nid);
  const shotButton = (nid, pid, label = '调整裁切 / 运动') => `<button class="secondary" data-crop="${esc(pid)}" data-nid="${esc(nid)}" ${disabled() ? 'disabled' : ''}>${label}</button>`;
  function draw() {
    const groups = state.groups || [], metrics = state.metrics || {}, ready = state.readiness;
    const approved = groups.length > 0 && groups.every(g => g.review_status === 'approved') && VoiceReview.ready(state) && ImageCleanup.ready(state);
    $('#editor').innerHTML = `${historyRevision ? '<p class="readonly">历史版本只读。选择“最新版本”继续编辑。</p>' : ''}
      <div class="toolbar"><button data-action="select" ${disabled() || ready && (!ready.segments || ready.approved !== ready.segments) ? 'disabled' : ''}>${groups.length ? '重新选择未锁定镜头' : '生成镜头候选'}</button>
      <label><input type="checkbox" id="use-model" ${useModel ? 'checked' : ''} ${disabled() ? 'disabled' : ''}> AI 看图选镜与裁切复核</label>
      <a class="button secondary" href="/story-review?project=${enc(project)}">文案审核</a>
      <button class="secondary" data-action="refresh" ${busy ? 'disabled' : ''}>刷新</button>
      <select id="history" aria-label="剪辑版本" ${busy ? 'disabled' : ''}><option value="">最新版本</option>${(state.history || []).map(h => `<option value="${h.revision}" ${historyRevision === h.revision ? 'selected' : ''}>${esc(h.action)} · ${esc(h.updated_at)}</option>`).join('')}</select></div>
      <div id="message" class="status" role="status">${busy ? '正在处理…' : ready ? `已审核文案 ${ready.approved}/${ready.segments} 段。全部通过审核后可选择镜头。` : '人工修改会锁定本段镜头组合。AI 重跑保留所有已锁定选择、排除、裁切和运动。'}</div>
      ${groups.length ? `<div class="metrics"><div class="metric"><strong>${percent(metrics.coverage_rate)}</strong><span>必要剧情覆盖率</span></div><div class="metric"><strong>${percent(metrics.duplicate_rate)}</strong><span>相似且无新增信息的镜头重复率</span></div><div class="metric"><strong>${metrics.selected_count}</strong><span>镜头 / ${(state.library || []).length} 个来源</span></div><div class="metric"><strong>${groups.filter(g => g.review_status === 'approved').length}/${groups.length}</strong><span>已审核镜头组</span></div></div>
      <nav class="tabs" aria-label="剪辑工作区">${[['selection','镜头选择审核'],['cleanup','图片清理'],['library','完整分镜库'],['voice','配音导演与试听'],['timeline','时间轴'],['preview','视频预览']].map(([id, label]) => `<button data-tab="${id}" class="${tab === id ? 'active' : ''}">${label}</button>`).join('')}</nav>
      <div id="content"></div>` : '<div class="empty">第一步：审核剧情解说稿，再生成有来源依据的镜头候选。</div>'}`;
    $('#use-model').onchange = event => { useModel = event.target.checked; };
    if (!groups.length) return;
    const content = $('#content');
    if ((tab === 'timeline' || tab === 'preview') && state.timeline) {
      content.insertAdjacentHTML('beforebegin', semanticNotice());
    }
    if (tab === 'voice') {VoiceReview.mount(content,{state,api,disabled,action,message,project});return;}
    if (tab === 'cleanup') {ImageCleanup.mount(content,{state,api,disabled,action,message});return;}
    if (tab === 'library') {
      content.innerHTML = '<label for="library-search">搜索分镜、事件、人物、动作或对白</label><input id="library-search" placeholder="搜索完整来源库"><p class="muted">来源库保留所有画面。边界待核对的来源区域可以在采用后调整裁切。</p><div class="library" id="library"></div>';
      drawLibrary(''); $('#library-search').oninput = e => drawLibrary(e.target.value); return;
    }
    if (tab === 'selection') {
      content.innerHTML = groups.map(g => `<article class="group"><div class="group-header"><h2>${esc(g.narration_id)} <span class="tag">${g.review_status === 'approved' ? '审核通过 · 已锁定' : g.locked ? '人工修改 · 已锁定' : '待审核'}</span></h2><div class="row">
      <button data-review="approve" data-nid="${esc(g.narration_id)}" ${disabled() || g.review_status === 'approved' ? 'disabled' : ''}>审核本段镜头</button><button class="secondary" data-review="unlock" data-nid="${esc(g.narration_id)}" ${disabled() || !g.locked ? 'disabled' : ''}>解锁供 AI 重选</button></div></div>
      <p class="narration">${esc(g.text)}</p><p class="muted">必要事件与因果前提：${esc(g.required_event_ids.join('、'))}</p>${g.text_mapping?.status === 'needs_review' ? `<p class="status error">图片与文字范围待审核：${esc(g.text_mapping.issues.join('；'))}</p>` : ''}<div class="selected">${g.selected.map((s, i) => `<div class="shot"><img loading="lazy" src="${shotImage(g.narration_id, s.panel_id)}" alt="已选分镜 ${esc(s.panel_id)}"><p>${i + 1}. 来源 ${esc(s.panel_id)} ${s.locked ? '🔒' : ''}</p>${textRanges(s)}${shotButton(g.narration_id, s.panel_id)}<div class="row"><button class="secondary" data-move="${i}" data-nid="${esc(g.narration_id)}" ${disabled() || i === 0 ? 'disabled' : ''}>前移</button><button class="danger" data-remove="${esc(s.panel_id)}" data-nid="${esc(g.narration_id)}" ${disabled() ? 'disabled' : ''}>移除</button></div></div>`).join('')}</div>
      <details><summary>查看本段候选及全部采用 / 排除理由（${g.decisions.length}）</summary>${[...g.decisions].sort((a,b) => Number(b.eligible) - Number(a.eligible) || b.score-a.score).map(r => `<div class="candidate"><img loading="lazy" src="${image(r.panel_id)}" alt="候选 ${esc(r.panel_id)}"><div><b>${esc(r.panel_id)} · ${r.selected ? '已选' : '排除'}</b><p>${esc(r.reason)}</p><div class="scores">${Object.entries(r.scores).map(([k,v]) => `${esc(({importance:'剧情',narration_match:'旁白',visual:'表现力',continuity:'人物连续性',quality:'质量'})[k] || k)} ${percent(v)}`).join(' · ')}</div></div><button class="secondary" data-add="${esc(r.panel_id)}" data-nid="${esc(g.narration_id)}" ${disabled() || !r.eligible || r.selected ? 'disabled' : ''}>采用</button></div>`).join('')}</details></article>`).join('');
      return;
    }
    if (tab === 'timeline') {
      if (state.voice_review_required && !state.voice_direction) {
        content.innerHTML = '<div class="empty">请先在“配音导演与试听”完成配音试听与审核，再安排时间轴。</div>';return;
      }
      if (state.voice_direction) {
        content.innerHTML = `<div class="toolbar"><button data-action="direct" ${disabled() || !VoiceReview.ready(state) ? 'disabled' : ''}>采用已审核配音重新排轴</button><span class="muted">先在配音导演中审核表演计划与试听版本。当前使用选定配音，额外停顿计入实际时长。</span></div>`;
        if (!state.timeline) {content.innerHTML += '<div class="empty">审核配音后安排镜头和字幕。</div>';return;}
        content.innerHTML += `<p>${state.timeline.duration.toFixed(2)} 秒 · ${state.timeline.total_frames} 帧 · 1080×1920 · H.264</p><p class="muted">${esc(state.timeline.subtitle_note)}</p>` + state.timeline.narration.map(n => `<section class="timeline-group"><b>${esc(n.narration_id)}</b><p>${esc(n.text)}</p><audio controls preload="none" src="${api()}/audio/${enc(n.narration_id)}?revision=${enc(state.revision)}"></audio><div class="track">${state.timeline.shots.filter(s=>s.narration_id===n.narration_id).map(s=>`<button data-crop="${esc(s.panel_id)}" data-nid="${esc(n.narration_id)}" style="flex:${s.frames}" ${disabled()?'disabled':''}>${esc(s.panel_id)}<br>${(s.frames/30).toFixed(2)}s · ${motionLabels[s.motion]}</button>`).join('')}</div></section>`).join('');
        return;
      }
      content.innerHTML = `<div class="toolbar"><label>TTS 引擎 <select id="engine"><option value="edge">Edge</option><option value="kokoro">Kokoro</option><option value="elevenlabs">ElevenLabs</option></select></label><label>声音 <input id="voice" value="${esc(state.tts_options?.voice || 'zh-CN-YunxiNeural')}"></label><button data-action="direct" ${disabled() ? 'disabled' : ''}>生成语音并安排时间轴</button><span class="muted">以实际音频为准，30fps</span></div>
      ${state.timeline ? `<p>${state.timeline.duration.toFixed(2)} 秒 · ${state.timeline.total_frames} 帧 · 1080×1920 · H.264</p><p class="muted">${esc(state.timeline.subtitle_note)}。淡入转场不占用额外时间。</p>${state.timeline.narration.map(n => `<section class="timeline-group"><b>${esc(n.narration_id)}</b> <span class="muted">${(n.start_frame/30).toFixed(2)}–${(n.end_frame/30).toFixed(2)} 秒 · 实测语音 ${n.audio_duration.toFixed(3)} 秒</span><p>${esc(n.text)}</p><audio controls preload="none" src="${api()}/audio/${enc(n.narration_id)}?revision=${enc(state.revision)}"></audio><div class="track">${state.timeline.shots.filter(s => s.narration_id === n.narration_id).map(s => `<button data-crop="${esc(s.panel_id)}" data-nid="${esc(n.narration_id)}" style="flex:${s.frames}" ${disabled() ? 'disabled' : ''}>${esc(s.panel_id)}<br>${(s.frames/30).toFixed(2)}s<br>${motionLabels[s.motion]} · ${s.transition === 'cut' ? '直切' : '淡入'}</button>`).join('')}</div></section>`).join('')}<details><summary>字幕时间安排</summary>${state.timeline.subtitles.map(s => `<p>${(s.start_frame/30).toFixed(2)}–${(s.end_frame/30).toFixed(2)}s ${esc(s.text)}</p>`).join('')}</details>` : '<div class="empty">先生成实际语音，再安排镜头。镜头时长可自动分配，也可在镜头编辑中指定。</div>'}`;
      $('#engine').value = state.tts_options?.engine || 'edge'; return;
    }
    const exp = state.export, exportUrl = exp ? `${api()}/exports/${exp.export_id}` : '';
    content.innerHTML = `<div class="toolbar"><button data-action="export" ${disabled() || !state.timeline || !approved ? 'disabled' : ''}>导出竖屏 MP4</button><label><input type="checkbox" id="export-quality"> 智能画质增强</label><label><input type="checkbox" id="export-quality-face"> 人脸修复</label><span class="muted">先审核全部镜头组，再生成时间轴。每次导出保留独立版本。</span></div>${exp ? `<div class="preview"><video controls playsinline preload="metadata" src="${exportUrl}/video.mp4"></video><div><h2>成片已完成校验</h2><p>1080×1920 · 9:16 · H.264 MP4</p><p>剧情覆盖 ${percent(exp.metrics.coverage_rate)} · 解码校验 ${exp.report.decoded_frames} 帧</p><p>最大音画偏差 ${(exp.sync.max_audio_visual_error*1000).toFixed(1)}ms</p><div class="row"><a class="button" href="${exportUrl}/video.mp4?download=1">下载视频</a><a class="button secondary" href="${exportUrl}/subtitles.srt?download=1">下载字幕</a><a class="button secondary" href="${exportUrl}/manifest.json?download=1">导出决策清单</a></div></div></div>` : '<div class="empty">完成镜头审核与时间轴后，这里将显示成片。</div>'}`;
  }
  function drawLibrary(query) {
    const selected = new Set(state.groups.flatMap(g => g.selected.map(s => s.panel_id)));
    $('#library').innerHTML = state.library.filter(p => JSON.stringify(p).toLowerCase().includes(query.toLowerCase())).map(p => `<article class="panel-card"><a href="${image(p.panel_id)}" target="_blank" rel="noopener"><img src="${image(p.panel_id)}" alt="来源分镜 ${esc(p.panel_id)}" loading="lazy"></a><p><b>${esc(p.panel_id)}</b> · ${p.width}×${p.height} ${selected.has(p.panel_id) ? '<span class="tag">已采用</span>' : ''}</p>${p.boundary_uncertain ? '<span class="tag warning">裁切边界待核对</span>' : ''}<p>${esc(p.event_ids.join('、') || '无剧情事件')}<br>${esc(p.actions.join('；'))}</p><p>${esc(p.dialogue.map(d => d.text).join(' / '))}</p></article>`).join('') || '<p>没有匹配结果。</p>';
  }
  async function load() {
    const token = generation;
    try {const value = await json(`${api()}${historyRevision ? '?revision='+enc(historyRevision) : ''}`); if(token !== generation) return;
      state = value; busy = Boolean(state.active_job); draw();
      if (busy) poll(state.active_job.job_id, token);
    } catch(e) {if(token === generation) {$('#editor').innerHTML = '<div id="message" class="status"></div>';message(e.message,true);}}
  }
  async function action(path, data = {}) {
    if (disabled()) return;
    const token = generation;
    busy = true;
    try {const value = await json(`${api()}/${path}`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:state.revision,...data})});
      if(token !== generation) return; draw(); message('正在处理，原图与先前审核版本均已保留。'); poll(value.job_id,token);
    } catch(e) {if(token !== generation) return;busy=false;draw();message(e.message,true);}
  }
  function poll(jid, token) {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      if(token !== generation) return;
      try {const job = await json(`${api()}/jobs/${enc(jid)}`);if(token !== generation) return;
        message(`${job.message}${job.total ? ` (${job.current}/${job.total})` : ''}`);
        if(job.status === 'running') return poll(jid,token);
        busy=false; await load(); if(job.status === 'failed') message(job.error,true);
      } catch(e) {if(token !== generation) return;busy=false;draw();message(e.message,true);}
    }, 800);
  }
  async function update(nid, selected) {await action(`groups/${enc(nid)}/update`,{selected});}
  $('#editor').onclick = async event => {
    const button = event.target.closest('button');if(!button || button.disabled) return;
    if(button.dataset.tab){tab=button.dataset.tab;draw();return;}
    if(button.dataset.action === 'refresh') return load();
    if(button.dataset.action === 'select') return action('select',{source_revision:state.source_revision || state.readiness?.source_revision,use_model:$('#use-model').checked});
    if(button.dataset.action === 'direct') return action('direct',state.voice_direction ? {} : {tts:{engine:$('#engine').value,voice:$('#voice').value}});
    if(button.dataset.action === 'export') return action('export', {quality_enhancement: {
      enabled: $('#export-quality').checked, face: $('#export-quality-face').checked
    }});
    const nid=button.dataset.nid;
    if(button.dataset.review) return action(`groups/${enc(nid)}/${button.dataset.review}`);
    if(button.dataset.crop) return openCrop(nid,button.dataset.crop);
    const selected = structuredClone(groupById(nid)?.selected || []);
    if(button.dataset.remove) return update(nid,selected.filter(s => s.panel_id !== button.dataset.remove));
    if(button.dataset.add) return update(nid,[...selected,{panel_id:button.dataset.add}]);
    if(button.dataset.move) {const i=Number(button.dataset.move);[selected[i-1],selected[i]]=[selected[i],selected[i-1]];return update(nid,selected);}
  };
  $('#editor').onchange = event => {if(event.target.id === 'history'){historyRevision=event.target.value || null;load();}};
  function openCrop(nid,pid) {
    const shot=structuredClone(groupById(nid).selected.find(s=>s.panel_id===pid));
    activeCrop={nid,pid,shot,image:new Image(),generation};
    $('#crop-error').textContent='';
    $('#crop-controls').innerHTML=`<p>来源 ${esc(pid)}</p><div class="row">${['X','Y','宽','高'].map((label,i)=>`<label>${label}（0–1）<input id="crop-${i}" type="number" min="0" max="1" step="0.001" value="${shot.crop[i]}"></label>`).join('')}</div><label>运动<select id="crop-motion">${Object.entries(motionLabels).map(([id,label])=>`<option value="${id}">${label}</option>`).join('')}</select></label><label>转场<select id="crop-transition"><option value="cut">直切</option><option value="fade">淡入（最多0.2秒）</option></select></label><label>指定秒数（留空自动）<input id="crop-duration" type="number" min="0.034" max="1200" step="0.033333" value="${shot.duration_frames ? shot.duration_frames/30 : ''}"></label><p class="muted">修改后的选择与裁切需重新审核。指定时长与实际音频冲突时，系统会要求调整。</p>`;
    if (groupById(nid).text_mapping) $('#crop-controls').insertAdjacentHTML('beforeend', `<label>对应文字范围（每行 起点:终点 | 承接方式 | 理由）<textarea id="crop-text-ranges" rows="5">${esc((shot.text_ranges || []).map(r=>`${r.start_char}:${r.end_char} | ${matchLabels[r.match_type || 'direct']} | ${r.reason || '原文直接对应'}`).join('\n'))}</textarea></label><p class="muted">承接方式：直接展示、过渡延续、吐槽反应、画面回扣。从0计数，末端不包含；表情算一个字符。文字由当前原文读取。</p>`);
    $('#crop-motion').value=shot.motion;$('#crop-transition').value=shot.transition;
    activeCrop.image.onload=()=>{if(!activeCrop || activeCrop.pid!==pid) return;const canvas=$('#crop-canvas');const scale=Math.min(1,600/activeCrop.image.width,640/activeCrop.image.height);canvas.width=Math.round(activeCrop.image.width*scale);canvas.height=Math.round(activeCrop.image.height*scale);paintCrop();};
    activeCrop.image.src=image(pid);$('#crop-dialog').showModal();
  }
  function paintCrop(){if(!activeCrop?.image.complete)return;const c=$('#crop-canvas'),ctx=c.getContext('2d');ctx.drawImage(activeCrop.image,0,0,c.width,c.height);const [x,y,w,h]=activeCrop.shot.crop;ctx.fillStyle='#153d1b44';ctx.fillRect(x*c.width,y*c.height,w*c.width,h*c.height);ctx.strokeStyle='#3bdd77';ctx.lineWidth=2;ctx.strokeRect(x*c.width,y*c.height,w*c.width,h*c.height);}
  $('#crop-controls').oninput=()=>{if(!activeCrop)return;activeCrop.shot.crop=[0,1,2,3].map(i=>Number($(`#crop-${i}`).value));paintCrop();};
  let drag;
  function point(event){const rect=$('#crop-canvas').getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];}
  $('#crop-canvas').onpointerdown=e=>{drag=point(e);e.target.setPointerCapture(e.pointerId);};
  $('#crop-canvas').onpointermove=e=>{if(!drag || !activeCrop)return;const p=point(e);activeCrop.shot.crop=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),Math.abs(p[0]-drag[0]),Math.abs(p[1]-drag[1])].map(v=>Math.round(v*1000)/1000);activeCrop.shot.crop.forEach((v,i)=>{$(`#crop-${i}`).value=v;});paintCrop();};
  $('#crop-canvas').onpointerup=()=>{drag=null;};$('#crop-canvas').onpointercancel=()=>{drag=null;};
  $('#crop-save').onclick=async()=>{const edit=activeCrop;if(!edit || edit.generation!==generation)return;
    const [x,y,w,h]=edit.shot.crop;if(![x,y,w,h].every(Number.isFinite)||x<0||y<0||w<=0||h<=0||x+w>1.000001||y+h>1.000001){$('#crop-error').textContent='请将裁切区域完整放在原图范围内。';return;}
    const rangeInput=$('#crop-text-ranges');
    if (rangeInput) {
      const chars=Array.from(groupById(edit.nid).text), ranges=[];
      let endBefore=-1;
      for (const line of rangeInput.value.split('\n').filter(s=>s.trim())) {
        const parts=line.split('|').map(s=>s.trim()),match=parts[0].match(/^(\d+)\s*:\s*(\d+)$/);
        if (!match) {$('#crop-error').textContent='文字范围请填写 起点:终点，每行一个。';return;}
        const start=Number(match[1]),end=Number(match[2]),text=chars.slice(start,end).join('');
        if (!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start<endBefore||start>=end||end>chars.length||!text.trim()) {$('#crop-error').textContent='文字范围越界、交叠或为空，请按原文检查。';return;}
        const previous=(edit.shot.text_ranges || []).find(r=>r.start_char===start&&r.end_char===end);
        const matchType=parts[1] ? Object.keys(matchLabels).find(k=>matchLabels[k]===parts[1]||k===parts[1]) : previous?.match_type || (edit.shot.context_only?'transition':'direct');
        const reason=parts.length>2 ? parts.slice(2).join('|').trim() : previous?.reason || '人工调整文字范围';
        if (!matchType||!reason) {$('#crop-error').textContent='请填写有效的承接方式与理由。';return;}
        ranges.push({start_char:start,end_char:end,text,match_type:matchType,reason});endBefore=end;
      }
      edit.shot.text_ranges=ranges;
    }
    edit.shot.motion=$('#crop-motion').value;edit.shot.transition=$('#crop-transition').value;const value=$('#crop-duration').value;edit.shot.duration_frames=value ? Math.round(Number(value)*30) : null;
    const selected=structuredClone(groupById(edit.nid).selected).map(s=>s.panel_id===edit.pid?edit.shot:s);$('#crop-dialog').close();await update(edit.nid,selected);
  };
  $('#project-form').onsubmit=event=>{event.preventDefault();clearTimeout(timer);generation++;busy=false;historyRevision=null;project=$('#project').value.trim();const url=new URL(location.href);url.searchParams.set('project',project);history.replaceState(null,'',url);load();};
  $('#project').value=new URLSearchParams(location.search).get('project') || 'first-chapter';
  project=$('#project').value;load();
})();
