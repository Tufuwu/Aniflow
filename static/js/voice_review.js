/* Voice plans and auditions remain separate from immutable approved narration. */
window.VoiceReview = (() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const styles = {suspense:'悬疑讲述',hotblood:'热血爽文',deadpan:'冷面吐槽'};
  const emotions = {neutral:'平稳',tense:'紧张',surprised:'惊讶',relieved:'释然',excited:'振奋',sad:'低落',sarcastic:'戏谑',serious:'严肃',calm:'平静'};
  const preferences = new Map();
  let capabilities;
  const ready = state => !state.voice_direction ? !state.voice_review_required : state.groups.every(g => {
    const plan=state.voice_direction.plans[g.narration_id], id=state.voice_selected?.[g.narration_id];
    const take=state.voice_samples?.[g.narration_id]?.find(t=>t.take_id===id);
    return plan?.review_status==='approved' && take?.review_status==='approved';
  });
  async function request(url, options) {
    const response=await fetch(url,options), data=await response.json();
    if(!response.ok)throw new Error(data.error || '配音请求失败'); return data;
  }
  function mount(host, ctx) {
    const {state}=ctx, plans=state.voice_direction?.plans || {}, options=state.voice_options || {};
    if(!preferences.has(ctx.project))preferences.set(ctx.project,{style:state.voice_direction?.style || 'suspense',use_model:true,
      engine:options.engine || 'edge',voice:options.voice || (options.engine==='elevenlabs'?'':'zh-CN-YunxiNeural'),model:options.model || 'eleven_v4'});
    const pref=preferences.get(ctx.project), disabled=ctx.disabled() ? 'disabled' : '';
    const choices=(items,value)=>Object.entries(items).map(([id,label])=>`<option value="${id}" ${id===value?'selected':''}>${label}</option>`).join('');
    host.innerHTML=`<section class="voice-intro"><h2>配音导演</h2><p>固定主播风格，逐句调整表演。原稿保持不变，人工修改和审核会锁定对应配音计划。</p>
      <div class="toolbar"><label>主播风格<select id="voice-style" ${disabled}>${choices(styles,pref.style)}</select></label>
      <label><input id="voice-use-model" type="checkbox" ${pref.use_model?'checked':''} ${disabled}> AI 根据剧情标注（取消则生成规则草案）</label>
      <button data-voice="plan" ${disabled}>${state.voice_direction?'重新标注未锁定配音':'生成逐句表演计划'}</button></div>
      <div class="toolbar"><label>语音服务<select id="performance-engine" ${disabled}>${choices({elevenlabs:'ElevenLabs · 情绪配音',edge:'Edge · 基础预演'},pref.engine)}</select></label>
      <label>声音 ID<input id="performance-voice" list="performance-voices" value="${esc(pref.voice)}" placeholder="ElevenLabs 声音 ID 或 Edge 声音名" ${disabled}></label><datalist id="performance-voices"></datalist>
      <label>情绪模型（Edge 不使用）<select id="performance-model" ${disabled || pref.engine==='edge'?'disabled':''}>${choices({eleven_v4:'Eleven v4',eleven_v3:'Eleven v3'},pref.model)}</select></label>
      <button class="secondary" data-voice="voices" ${disabled}>读取 ElevenLabs 声音列表</button></div>
      <p class="muted" id="voice-service-status">${pref.engine==='edge'?'Edge 只执行语速和额外停顿；情绪、重音不会传给基础引擎。':'ElevenLabs 使用情绪和重音提示；语速为表演意图，实际效果须试听。'} ${capabilities?capabilities.elevenlabs_ready?'ElevenLabs 密钥已配置。':'ElevenLabs 密钥未配置。':''}</p>
      <details><summary>配置 ElevenLabs 密钥（仅保存在当前服务内存）</summary><div class="row"><input id="performance-key" type="password" autocomplete="off" aria-label="ElevenLabs API 密钥" placeholder="输入 API 密钥" ${disabled}><button data-voice="key" ${disabled}>应用密钥</button></div><p class="muted">密钥不写入漫画项目或剪辑历史。也可以通过 ELEVENLABS_API_KEY 环境变量配置。</p></details></section>
      ${state.voice_direction?state.groups.map(g=>renderGroup(g,plans[g.narration_id],ctx,disabled)).join(''):'<div class="empty">先生成配音计划。可以调用已配置的剧情模型，或使用明确标注的规则草案。</div>'}
      ${state.voice_direction?`<div class="toolbar"><button data-voice="timeline" ${disabled || !ready(state)?'disabled':''}>采用已审核配音安排时间轴</button><span class="muted">每段需审核表演计划、选择试听版本，并确认已试听。</span></div>`:''}`;
    const savePreferences=()=>{pref.style=host.querySelector('#voice-style').value;pref.use_model=host.querySelector('#voice-use-model').checked;
      pref.engine=host.querySelector('#performance-engine').value;pref.voice=host.querySelector('#performance-voice').value.trim();pref.model=host.querySelector('#performance-model').value;};
    host.onchange=e=>{if(['voice-style','voice-use-model','performance-engine','performance-voice','performance-model'].includes(e.target.id)){
      savePreferences();if(e.target.id==='performance-engine'){if(pref.engine==='edge'&&!pref.voice)pref.voice='zh-CN-YunxiNeural';mount(host,ctx);}}};
    host.onclick=async event=>{
      const button=event.target.closest('button[data-voice]');if(!button || button.disabled)return;
      const kind=button.dataset.voice,nid=button.dataset.nid;savePreferences();
      const route=action=>`voice/${encodeURIComponent(nid)}/${action}`;
      try{
        if(kind==='key'){
          const input=host.querySelector('#performance-key');await request('/api/editing/voice-key',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:input.value})});input.value='';
          capabilities=await request('/api/editing/voice-capabilities');ctx.message('ElevenLabs 密钥已应用到当前服务内存。');return;
        }
        if(kind==='voices'){
          const value=await request('/api/editing/voices');host.querySelector('#performance-voices').innerHTML=value.voices.map(v=>`<option value="${esc(v.voice_id)}">${esc(v.name)}</option>`).join('');ctx.message(`已读取 ${value.voices.length} 个声音；在声音 ID 输入框选择。`);return;
        }
        if(kind==='plan')return ctx.action('voice/plan',{style:pref.style,use_model:pref.use_model});
        if(kind==='timeline')return ctx.action('direct');
        if(kind==='audition'||kind==='compare'||kind==='fresh'){
          if(!pref.voice)throw new Error('请先填写声音 ID，或读取声音列表。');
          if(pref.engine==='elevenlabs'&&capabilities&&!capabilities.elevenlabs_ready)throw new Error('请先配置 ElevenLabs 密钥，再生成情绪配音试听。');
          return ctx.action(route('audition'),{tts:{engine:pref.engine,voice:pref.voice,model:pref.model},compare:kind==='compare',fresh:kind==='fresh'});
        }
        if(kind==='save'){
          const card=button.closest('[data-voice-group]');
          const units=[...card.querySelectorAll('[data-voice-unit]')].map(row=>({
            emotion:row.querySelector('[data-field=emotion]').value,intensity:Number(row.querySelector('[data-field=intensity]').value),
            rate:Number(row.querySelector('[data-field=rate]').value),pause_after_ms:Number(row.querySelector('[data-field=pause_after_ms]').value),
            emphasis:row.querySelector('[data-field=emphasis]').value.split(/[,，]/).map(s=>s.trim()).filter(Boolean),reason:row.querySelector('[data-field=reason]').value}));
          return ctx.action(route('update'),{units});
        }
        if(kind==='select')return ctx.action(route('select_take'),{take_id:button.dataset.take});
        if(kind==='approve-audio'){
          const card=button.closest('[data-voice-group]');return ctx.action(route('approve_audio'),{
            listened:card.querySelector('[data-listened]').checked,review_note:card.querySelector('[data-review-note]').value});
        }
        if(kind==='approve'||kind==='unlock')return ctx.action(route(kind));
      }catch(error){ctx.message(error.message,true);}
    };
    if(!capabilities)request('/api/editing/voice-capabilities').then(value=>{
      capabilities=value;if(host.isConnected){const status=host.querySelector('#voice-service-status');if(status)status.textContent+=value.elevenlabs_ready?' ElevenLabs 已配置。':' ElevenLabs 未配置密钥。';}
    }).catch(error=>ctx.message(error.message,true));
  }
  function renderGroup(group,plan,ctx,disabled){
    if(!plan)return '';
    const nid=group.narration_id, samples=ctx.state.voice_samples?.[nid] || [], selected=ctx.state.voice_selected?.[nid];
    const active=samples.find(t=>t.take_id===selected);
    return `<article class="group voice-group" data-voice-group="${esc(nid)}"><div class="group-header"><h2>${esc(nid)} <span class="tag">${styles[plan.style]}</span><span class="tag">${plan.origin==='model'?'AI 剧情标注':'规则草案'} · ${plan.review_status==='approved'?'计划已审核':plan.locked?'人工锁定':'待审核'}</span></h2><div class="row"><button data-voice="approve" data-nid="${esc(nid)}" ${disabled || plan.review_status==='approved'?'disabled':''}>审核表演计划</button><button class="secondary" data-voice="unlock" data-nid="${esc(nid)}" ${disabled || !plan.locked?'disabled':''}>解锁供 AI 重标</button></div></div>
      ${plan.units.map((u,i)=>`<div class="voice-unit" data-voice-unit="${i}"><blockquote>${esc(u.text)}</blockquote><div class="voice-fields">
      <label>语气<select data-field="emotion" ${disabled}>${Object.entries(emotions).map(([id,label])=>`<option value="${id}" ${u.emotion===id?'selected':''}>${label}</option>`).join('')}</select></label>
      <label>情绪强度<input data-field="intensity" type="number" min="0" max="1" step="0.1" value="${u.intensity}" ${disabled}></label>
      <label>语速意图<input data-field="rate" type="number" min="0.75" max="1.3" step="0.01" value="${u.rate}" ${disabled}></label>
      <label>额外停顿（毫秒）<input data-field="pause_after_ms" type="number" min="0" max="1500" step="10" value="${u.pause_after_ms}" ${disabled}></label>
      <label>重音词（逗号分隔）<input data-field="emphasis" value="${esc(u.emphasis.join('，'))}" ${disabled}></label></div>
      <label>安排理由<input class="voice-reason" data-field="reason" value="${esc(u.reason)}" maxlength="500" ${disabled}></label><p class="muted">剧情依据：${esc(u.evidence_event_ids.join('、'))}</p></div>`).join('')}
      <div class="row"><button class="secondary" data-voice="save" data-nid="${esc(nid)}" ${disabled}>保存修改并锁定</button><button data-voice="audition" data-nid="${esc(nid)}" ${disabled}>生成本段试听</button><button class="secondary" data-voice="fresh" data-nid="${esc(nid)}" ${disabled}>生成新的试听版本</button><button class="secondary" data-voice="compare" data-nid="${esc(nid)}" ${disabled}>三种风格对比试听</button></div>
      <p class="muted">先保存修改再生成试听。三种风格对比使用同一段原文的规则草案，不覆盖当前计划；ElevenLabs 会生成3次配音。</p>
      <div class="voice-takes">${samples.map(t=>`<section class="voice-take ${t.take_id===selected?'chosen':''}"><b>${styles[t.plan.style]} · ${t.duration.toFixed(2)} 秒</b><p class="muted">${t.options.engine==='edge'?'基础语速预演':t.options.model} · ${esc(t.options.voice)}<br>${t.review_status==='approved'?'配音已审核并锁定':t.take_id===selected?'当前选定，待试听审核':'候选试听'}</p><audio controls preload="none" src="${ctx.api()}/voice-audio/${encodeURIComponent(nid)}/${t.take_id}?revision=${ctx.state.revision}"></audio><p class="muted">${t.quality.transcript_status==='not_run'?'未启用自动转写，需人工核对原文':t.quality.transcript_status==='matched'?'转写与原文一致':'转写发现差异，请核对'}${t.quality.clipping_suspected?'；疑似音频削波':''}</p>${t.quality.transcript?`<details><summary>转写结果</summary><p>${esc(t.quality.transcript)}</p></details>`:''}<button class="secondary" data-voice="select" data-nid="${esc(nid)}" data-take="${t.take_id}" ${disabled || t.take_id===selected?'disabled':''}>${t.take_id===selected?'已选用此版本':'选用此版本及表演计划'}</button></section>`).join('')}</div>
      ${active?`<div class="voice-accept"><label><input type="checkbox" data-listened ${disabled}> 我已试听选定版本，确认无漏读、错读或情绪标签被念出</label><label>试听结论（自动检查发现差异时必填）<input data-review-note maxlength="1000" value="${esc(active.review_note || '')}" ${disabled}></label><button data-voice="approve-audio" data-nid="${esc(nid)}" ${disabled || active.review_status==='approved'?'disabled':''}>审核并锁定这份配音</button></div>`:''}</article>`;
  }
  return {mount,ready};
})();
