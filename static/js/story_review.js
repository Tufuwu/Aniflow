/* Shared by existing Review Text and the lightweight first-chapter page. */
window.StoryReview = (() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const labels = {pending:'待人工审核', approved:'已审核 · 已锁定', rejected:'已驳回',
    passed:'事实核验通过', issues:'发现事实问题', uncertain:'存在不确定内容', unavailable:'核验未完成',
    unsupported:'无来源依据', character_confusion:'人物混淆', event_contradiction:'事件矛盾', wrong_causality:'错误因果'};
  const eventKinds = {current_action:'当前行动', world_rule:'世界规则或限制', recollection:'回忆',
    historical_explanation:'历史说明', unresolved:'尚未归并', non_story:'非剧情'};
  const eventStates = {supported:'已有分析支持', uncertain:'尚不确定', unknown:'未知', conflicting:'存在矛盾', needs_history:'待补历史'};
  const facetRoles = {premise:'前提与处境', goal:'已有目标', obstacle:'阻碍', choice:'选择', trigger:'转折触发',
    action:'行动', reaction:'反应', result:'结果', setting:'必要设定及作用', unknown:'未知限制'};
  let host, project, state, timer, chapter = 1, viewRevision = null, busy = false, generation = 0;
  const drafts = new Map();
  const noteDrafts = new Map();
  const polishDrafts = new Map();
  const polishDefaults = {humor_percent:35, suspense:'medium', narrative_style:'commentary', length:'storytelling', analyze_first:false};
  const legacyPolishDefaults = {humor_percent:25, suspense:'medium', narrative_style:'immersive', length:'balanced', analyze_first:false};
  const polishKey = () => `${project}/${chapter}`;
  const draftKey = sid => `${project}/${chapter}/${state?.script?.generation_basis || state?.plan?.bible_id || 'script'}/${sid}`;
  const endpoint = () => `/api/story/${encodeURIComponent(project)}` + (chapter === 1 ? '' : `/chapters/${chapter}`);
  async function json(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
    return data;
  }
  function message(value) {
    const target = host?.querySelector('[data-story-message]');
    if (target) target.textContent = value;
  }
  function isMounted() { return host && document.contains(host); }
  function evidence(pid) {
    return `${endpoint()}/evidence/${encodeURIComponent(pid)}?revision=${state.revision}`;
  }
  const profileStatus = {confirmed:'已确认',reviewed:'已核对',uncertain:'推测，未确认',unknown:'未知',
    needs_verification:'待核对',unclassified:'旧记录，未分类',superseded:'历史值，已更新'};
  function profileSources(ids) {
    return (ids || []).map(pid => `<a href="${evidence(pid)}" target="_blank" rel="noopener">${esc(pid)}</a>`).join('、');
  }
  function profileEntries(rows) {
    if (!rows.length) return '未知';
    return rows.map(r => `<p>${r.key ? `${esc(r.key)}：` : ''}${esc(r.value)} · ${esc(profileStatus[r.status] || r.status)}
      <br>来源 ${profileSources((r.source?.panels || []).map(p => p.source_panel_id)) || '待核对'}
      ${r.reveal_panel_id ? `；揭晓来源 ${esc(r.reveal_panel_id)}` : ''}</p>`).join('');
  }
  function profileCard(record) {
    const fields = {identity:'人物身份',personality:'人物性格',ability:'人物能力',state:'当前状态'};
    return `<article class="story-event"><strong>${esc(record.name.value || '姓名未知')} (${esc(record.character_id)})</strong>
      <p>人物姓名：${esc(record.name.value || '未知')} · ${esc(profileStatus[record.name.status] || record.name.status)}
        ${record.name.source_panel_ids.length ? `；来源 ${profileSources(record.name.source_panel_ids)}` : ''}</p>
      <table><tbody>${Object.entries(fields).map(([key,label]) => `<tr><th scope="row">${label}</th><td>
        ${profileEntries((record[key] || []).filter(r => r.current))}</td></tr>`).join('')}</tbody></table>
      <details><summary>待核对资料与未分类记录</summary>
        ${profileEntries(Object.keys(fields).flatMap(k => (record[k] || []).filter(r => !r.current)).concat(record.other || []))}
      </details></article>`;
  }
  function draw() {
    if (!isMounted()) return;
    const bible = state.bible, script = state.script;
    const polish = script?.narrative_review?.engagement_polish;
    const locked = (script?.segments || []).some(s => s.review_status === 'approved');
    const readOnly = Boolean(viewRevision) || busy;
    const notes = noteDrafts.get(project) || state.series_notes || {};
    const upgradePolishDefaults = !viewRevision && (polish?.prompt_version || 0) < 5 &&
      Object.entries(legacyPolishDefaults).every(([key,value]) => script?.polish_options?.[key] === value);
    const polishOptions = (!viewRevision && polishDrafts.get(polishKey())) ||
      (upgradePolishDefaults ? polishDefaults : script?.polish_options) || polishDefaults;
    const polishSelect = (key, title, choices) => `<label>${title} <select data-polish-option="${key}" ${readOnly || locked ? 'disabled' : ''}>
      ${choices.map(([value,label]) => `<option value="${esc(value)}" ${String(polishOptions[key]) === String(value) ? 'selected' : ''}>${esc(label)}</option>`).join('')}
      </select></label>`;
    host.innerHTML = `<section class="story-review">
      <h2>剧情理解与中文解说 · 第${chapter}话</h2>
      <p>一次图片分析 → 本话信息整合与事件还原 → 叙事规划 → 单稿写作与趣味性润色 → 严格原图核验 → 人工审核。</p>
      ${state.input?.kind === 'raw_slices' ? `<p class="story-warning">已保留 ${state.input.count} 张原始切片。请先在 AniFlow 完成拼接、分镜检查，或导入确认后的完整分镜。</p>` : ''}
      ${state.input?.kind === 'detected_panels' ? '<p class="story-warning">来源区域已保留整话内容，可先预分析和写草稿。请检查阅读顺序、截断和遗漏；确认分镜后才能审核锁定。</p>' : ''}
      ${viewRevision ? '<p class="story-warning">正在查看历史版本，只读。返回最新版本后可继续审核。</p>' : ''}
      <div class="story-toolbar">
        ${chapter === 1 ? `<a class="btn btn-secondary" href="/shot-review?project=${encodeURIComponent(project)}">进入剧情分镜剪辑</a>` : ''}
        <select data-story-style aria-label="解说风格" ${readOnly || locked ? 'disabled' : ''}>
          ${[['faithful','忠实叙事'],['payoff','爽点解说'],['suspense','高悬念']].map(([id, label]) => `<option value="${id}" ${script?.style === id ? 'selected' : ''}>${label}</option>`).join('')}
        </select>
        <button class="btn btn-secondary" data-action="analyze" ${readOnly || locked || state.input?.kind === 'raw_slices' ? 'disabled' : ''}>分析剧情</button>
        ${state.input?.kind === 'raw_slices' ? `<button class="btn btn-secondary" data-action="prepare" ${readOnly ? 'disabled' : ''}>拼接并检测分镜</button>` : ''}
        ${state.input?.kind === 'detected_panels' ? `<button class="btn btn-secondary" data-action="confirm_panels" ${readOnly ? 'disabled' : ''}>已核对分镜完整与阅读顺序</button>` : ''}
        <button class="btn btn-secondary" data-action="integrate" ${readOnly || locked || !bible ? 'disabled' : ''}>本话信息整合与事件还原</button>
        <button class="btn btn-primary" data-action="generate" ${readOnly || locked || !bible ? 'disabled' : ''}>生成解说稿</button>
        <button class="btn btn-secondary" data-action="revise" ${readOnly || locked || !script?.segments?.length ? 'disabled' : ''}>口播编剧重写</button>
        <button class="btn btn-ghost" data-action="refresh" ${busy ? 'disabled' : ''}>返回最新 / 刷新</button>
        ${script ? `<a class="btn btn-secondary" href="${endpoint()}/export" target="_blank" rel="noopener">导出最新文案与证据索引</a>` : ''}
      </div>
      <p data-story-message role="status" aria-live="polite">${busy ? '正在处理，请稍候…' : locked ? '已审核段落受保护，不能编辑或覆盖。全部锁定后进入剧情分镜剪辑。' : '第四阶段完成后直接逐段人工审核锁定，再进行分镜、配音、时间轴和视频导出。'}</p>
      <details class="story-details"><summary>生成前资料 · 漫画背景与主要人物</summary>
        <p>供本项目各话生成前参考，事实与人物对应仍会核对原图。更新资料不会改写旧稿或历史版本。</p>
        <label>背景资料<textarea data-series-note="background" rows="4" ${readOnly ? 'readonly' : ''}>${esc(notes.background)}</textarea></label>
        <label>主要人物资料<textarea data-series-note="characters" rows="4" ${readOnly ? 'readonly' : ''}>${esc(notes.characters)}</textarea></label>
        <button class="btn btn-secondary" data-save-notes ${readOnly ? 'disabled' : ''}>保存生成前资料</button>
      </details>
      ${bible?.prior_context?.status === 'needs_review' ? `<p class="story-warning">前话资料待核对：${esc(bible.prior_context.reason)}</p>` : ''}
      ${(state.character_records || []).length ? `<details class="story-details"><summary>主要人物档案 · 姓名、身份、性格、能力</summary>
        ${(state.character_records || []).map(profileCard).join('')}</details>` : ''}
      ${script ? `<details class="story-details"><summary>世界背景与主线事件 · ${state.memory_usable ? '可供下一话参考，仍需人工核对' : '待核对或来源核验未完成'}</summary>
        <p>${esc(state.chapter_memory?.error || '')}</p>
        <h3>世界观概述</h3>${(state.world_background?.overview || []).map(s => `<p>${esc(s.text)} · ${esc(profileStatus[s.status] || s.status)}<br>来源 ${profileSources((s.source?.panels || []).map(p => p.source_panel_id))}</p>`).join('') || '<p>尚无已整理的世界观资料。</p>'}
        <h3>主线事件</h3>${(state.world_background?.main_events || []).map(s => `<p>${esc(({goal:'目标',conflict:'冲突',turn:'转折',action:'行动',outcome:'结果'})[s.role] || '')}${s.role ? '：' : ''}${esc(s.text)} · ${esc(profileStatus[s.status] || s.status)}<br>来源 ${profileSources((s.source?.panels || []).map(p => p.source_panel_id))}</p>`).join('') || '<p>尚无已整理的主线事件。</p>'}
        ${(state.chapter_memory?.open_threads || []).map(s => `<p>未完成目标：${esc(s.text)}</p>`).join('')}
        <button class="btn btn-secondary" data-action="memory" ${readOnly ? 'disabled' : ''}>重新整理并核对本话摘要</button>
      </details>` : ''}
      <details class="story-details"><summary>版本历史（${state.history.length}）</summary>
        ${state.history.map(h => `<button class="btn btn-ghost" data-history="${h.revision}" ${busy ? 'disabled' : ''}>${esc(h.action)} · ${esc(h.updated_at)}</button>`).join('')}
      </details>
      ${state.input ? `<details class="story-details"><summary>查看输入图片（${state.input.count}）</summary><div style="display:flex;flex-wrap:wrap;gap:12px">${Array.from({length:state.input.count}, (_,i) => `<a href="${endpoint()}/inputs/${i}" target="_blank" rel="noopener"><img class="story-proof" src="${endpoint()}/inputs/${i}" alt="输入 ${i+1}" loading="lazy"><p>输入 ${i+1}</p></a>`).join('')}</div></details>` : ''}
      ${bible ? `<details class="story-details"><summary>Story Bible · ${bible.panels.length} 个分镜 / ${bible.characters.length} 个人物 / ${bible.events.length} 个事件</summary>
        <h3>人物识别记录（核验用，包含配角）</h3>${bible.characters.map(c => `<p><strong>${esc(c.name)} (${esc(c.character_id)})</strong>：${esc(c.description)}<br>来源 ${esc(c.source_panel_ids.join('、'))}${c.name_status === 'unknown' ? '<br>姓名未确认，可使用有依据的稳定称呼' : ''}${(c.label_claims || []).map(l => `<br>称呼：${esc(l.label)} · ${esc(l.basis)}（来源 ${esc(l.reveal_panel_id)} 起）`).join('')}${(c.name_claims || []).map(n => `<br>姓名：${esc(n.name)} · 原文“${esc(n.quote)}”；归属依据：${esc(n.attribution?.basis)}（来源 ${esc(n.reveal_panel_id)} 揭晓）`).join('')}</p>`).join('')}
        ${bible.identity_review?.status === 'needs_review' ? `<p class="story-warning">人物对应仍有待核对内容：${esc(bible.identity_review.error || JSON.stringify(bible.identity_review.unresolved || []))}</p>` : ''}
        ${(bible.relationships || []).map(r => `<p>关系：${esc(r.from_id)} → ${esc(r.to_id)} · ${esc(r.relation)}；原文“${esc(r.quote)}”（来源 ${esc(r.source_panel_id)}）</p>`).join('')}
        <h3>${bible.event_integration ? '本话信息整合与事件还原' : '原始分析事件时间线'}</h3>
        ${bible.event_integration ? `<p>原始事件、对白和原图保留在分析版本 ${esc(bible.event_integration.source_revision)}；当前为本话整合结果。</p>` : ''}
        ${bible.events.map(e => `<article class="story-event"><strong>${esc(e.event_id)}</strong> ${esc(e.summary)}
          <p>动作：${esc(e.actions.join('；'))}</p><p>来源：${esc(e.source_panel_ids.join('、'))} · 重要性 ${e.importance}/5</p>
          ${e.kind ? `<p>类型：${esc(eventKinds[e.kind] || e.kind)} · 状态：${esc(eventStates[e.status] || e.status)} · 原作时间：${esc(e.time?.original_time)}（${esc(eventStates[e.time?.status] || e.time?.status)}） · 揭示来源：${esc(e.reveal_panel_id)}</p>
            ${(e.facets || []).map(f => `<p>${esc(facetRoles[f.role] || f.role)}：${esc(f.text)} · ${esc(eventStates[f.status] || f.status)}</p>`).join('')}` : ''}
          ${e.narrative_eligible === false ? `<p class="story-warning">非剧情，不用于写稿：${esc(e.exclusion_reason)}</p>` : ''}
          <p>因果：${esc(e.causes.map(c => `${c.event_id}：${c.evidence}`).join('；') || '无明确依据，未推定因果')}</p>
          ${(e.internal_causes || []).map(c => `<p>合并内部已有因果：${esc(c.evidence)}</p>`).join('')}
          ${(e.pending_causes || []).map(c => `<p class="story-warning">因果待核对：${esc(eventStates[c.status] || c.status)}${c.evidence ? ` · ${esc(c.evidence)}` : ''}</p>`).join('')}
          <details><summary>查看事件原图、对白与识别疑点</summary>${e.source_panel_ids.map(pid => {
            const p = bible.panels.find(p => p.source_panel_id === pid);
            return `<a href="${evidence(pid)}" target="_blank" rel="noopener"><img class="story-proof" src="${evidence(pid)}" alt="原始分镜 ${esc(pid)}" loading="lazy"></a><p>${esc(p.dialogue.map(d => d.text).join(' / '))}</p><p>${esc(p.uncertainties.join('；'))}</p>`;
          }).join('')}</details></article>`).join('')}
      </details>` : '<p>尚无剧情分析。请先确认分镜，再开始分析。</p>'}
      ${state.plan ? `<details class="story-details"><summary>叙事规划 · ${state.plan.beats.length} 个关键事件</summary>
        <p>开场 ${esc(state.plan.opening_event_id)} → 高潮 ${esc(state.plan.climax_event_id)} → 结尾 ${esc(state.plan.ending_event_id)}</p>
        <p>${esc(state.plan.ending_rule)}；${esc(state.plan.budget_note)}</p>
        ${state.plan.beats.map(b => `<p>${esc((b.event_ids || [b.event_id]).join('、'))} · ${esc(b.title || b.section)} · ${esc(b.reason)}</p>`).join('')}
      </details>` : ''}
      <details class="story-details"><summary>第四阶段 · 口播编剧重写</summary>
        <div class="story-toolbar">
          ${polishSelect('humor_percent','幽默倾向',[[0,'0%'],[10,'10%'],[25,'25%'],[40,'40%'],[50,'50%']].concat(
            [0,10,25,40,50].includes(polishOptions.humor_percent) ? [] : [[polishOptions.humor_percent,`${polishOptions.humor_percent}%`]]))}
          ${polishSelect('suspense','悬念',[['low','自然'],['medium','适中'],['high','较强']])}
          ${polishSelect('narrative_style','叙述',[['immersive','沉浸讲故事'],['commentary','旁白吐槽'],['fast_paced','快节奏']])}
          ${polishSelect('length','篇幅',[['storytelling','完整解说 90%–110%'],['concise','精简 80%–100%'],['balanced','均衡 100%–125%'],['expanded','扩写 125%–150%']])}
          <label><input type="checkbox" data-polish-option="analyze_first" ${polishOptions.analyze_first ? 'checked' : ''} ${readOnly || locked ? 'disabled' : ''}>先设计包袱和节奏（增加一次模型调用）</label>
        </div>
        <p>按完整解说示范重写，展开反差、旁白吐槽和口播停顿。篇幅为参考范围；修改参数后点击按钮生效。</p>
      </details>
      ${script ? `<details class="story-details"><summary>趣味性润色 · ${polish?.version === 1 && polish.status === 'completed' ? '已完成' : '尚未完成或正文已更新'}</summary>
        <p>${polish?.status === 'completed' ? (polish.changed_event_ids.length ? `直接修改 ${polish.changed_event_ids.length} 段，其余原文保留。` : '本次保持原稿，无需修改。') : esc(script.narrative_review?.error || '可按需直接润色当前稿。')}</p>
      </details>` : ''}
      <div class="story-segments">${(script?.segments || []).map(segment => {
        const approved = segment.review_status === 'approved', check = segment.fact_check;
        const primary = [...new Set(bible.events.filter(e => (segment.event_ids || [segment.event_id]).includes(e.event_id)).flatMap(e => e.source_panel_ids))];
        const extra = segment.source_panel_ids.filter(pid => !primary.includes(pid));
        const key = draftKey(segment.segment_id);
        if (drafts.get(key) === segment.text || approved) drafts.delete(key);
        const draft = !viewRevision && !approved && drafts.has(key) ? drafts.get(key) : segment.text;
        return `<article class="story-segment" data-segment="${esc(segment.segment_id)}">
          <div class="story-proof-column">${primary.map(pid => `<a href="${evidence(pid)}" target="_blank" rel="noopener"><img src="${evidence(pid)}" alt="原始分镜 ${esc(pid)}（点击放大）" loading="lazy"></a>`).join('')}
          ${extra.length ? `<details><summary>因果与背景原图（${extra.length}）</summary>${extra.map(pid => `<a href="${evidence(pid)}" target="_blank" rel="noopener"><img src="${evidence(pid)}" alt="背景来源 ${esc(pid)}" loading="lazy"></a>`).join('')}</details>` : ''}
          <small>点击原图放大 · 来源 ${esc(segment.source_panel_ids.join('、'))}</small></div>
          <div class="story-text-column"><p><strong>${esc(segment.event_id)} · ${esc(segment.section)}</strong></p>
            <p class="story-status">${labels[segment.review_status]}</p>
            <p>对照左侧原图确认文案，审核通过后锁定，并在选镜工作台审核实际展示画面。</p>
            <textarea aria-label="${esc(segment.segment_id)} 解说稿" rows="4" ${approved || readOnly ? 'readonly' : ''}>${esc(draft)}</textarea>
            ${draft !== segment.text ? '<p class="story-warning">有未保存修改，请先保存再审核。</p>' : ''}
            <div class="story-toolbar">${[['edit','保存修改'],['regenerate','单段重新生成'],['approve','审核通过并锁定'],['reject','驳回']].map(([action,label]) => `<button class="btn btn-secondary" data-segment-action="${action}" ${readOnly || approved || (action === 'approve' && state.input && state.input.kind !== 'panels') ? 'disabled' : ''}>${label}</button>`).join('')}</div>
            ${check ? `<details class="story-check"><summary>已有模型核验记录（供参考） · ${labels[check.status] || '未知'}</summary>${check.error ? `<p>${esc(check.error)}</p>` : ''}
            ${(check.sentences || []).map(s => `<p><strong>${s.sentence_index + 1}. ${esc(s.text)}</strong><br>${esc(s.evidence || '')}<br>${(s.issues || []).map(i => `${labels[i.type]}：${esc(i.detail)}`).join('<br>')}</p>`).join('')}</details>` : ''}
          </div></article>`;
      }).join('')}</div>
    </section>`;
    host.querySelectorAll('[data-action]').forEach(button => button.onclick = () => {
      if (button.dataset.action === 'refresh') { viewRevision = null; reload().catch(e => message(e.message)); }
      else run(button.dataset.action).catch(e => message(e.message));
    });
    host.querySelectorAll('[data-polish-option]').forEach(input => input.onchange = () => {
      const options = {...polishOptions};
      host.querySelectorAll('[data-polish-option]').forEach(control => {
        const key = control.dataset.polishOption;
        options[key] = key === 'analyze_first' ? control.checked : key === 'humor_percent' ? Number(control.value) : control.value;
      });
      polishDrafts.set(polishKey(),options);
    });
    host.querySelectorAll('[data-series-note]').forEach(area => area.oninput = () => {
      noteDrafts.set(project, {background:host.querySelector('[data-series-note="background"]').value,
        characters:host.querySelector('[data-series-note="characters"]').value});
    });
    host.querySelector('[data-save-notes]').onclick = async () => {
      try {
        await json(`/api/story/${encodeURIComponent(project)}/context`, {method:'PUT',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({revision:state.series_notes?.revision ?? null,background:notes.background || '',characters:notes.characters || '',
            ...noteDrafts.get(project)})});
        noteDrafts.delete(project); await reload(); message('资料已保存，下次生成时提供给模型；旧稿与人审状态保持原样。');
      } catch (error) { message(error.message); }
    };
    host.querySelectorAll('[data-history]').forEach(button => button.onclick = () => {
      viewRevision = button.dataset.history; reload().catch(e => message(e.message));
    });
    host.querySelectorAll('[data-segment-action]').forEach(button => button.onclick = () => {
      const card = button.closest('[data-segment]');
      const segment = state.script.segments.find(s => s.segment_id === card.dataset.segment);
      if (button.dataset.segmentAction !== 'edit' && card.querySelector('textarea').value !== segment.text) {
        message('此段有尚未保存的修改，请先“保存修改”。'); return;
      }
      run(`segments/${encodeURIComponent(card.dataset.segment)}/${button.dataset.segmentAction}`,
          {text:card.querySelector('textarea').value}).catch(e => message(e.message));
    });
    host.querySelectorAll('[data-segment] textarea').forEach(area => area.oninput = () => {
      drafts.set(draftKey(area.closest('[data-segment]').dataset.segment), area.value);
    });
  }
  async function reload() {
    const token = generation;
    const data = await json(endpoint() + (viewRevision ? `?revision=${viewRevision}` : ''));
    if (token !== generation) return;
    state = data; busy = Boolean(state.active_job); draw();
    if (state.active_job) poll(state.active_job.job_id, token);
  }
  async function poll(id, token) {
    clearTimeout(timer);
    if (!isMounted() || token !== generation) return;
    try {
      const job = await json(`${endpoint()}/jobs/${id}`);
      if (token !== generation) return;
      message(`${job.message} ${job.total ? `${job.current}/${job.total}` : ''}`);
      if (job.status === 'running') { timer = setTimeout(() => poll(id, token), 1500); return; }
      busy = false; await reload();
      if (job.error) message(job.error);
    } catch (error) { busy = false; await reload(); message(error.message); }
  }
  async function run(action, payload = {}) {
    if (busy || viewRevision) return;
    if (['generate','integrate','analyze','verify','revise','memory'].includes(action) && (state.script?.segments || []).some(s => drafts.has(draftKey(s.segment_id)))) {
      message('有尚未保存的旁白修改，请先逐段保存。'); return;
    }
    if (action === 'analyze' && noteDrafts.has(project)) {
      message('有尚未保存的背景或人物资料，请先保存。'); return;
    }
    const style = host.querySelector('[data-story-style]').value;
    if (['generate','revise'].includes(action)) {
      const options = {};
      host.querySelectorAll('[data-polish-option]').forEach(input => {
        const key = input.dataset.polishOption;
        options[key] = key === 'analyze_first' ? input.checked : key === 'humor_percent' ? Number(input.value) : input.value;
      });
      polishDrafts.set(polishKey(),options);
      payload = {polish_options:options, ...payload};
    }
    busy = true; draw();
    try {
      const job = await json(`${endpoint()}/${action}`, {method:'POST',
        headers:{'Content-Type':'application/json'}, body:JSON.stringify({revision:state.revision, style, ...payload})});
      await poll(job.job_id, generation);
    } catch (error) { busy = false; draw(); throw error; }
  }
  async function mount(element, name, chapterNumber = 1) {
    generation++; clearTimeout(timer); host = element; project = name; chapter = Number(chapterNumber); viewRevision = null; busy = false;
    host.innerHTML = '<p data-story-message role="status">正在加载剧情审核…</p>';
    try { await reload(); } catch (error) { message(error.message); }
  }
  return {mount};
})();
