/* Shared by existing Review Text and the lightweight first-chapter page. */
window.StoryReview = (() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const labels = {pending:'待人工审核', approved:'已审核 · 已锁定', rejected:'已驳回',
    passed:'事实核验通过', issues:'发现事实问题', uncertain:'存在不确定内容', unavailable:'核验未完成',
    unsupported:'无来源依据', character_confusion:'人物混淆', event_contradiction:'事件矛盾', wrong_causality:'错误因果'};
  let host, project, state, timer, viewRevision = null, busy = false, generation = 0;
  const drafts = new Map();
  const draftKey = sid => `${project}/${sid}`;
  const endpoint = () => `/api/story/${encodeURIComponent(project)}`;
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
  function draw() {
    if (!isMounted()) return;
    const bible = state.bible, script = state.script;
    const locked = (script?.segments || []).some(s => s.review_status === 'approved');
    const readOnly = Boolean(viewRevision) || busy;
    host.innerHTML = `<section class="story-review">
      <h2>剧情理解与中文解说 · 第一话</h2>
      <p>逐分镜分析 → 事件规划 → 中文写稿 → 逐句核验 → 人工审核。完成后可导出文案。</p>
      ${state.input?.kind === 'raw_slices' ? `<p class="story-warning">已保留 ${state.input.count} 张原始切片。请先在 AniFlow 完成拼接、分镜检查，或导入确认后的完整分镜。</p>` : ''}
      ${state.input?.kind === 'detected_panels' ? '<p class="story-warning">来源区域已保留整话内容，可先预分析和写草稿。请检查阅读顺序、截断和遗漏；确认分镜后才能审核锁定。</p>' : ''}
      ${viewRevision ? '<p class="story-warning">正在查看历史版本，只读。返回最新版本后可继续审核。</p>' : ''}
      <div class="story-toolbar">
        <select data-story-style aria-label="解说风格" ${readOnly || locked ? 'disabled' : ''}>
          ${[['faithful','忠实叙事'],['payoff','爽点解说'],['suspense','高悬念']].map(([id, label]) => `<option value="${id}" ${script?.style === id ? 'selected' : ''}>${label}</option>`).join('')}
        </select>
        <button class="btn btn-secondary" data-action="analyze" ${readOnly || locked || state.input?.kind === 'raw_slices' ? 'disabled' : ''}>分析剧情</button>
        ${state.input?.kind === 'raw_slices' ? `<button class="btn btn-secondary" data-action="prepare" ${readOnly ? 'disabled' : ''}>拼接并检测分镜</button>` : ''}
        ${state.input?.kind === 'detected_panels' ? `<button class="btn btn-secondary" data-action="confirm_panels" ${readOnly ? 'disabled' : ''}>已核对分镜完整与阅读顺序</button>` : ''}
        <button class="btn btn-primary" data-action="generate" ${readOnly || locked || !bible ? 'disabled' : ''}>规划并生成解说</button>
        <button class="btn btn-ghost" data-action="refresh" ${busy ? 'disabled' : ''}>返回最新 / 刷新</button>
        ${script ? `<a class="btn btn-secondary" href="${endpoint()}/export" target="_blank" rel="noopener">导出最新文案与证据索引</a>` : ''}
      </div>
      <p data-story-message role="status" aria-live="polite">${busy ? '正在处理，请稍候…' : locked ? '已审核段落受保护，不能编辑或覆盖。' : '所有生成段落都需要人工确认。'}</p>
      <details class="story-details"><summary>版本历史（${state.history.length}）</summary>
        ${state.history.map(h => `<button class="btn btn-ghost" data-history="${h.revision}" ${busy ? 'disabled' : ''}>${esc(h.action)} · ${esc(h.updated_at)}</button>`).join('')}
      </details>
      ${state.input ? `<details class="story-details"><summary>查看输入图片（${state.input.count}）</summary><div style="display:flex;flex-wrap:wrap;gap:12px">${Array.from({length:state.input.count}, (_,i) => `<a href="${endpoint()}/inputs/${i}" target="_blank" rel="noopener"><img class="story-proof" src="${endpoint()}/inputs/${i}" alt="输入 ${i+1}" loading="lazy"><p>输入 ${i+1}</p></a>`).join('')}</div></details>` : ''}
      ${bible ? `<details class="story-details"><summary>Story Bible · ${bible.panels.length} 个分镜 / ${bible.characters.length} 个人物 / ${bible.events.length} 个事件</summary>
        <h3>人物档案</h3>${bible.characters.map(c => `<p><strong>${esc(c.name)} (${esc(c.character_id)})</strong>：${esc(c.description)}<br>来源 ${esc(c.source_panel_ids.join('、'))}</p>`).join('')}
        ${(bible.relationships || []).map(r => `<p>关系：${esc(r.from_id)} → ${esc(r.to_id)} · ${esc(r.relation)}；原文“${esc(r.quote)}”（来源 ${esc(r.source_panel_id)}）</p>`).join('')}
        <h3>事件时间线</h3>${bible.events.map(e => `<article class="story-event"><strong>${esc(e.event_id)}</strong> ${esc(e.summary)}
          <p>动作：${esc(e.actions.join('；'))}</p><p>来源：${esc(e.source_panel_ids.join('、'))} · 重要性 ${e.importance}/5</p>
          ${e.narrative_eligible === false ? `<p class="story-warning">非剧情，不用于写稿：${esc(e.exclusion_reason)}</p>` : ''}
          <p>因果：${esc(e.causes.map(c => `${c.event_id}：${c.evidence}`).join('；') || '无明确依据，未推定因果')}</p>
          <details><summary>查看事件原图、对白与识别疑点</summary>${e.source_panel_ids.map(pid => {
            const p = bible.panels.find(p => p.source_panel_id === pid);
            return `<a href="${evidence(pid)}" target="_blank" rel="noopener"><img class="story-proof" src="${evidence(pid)}" alt="原始分镜 ${esc(pid)}" loading="lazy"></a><p>${esc(p.dialogue.map(d => d.text).join(' / '))}</p><p>${esc(p.uncertainties.join('；'))}</p>`;
          }).join('')}</details></article>`).join('')}
      </details>` : '<p>尚无剧情分析。请先确认分镜，再开始分析。</p>'}
      ${state.plan ? `<details class="story-details"><summary>叙事规划 · ${state.plan.beats.length} 个关键事件</summary>
        <p>开场 ${esc(state.plan.opening_event_id)} → 高潮 ${esc(state.plan.climax_event_id)} → 结尾 ${esc(state.plan.ending_event_id)}</p>
        <p>${esc(state.plan.ending_rule)}；${esc(state.plan.budget_note)}</p>
        ${state.plan.beats.map(b => `<p>${esc(b.event_id)} · ${esc(b.section)} · ${esc(b.reason)}</p>`).join('')}
      </details>` : ''}
      <div class="story-segments">${(script?.segments || []).map(segment => {
        const approved = segment.review_status === 'approved', check = segment.fact_check;
        const primary = bible.events.find(e => e.event_id === segment.event_id).source_panel_ids;
        const extra = segment.source_panel_ids.filter(pid => !primary.includes(pid));
        const key = draftKey(segment.segment_id);
        if (drafts.get(key) === segment.text || approved) drafts.delete(key);
        const draft = !viewRevision && !approved && drafts.has(key) ? drafts.get(key) : segment.text;
        return `<article class="story-segment" data-segment="${esc(segment.segment_id)}">
          <div class="story-proof-column">${primary.map(pid => `<a href="${evidence(pid)}" target="_blank" rel="noopener"><img src="${evidence(pid)}" alt="原始分镜 ${esc(pid)}（点击放大）" loading="lazy"></a>`).join('')}
          ${extra.length ? `<details><summary>因果与背景原图（${extra.length}）</summary>${extra.map(pid => `<a href="${evidence(pid)}" target="_blank" rel="noopener"><img src="${evidence(pid)}" alt="背景来源 ${esc(pid)}" loading="lazy"></a>`).join('')}</details>` : ''}
          <small>点击原图放大 · 来源 ${esc(segment.source_panel_ids.join('、'))}</small></div>
          <div class="story-text-column"><p><strong>${esc(segment.event_id)} · ${esc(segment.section)}</strong></p>
            <p class="story-status">${labels[segment.review_status]} ｜ ${labels[check?.status] || '待核验'}</p>
            <textarea aria-label="${esc(segment.segment_id)} 解说稿" rows="4" ${approved || readOnly ? 'readonly' : ''}>${esc(draft)}</textarea>
            ${draft !== segment.text ? '<p class="story-warning">有未保存修改；以下核验结果对应已保存版本。</p>' : ''}
            <div class="story-toolbar">${[['edit','保存并核验'],['regenerate','单段重新生成'],['check','重新核验'],['approve','审核通过并锁定'],['reject','驳回']].map(([action,label]) => `<button class="btn btn-secondary" data-segment-action="${action}" ${readOnly || (approved && action !== 'check') || (action === 'approve' && (check?.status !== 'passed' || (state.input && state.input.kind !== 'panels'))) ? 'disabled' : ''}>${label}</button>`).join('')}</div>
            <div class="story-check">${check?.error ? `<p class="story-warning">${esc(check.error)}</p>` : ''}
            ${(check?.sentences || []).map(s => `<p><strong>${s.sentence_index + 1}. ${esc(s.text)}</strong><br>${esc(s.verdict === 'supported' ? '有来源依据' : s.verdict === 'issues' ? '存在问题' : '需人工核对')}：${esc(s.evidence || '')}<br>${s.issues.map(i => `${labels[i.type]}：${esc(i.detail)}`).join('<br>')}</p>`).join('')}</div>
          </div></article>`;
      }).join('')}</div>
    </section>`;
    host.querySelectorAll('[data-action]').forEach(button => button.onclick = () => {
      if (button.dataset.action === 'refresh') { viewRevision = null; reload().catch(e => message(e.message)); }
      else run(button.dataset.action).catch(e => message(e.message));
    });
    host.querySelectorAll('[data-history]').forEach(button => button.onclick = () => {
      viewRevision = button.dataset.history; reload().catch(e => message(e.message));
    });
    host.querySelectorAll('[data-segment-action]').forEach(button => button.onclick = () => {
      const card = button.closest('[data-segment]');
      const segment = state.script.segments.find(s => s.segment_id === card.dataset.segment);
      if (button.dataset.segmentAction !== 'edit' && card.querySelector('textarea').value !== segment.text) {
        message('此段有尚未保存的修改，请先“保存并核验”。'); return;
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
    if (['generate','analyze'].includes(action) && (state.script?.segments || []).some(s => drafts.has(draftKey(s.segment_id)))) {
      message('有尚未保存的旁白修改，请先逐段保存。'); return;
    }
    const style = host.querySelector('[data-story-style]').value;
    busy = true; draw();
    try {
      const job = await json(`${endpoint()}/${action}`, {method:'POST',
        headers:{'Content-Type':'application/json'}, body:JSON.stringify({revision:state.revision, style, ...payload})});
      await poll(job.job_id, generation);
    } catch (error) { busy = false; draw(); throw error; }
  }
  async function mount(element, name) {
    generation++; clearTimeout(timer); host = element; project = name; viewRevision = null; busy = false;
    host.innerHTML = '<p data-story-message role="status">正在加载剧情审核…</p>';
    try { await reload(); } catch (error) { message(error.message); }
  }
  return {mount};
})();
