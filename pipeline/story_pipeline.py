"""Text-only chapter workflow; deliberately has no audio/video entry point."""
from copy import deepcopy

from pipeline.fact_checker import FactChecker
from pipeline.narrative_director import NarrativeDirector, PLANNING_CONTRACT_VERSION
from pipeline.script_writer import ScriptWriter
from pipeline.story_analyzer import StoryAnalyzer
from pipeline.story_bible import StoryConflict, StoryError
from pipeline.semantic_grounding import grounded_units
from pipeline.narration_review import review_ready
from pipeline.chapter_memory import generation_context, build_memory, context_current, planning_settings
from pipeline.story_bible import digest
from pipeline.story_model import validated_result, require
from pipeline.story_candidates import VARIANTS, compare_candidates, sync_selected, comparison_pool
from pipeline.event_integration import integrate_chapter, chapter_input
from pipeline.prompt_material import current_fact_sheet
from pipeline.fact_material import source_material_basis
from pipeline.opening_recap import with_opening_recap


class StoryPipeline:
    def __init__(self, store, model):
        self.store, self.model = store, model
        self.checker = FactChecker(store, model)

    def _save(self, state, action, revision):
        sync_selected(state)
        return self.store.save(state,action,revision)

    def _current(self, expected_revision):
        state = self.store.read()
        if state["revision"] != expected_revision:
            raise StoryConflict("内容已变化，请刷新后重试")
        return state

    @staticmethod
    def _unlocked(state):
        if any(s["review_status"] == "approved" for s in (state.get("script") or {}).get("segments", [])):
            raise StoryConflict("已审核文案已锁定；不能重新分析或整稿覆盖，请创建独立草稿项目")

    @staticmethod
    def _review(writer, state, segments):
        error = None
        for _ in range(2):
            try:
                return writer.review_story(state['bible'],state['plan'],segments,feedback=error,
                    facts=current_fact_sheet(state['bible'],state['plan'],state.get('script') or {}))
            except Exception as exc:
                error = str(exc)
            # A malformed or unavailable reviewer must not discard the actual
            # draft or invite a rewrite without specific, evidenced feedback.
        return {'passed':False,'issues':[],'blocking_issues':[], 'error':error,
                'information_review':{'status':'needs_review','reason':'整稿复核响应不可用'},
                'content_review':{'status':'needs_review','reason':'整稿复核响应不可用'}}

    def _remember(self, state, force=False):
        state['chapter_memory'] = build_memory(self.store,self.model,state,force)
        return self._save(state,'chapter_memory',state['revision'])

    def _check(self, bible, segment, fresh=False):
        result = self.checker.check(bible,segment,fresh=fresh)
        if result['status']=='unavailable':
            result = self.checker.check(bible,segment,fresh=True,feedback=result.get('error','响应不可用'))
        return result

    def remember(self, expected_revision):
        with self.store.lock:
            state = self._current(expected_revision)
            if not (state.get('script') or {}).get('segments'):
                raise StoryError('请先生成稿件')
            return self._remember(state,True)

    def analyze(self, panels, expected_revision, progress=None):
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            history = generation_context(self.store)
            history['reference_notes'] = deepcopy(history.get('series_notes',{}))
            bible = StoryAnalyzer(self.store, self.model).analyze(panels, progress, history)
            state.pop('chapter_memory',None)
            state.pop('selected_candidate_id',None)
            state.update(bible=bible, plan=None, script=None, stage="analyzed")
            saved = self._save(state, "analyze", expected_revision)
            # The immutable analysis revision is its own reusable source document.
            saved['analysis_revision'] = saved['revision']
            return self._save(saved,'pin_analysis_version',saved['revision'])

    def _integrate(self, state):
        bible = state['bible']
        # Pin the actual existing analysis even for legacy chapters which never
        # recorded analysis_revision. Subsequent text saves are not new inputs.
        source_revision = (bible.get('event_integration') or {}).get('source_revision') or state.get('analysis_revision') or state['revision']
        if bible.get('event_integration') and digest(chapter_input(bible, source_revision)) != bible['event_integration']['input_hash']:
            source_revision = state['revision']
        integrated = integrate_chapter(self.store, self.model, bible, source_revision)
        if integrated == bible:
            return state
        state.update(bible=integrated, plan=None, script=None, stage='integrated')
        state.pop('selected_candidate_id', None)
        state.pop('chapter_memory', None)
        return self._save(state, 'integrate_chapter_events', state['revision'])

    def integrate(self, expected_revision):
        """First text step, independently reviewable without writing a draft."""
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            require(state.get('bible'), '请先完成剧情分析')
            return self._integrate(state)

    def generate(self, expected_revision, style="faithful", max_events=None, progress=None, candidate_count=2):
        """Generate missing candidates only; no image checking, TTS or export."""
        require(type(candidate_count) is int and candidate_count in (2,3),'候选总数只能为2或3')
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            require(state.get('bible'),'请先完成剧情分析')
            require(context_current(self.store,state['bible']),'前话资料已变化，请重新分析本话后再生成')
            state = self._integrate(state)
            bible = state['bible']
            if state.get('selected_candidate_id'):
                selected = next(c for c in state['candidates'] if c['candidate_id']==state['selected_candidate_id'])
                if selected['analysis_id']!=bible['bible_id']:
                    state.pop('selected_candidate_id',None)
                    state.update(script=None,plan=None)
                    state.pop('chapter_memory',None)
            candidates = state.setdefault('candidates',[])
            # Share the plan and deterministic source packaging, without a digest API.
            settings = planning_settings(self.store)
            plan = validated_result(self.store,self.model,'direct',
                {'contract':PLANNING_CONTRACT_VERSION,'bible':bible,'style':style,'max_events':max_events,
                 'settings':settings},
                lambda:NarrativeDirector(self.model,settings=settings).plan(bible,style,max_events))
            plan = with_opening_recap(bible, plan)
            source_basis = source_material_basis(bible, plan) if plan.get('version',0)>=5 else None
            matching = [c for c in candidates if c['analysis_id']==bible['bible_id'] and c['plan']==plan
                        and c['script'].get('source_material_basis')==source_basis]
            writer = ScriptWriter(self.model,self.store)
            for index in range(candidate_count):
                if index < len(matching):
                    existing = next(c for c in state['candidates'] if c['candidate_id']==matching[index]['candidate_id'])
                    if existing['script']['segments']:
                        continue  # Invalid reviews need explicit review, never blind regeneration.
                    cid = existing['candidate_id']
                else:
                    existing = None
                    cid = f"K{len(candidates)+1:03d}"
                candidate = {'candidate_id':cid,'analysis_id':bible['bible_id'],
                    'source_revision':state['revision'],'variant':VARIANTS[index], 'plan':deepcopy(plan),
                    'script':{'bible_id':bible['bible_id'],'style':style,'segments':[],
                              'source_material_basis':deepcopy(source_basis)},'status':'writing'}
                if existing is None:
                    candidates.append(candidate)
                state = self._save(state,'candidate_started:'+cid,state['revision'])
                candidates = state['candidates'];candidate = next(c for c in candidates if c['candidate_id']==cid)
                try:
                    if source_basis is not None:
                        drafts = writer.write_story(bible,plan,variant={'candidate_id':cid,'direction':VARIANTS[index]})
                    else:
                        drafts = writer.write(bible,plan)['segments']
                    candidate['script']['segments'] = drafts
                    candidate['status'] = 'text_review_pending'
                    state = self._save(state,'candidate_written:'+cid,state['revision'])
                    candidate = next(c for c in state['candidates'] if c['candidate_id']==cid)
                    candidate['script']['narrative_review'] = self._review(writer,
                        {'bible':bible,'plan':plan,'script':candidate['script']},drafts)
                    candidate['status'] = 'text_reviewed' if review_ready(candidate['script']['narrative_review'],bible,plan,drafts) else 'needs_review'
                except Exception as exc:
                    candidate.update(status='needs_review',error=str(exc))
                state = self._save(state,'candidate_reviewed:'+cid,state['revision'])
                candidates = state['candidates']
                if progress:progress(index+1,candidate_count,'已保存候选 '+cid+'；未执行原图核验')
            state['candidate_set'] = [c['candidate_id'] for c in state['candidates']
                if c['analysis_id']==bible['bible_id'] and c['plan']==plan
                and c['script'].get('source_material_basis')==source_basis][:candidate_count]
            state['candidate_comparison'] = compare_candidates(self.model,bible,comparison_pool(state))
            if not state.get('selected_candidate_id'):
                state['stage'] = 'pending_selection'
            return self._save(state,'compare_candidates',state['revision'])

    def compare(self, expected_revision):
        with self.store.lock:
            state = self._current(expected_revision)
            state['candidate_comparison'] = compare_candidates(self.model,state['bible'],comparison_pool(state))
            return self._save(state,'compare_candidates',state['revision'])

    def review_candidate(self, expected_revision, candidate_id):
        with self.store.lock:
            state = self._current(expected_revision)
            candidate = next((c for c in state.get('candidates',[]) if c['candidate_id']==candidate_id),None)
            require(candidate is not None and candidate['script']['segments'],'候选正文不存在')
            require(candidate['analysis_id']==state['bible']['bible_id'],'候选依据已过期')
            script = candidate['script']
            script['narrative_review'] = self._review(ScriptWriter(self.model,self.store),
                {'bible':state['bible'],'plan':candidate['plan'],'script':script},script['segments'])
            candidate['status'] = 'text_reviewed' if review_ready(script['narrative_review'],state['bible'],candidate['plan'],script['segments']) else 'needs_review'
            if state.get('selected_candidate_id')==candidate_id:state['script']=deepcopy(script)
            return self._save(state,'review_candidate:'+candidate_id,state['revision'])

    def select(self, expected_revision, candidate_id):
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            candidate = next((c for c in state.get('candidates',[]) if c['candidate_id']==candidate_id),None)
            require(candidate is not None,'候选不存在')
            require(candidate['analysis_id']==state['bible']['bible_id'],'候选分析依据已变化，请重新生成相关候选')
            require(candidate['script']['segments'],'候选正文尚未生成')
            require(context_current(self.store,state['bible']),'前话资料已变化，请重新分析')
            state.update(selected_candidate_id=candidate_id,plan=deepcopy(candidate['plan']),
                         script=deepcopy(candidate['script']),stage='selected_pending_check')
            state.pop('chapter_memory',None)
            return self._save(state,'select_candidate:'+candidate_id,state['revision'])

    def verify_selected(self, expected_revision):
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            require(state.get('selected_candidate_id'),'请先选定候选；选定不等于核验通过')
            require(context_current(self.store,state['bible']),'前话资料已变化，请重新分析')
            writer = ScriptWriter(self.model,self.store)
            for panel in state['bible']['panels']:
                self.store.evidence_path(panel)
            prior = []
            for beat,segment in zip(state['plan']['beats'],state['script']['segments']):
                if not grounded_units(state['bible'],segment) or (segment.get('fact_check') or {}).get('status')!='passed':
                    segment['fact_check'] = self._check(state['bible'],segment)
                    for _ in range(2):
                        if segment['fact_check']['status'] not in {'issues','uncertain'}:break
                        fixed = writer.write_segment(state['bible'],state['plan'],beat,prior[-1]['text'] if prior else '',
                            feedback=segment['fact_check'],fresh=True,original_text=segment['text'],
                            facts=current_fact_sheet(state['bible'],state['plan'],state['script']))
                        if fixed['text']==segment['text']:break
                        segment = fixed
                        segment['fact_check'] = self._check(state['bible'],segment)
                prior.append(segment)
                state['script']['segments'][len(prior)-1] = segment
                state = self._save(state,'verify_selected_segment',state['revision'])
            # Review transitions after fact repairs, then retain the existing targeted repair loop.
            return self.revise(state['revision'])

    def generate_verified(self, expected_revision, style="faithful", max_events=None, progress=None):
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            if not state["bible"]:
                raise StoryError("请先完成剧情分析")
            if not context_current(self.store,state['bible']):
                raise StoryConflict('前话资料已变化，请重新分析本话后再生成，旧版本仍保留')
            state = self._integrate(state)
            plan = NarrativeDirector(self.model,settings=planning_settings(self.store)).plan(state["bible"], style, max_events)
            plan = with_opening_recap(state['bible'], plan)
            state.update(plan=plan, script={"bible_id": state["bible"]["bible_id"],
                                           "style": style, "segments": []}, stage="writing")
            state = self._save(state, "plan", state["revision"])
            writer = ScriptWriter(self.model,self.store)
            drafts = None
            if plan.get('version',0) >= 5:
                state['script']['source_material_basis'] = source_material_basis(state['bible'], plan)
                feedback = None
                for attempt in range(3):
                    try:
                        drafts = writer.revise_story(state['bible'],plan,drafts,feedback,
                            facts=current_fact_sheet(state['bible'],plan,state['script'])) if drafts is not None else writer.write_story(
                            state['bible'],plan,feedback,fresh=attempt > 0,
                            facts=current_fact_sheet(state['bible'],plan,state['script']))
                        feedback = self._review(writer,state,drafts)
                    except StoryError as exc:
                        if attempt == 2:
                            raise
                        feedback = {'passed':False,'issues':[{'detail':str(exc)}]}
                        continue
                    if feedback['passed'] and not feedback['issues']:
                        break
                    if not any(isinstance(i,dict) and i.get('event_id') in {s['event_id'] for s in drafts}
                               for i in feedback['issues']):
                        # An incomplete review is pending work, not permission to
                        # recreate every correct paragraph from the ledger.
                        break
                else:
                    if drafts:
                        state['script']['segments'] = drafts
                        state['script']['narrative_review'] = feedback
                        state['stage'] = 'review'
                        self._save(state,'narrative_review_failed',state['revision'])
                    raise StoryError('整稿连贯性尚未通过：'+str(feedback))
                state['script']['narrative_review'] = feedback
            for index, beat in enumerate(plan["beats"]):
                prior = state["script"]["segments"]
                segment = drafts[index] if drafts is not None else writer.write_segment(
                    state["bible"], plan, beat, prior[-1]["text"] if prior else "")
                segment["fact_check"] = self._check(state["bible"], segment)
                for _ in range(2):
                    if segment["fact_check"]["status"] not in {"issues", "uncertain"}:
                        break
                    segment = writer.write_segment(state["bible"], plan, beat, prior[-1]["text"] if prior else "",
                                                   feedback=segment["fact_check"], fresh=True,original_text=segment['text'],
                                                   facts=current_fact_sheet(state['bible'],plan,state['script']))
                    segment["fact_check"] = self._check(state["bible"], segment)
                prior.append(segment)
                # Commit each paragraph so interruptions don't erase successful work.
                state = self._save(state, "write_segment", state["revision"])
                if progress:
                    progress(index + 1, len(plan["beats"]), f"已写稿并核验 {index + 1} 段")
            if drafts is not None:
                state['script']['narrative_review'] = self._review(writer,state,state['script']['segments'])
                # Fact repairs can introduce cross-paragraph repetition. Review
                # the actual final draft and recheck only text that changed.
                for _ in range(2):
                    review = state['script']['narrative_review']
                    if review['passed'] and not review['issues']:
                        break
                    if not any(isinstance(i,dict) and i.get('event_id') in {s['event_id'] for s in state['script']['segments']}
                               for i in review['issues']):
                        break
                    repaired = writer.revise_story(state['bible'],plan,state['script']['segments'],review,
                        facts=current_fact_sheet(state['bible'],plan,state['script']))
                    for segment in repaired:
                        if segment.get('fact_check') is None:
                            segment['fact_check'] = self._check(state['bible'],segment)
                    state['script']['segments'] = repaired
                    state['script']['narrative_review'] = self._review(writer,state,repaired)
                    state = self._save(state,'targeted_story_revision',state['revision'])
            state["stage"] = "review"
            state = self._save(state, "ready_for_review", state["revision"])
            return self._remember(state)

    def segment_action(self, segment_id, action, expected_revision, narration=None):
        with self.store.lock:
            state = self._current(expected_revision)
            script = state.get("script") or {}
            segment = next((s for s in script.get("segments", []) if s["segment_id"] == segment_id), None)
            if segment is None:
                raise StoryError("旁白段落不存在")
            if segment["review_status"] == "approved" and action != "check":
                raise StoryConflict("此段已审核并锁定，不能编辑、驳回或重新生成")
            if action == "edit":
                if not isinstance(narration, str) or not narration.strip() or len(narration) > 1000:
                    raise StoryError("请输入1到1000字的旁白")
                if narration.strip() != segment['text']:
                    segment.update(text=narration.strip(), review_status="pending", fact_check=None)
                    segment["fact_check"] = self._check(state["bible"], segment)
            elif action == "regenerate":
                beat = next(b for b in state["plan"]["beats"] if b["event_id"] == segment["event_id"])
                idx = script["segments"].index(segment)
                candidate = ScriptWriter(self.model).write_segment(
                    state["bible"], state["plan"], beat,
                    script["segments"][idx - 1]["text"] if idx else "", fresh=True,
                    feedback=segment.get("fact_check"),original_text=segment['text'],
                    facts=current_fact_sheet(state['bible'],state['plan'],script))
                if candidate['text'] != segment['text']:
                    candidate["fact_check"] = self._check(state["bible"], candidate)
                    script["segments"][idx] = candidate
            elif action == "check":
                segment["fact_check"] = self._check(state["bible"], segment, fresh=True)
            elif action == "approve":
                if not context_current(self.store,state['bible']):
                    raise StoryConflict('前话资料已变化，请重新分析核对后再审核本话')
                if not segment.get("fact_check") or segment["fact_check"]["status"] != "passed":
                    raise StoryConflict("事实核验尚未通过，请先修正文案或补充剧情证据")
                if state['plan'].get('version',0)>=5:
                    review = script.get('narrative_review') or {}
                    if not review.get('passed') or review.get('issues') or (review.get('information_review') or {}).get('status')!='reviewed':
                        raise StoryConflict('整稿新增信息与重复检查尚未通过')
                    if not grounded_units(state['bible'], segment):
                        raise StoryConflict('当前句子的事实与展示来源尚未关联，请重新核验')
                    if not review_ready(review,state['bible'],state['plan'],script['segments']):
                        raise StoryConflict('当前整稿的叙事内容复核尚未完成或已失效，请重新检查当前稿')
                segment["review_status"] = "approved"
            elif action == "reject":
                segment["review_status"] = "rejected"
            else:
                raise StoryError("不支持的审核操作")
            if state.get('plan',{}).get('version',0) >= 5 and action in {'edit','regenerate','check'}:
                script['narrative_review'] = self._review(ScriptWriter(self.model),state,script['segments'])
            complete = len(script["segments"]) == len(state["plan"]["beats"])
            state["stage"] = "approved" if complete and all(
                s["review_status"] == "approved" and (s.get("fact_check") or {}).get("status") == "passed"
                for s in script["segments"]) else "review"
            if state.get('plan',{}).get('version',0) >= 5:
                review = script.get('narrative_review') or {}
                if not review_ready(review,state['bible'],state['plan'],script['segments']):
                    state['stage'] = 'review'
            state = self._save(state, f"{action}:{segment_id}", expected_revision)
            return self._remember(state) if action in {'edit','regenerate'} else state

    def revise(self, expected_revision):
        """Review the current draft; preserve good paragraphs and their checks."""
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            if not (state.get('script') or {}).get('segments'):
                raise StoryError('请先生成稿件')
            writer = ScriptWriter(self.model,self.store)
            if not context_current(self.store,state['bible']):
                raise StoryConflict('前话资料已变化，请重新分析本话后再返修，旧版本仍保留')
            for segment in state['script']['segments']:
                if ((segment.get('fact_check') or {}).get('grounding') or {}).get('status')=='mapped' and not grounded_units(state['bible'],segment):
                    segment['fact_check'] = self._check(state['bible'],segment)
            for attempt in range(3):
                segments = state['script']['segments']
                review = self._review(writer,state,segments)
                state['script']['narrative_review'] = review
                state['stage'] = 'review'
                state = self._save(state,'review_current_draft',state['revision'])
                if review['passed'] and not review['issues'] or attempt==2:
                    break
                if not any(isinstance(i,dict) and i.get('event_id') in {s['event_id'] for s in segments}
                           for i in review['issues']):
                    break
                repaired = writer.revise_story(state['bible'],state['plan'],segments,review,
                    facts=current_fact_sheet(state['bible'],state['plan'],state['script']))
                for segment in repaired:
                    if segment.get('fact_check') is None:
                        segment['fact_check'] = self._check(state['bible'],segment)
                state['script']['segments'] = repaired
                state = self._save(state,'revise_current_draft',state['revision'])
            return self._remember(state)

    def export(self, state=None):
        state = deepcopy(state or self.store.read())
        if not state.get("script"):
            raise StoryError("尚未生成解说稿")
        return {"revision": state["revision"], "stage": state["stage"],
                "bible_id": state["bible"]["bible_id"], **state["script"],
                'chapter_memory':state.get('chapter_memory')}
