"""Text-only chapter workflow; deliberately has no audio/video entry point."""
from copy import deepcopy

from pipeline.fact_checker import FactChecker
from pipeline.narrative_director import NarrativeDirector, PLANNING_CONTRACT_VERSION
from pipeline.script_writer import ScriptWriter
from pipeline.story_analyzer import StoryAnalyzer
from pipeline.story_bible import StoryConflict, StoryError
from pipeline.semantic_grounding import grounded_units
from pipeline.narration_review import polish_ready, ENGAGEMENT_POLISH_VERSION
from pipeline.narration_polish import polish_options as normalize_polish_options, saved_polish_options
from pipeline.chapter_memory import generation_context, build_memory, context_current, planning_settings
from pipeline.story_bible import digest
from pipeline.story_model import validated_result, require
from pipeline.event_integration import integrate_chapter, chapter_input
from pipeline.prompt_material import current_fact_sheet
from pipeline.fact_material import source_material_basis
from pipeline.opening_recap import with_opening_recap
from pipeline.human_review import approve_segment, segment_approved


class StoryPipeline:
    def __init__(self, store, model):
        self.store, self.model = store, model
        self.checker = FactChecker(store, model)

    def _save(self, state, action, revision):
        # Historical revisions retain the former draft pool unchanged. New saves
        # contain only the current script; edits never write back to old drafts.
        for key in ('candidates', 'candidate_set', 'candidate_comparison', 'selected_candidate_id'):
            state.pop(key, None)
        script = state.get('script') or {}
        record = script.get('narrative_review') or {}
        operation = record.get('engagement_polish') or {}
        if operation.get('status')=='completed' and not polish_ready(record,state.get('bible'),state.get('plan'),script.get('segments',[]),script.get('polish_options')):
            operation['status']='stale'
            record['passed']=False
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

    def _polish(self, state, fresh=False):
        """Apply a validated edit and save before/after revisions."""
        script = state['script']
        try:
            result = ScriptWriter(self.model,self.store).polish_story(
                state['bible'],state['plan'],script['segments'],fresh=fresh,options=script.get('polish_options'))
        except Exception as exc:
            script['narrative_review'] = {'passed':False,'issues':[], 'error':str(exc),
                'engagement_polish':{'version':ENGAGEMENT_POLISH_VERSION,'status':'unavailable'}}
            state['stage'] = 'review'
            return self._save(state,'polish_unavailable',state['revision'])
        if result['completion']['engagement_polish']['changed_event_ids']:
            state.pop('chapter_memory',None)
        script.update(segments=result['segments'],narrative_review=result['completion'])
        state['stage'] = 'review'
        return self._save(state,'polish_current_draft',state['revision'])

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
        state.pop('chapter_memory', None)
        return self._save(state, 'integrate_chapter_events', state['revision'])

    def integrate(self, expected_revision):
        """First text step, independently reviewable without writing a draft."""
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            require(state.get('bible'), '请先完成剧情分析')
            return self._integrate(state)

    def generate(self, expected_revision, style="faithful", max_events=None, progress=None, polish_options=None):
        """Write one current draft and directly polish its interest; no image/TTS calls."""
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            options = normalize_polish_options(polish_options if polish_options is not None else
                                               saved_polish_options(state.get('script')))
            require(state.get('bible'), '请先完成剧情分析')
            if not context_current(self.store,state['bible']):
                raise StoryConflict('前话资料已变化，请重新分析本话后再生成')
            state = self._integrate(state)
            bible = state['bible']
            settings = planning_settings(self.store)
            plan = validated_result(self.store,self.model,'direct',
                {'contract':PLANNING_CONTRACT_VERSION,'bible':bible,'style':style,'max_events':max_events,
                 'settings':settings},
                lambda:NarrativeDirector(self.model,settings=settings).plan(bible,style,max_events))
            plan = with_opening_recap(bible, plan)
            source_basis = source_material_basis(bible, plan) if plan.get('version',0)>=5 else None
            writer_identity = self.model.cache_identity('write_story') if hasattr(self.model,'cache_identity') else {
                'class':type(self.model).__name__}
            basis = digest({'single_script':1,'bible':bible,'plan':plan,'source_material_basis':source_basis,
                            'writer':writer_identity})
            script = state.get('script') or {}
            if (script.get('generation_basis') == basis and state.get('plan') == plan
                    and len(script.get('segments',[])) == len(plan['beats'])):
                options_changed = options != normalize_polish_options(script.get('polish_options'))
                script['polish_options'] = options
                operation = (script.get('narrative_review') or {}).get('engagement_polish') or {}
                already_edited = (operation.get('version')==ENGAGEMENT_POLISH_VERSION
                                  and operation.get('status')=='stale')
                polish_identity = self.model.cache_identity('script_polish') if hasattr(self.model,'cache_identity') else None
                if options_changed or (not already_edited and not polish_ready(
                        script.get('narrative_review'),bible,plan,script['segments'],options,polish_identity)):
                    return self._polish(state)
                return state  # Repeated requests never rewrite saved or edited text.
            state.pop('chapter_memory',None)
            state.update(plan=plan, script={'bible_id':bible['bible_id'],'style':style,
                         'segments':[], 'source_material_basis':source_basis,'generation_basis':basis,
                         'polish_options':options}, stage='writing')
            state = self._save(state,'script_started',state['revision'])
            writer = ScriptWriter(self.model,self.store)
            try:
                if source_basis is not None:
                    drafts = writer.write_story(bible,plan)
                else:
                    drafts = writer.write(bible,plan)['segments']
            except Exception as exc:
                state.update(stage='writing_failed', error=str(exc))
                self._save(state,'script_failed',state['revision'])
                raise
            state.pop('error',None)
            state['script']['segments'] = drafts
            state['stage'] = 'text_review_pending'
            state = self._save(state,'script_written',state['revision'])
            state = self._polish(state)
            if progress:
                progress(1,1,'口播稿已保存，待人工审核锁定；通过后进入分镜、配音与视频制作')
            return state

    def verify(self, expected_revision):
        """Strictly check the current script without selecting another draft."""
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            require((state.get('script') or {}).get('segments'), '请先生成稿件')
            require(len(state['script']['segments']) == len(state['plan']['beats']), '稿件节点尚未完整')
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
                        if fixed['text']==segment['text']:
                            segment['fact_check'] = self._check(state['bible'],segment,fresh=True)
                            break
                        segment = fixed
                        segment['fact_check'] = self._check(state['bible'],segment)
                prior.append(segment)
                state['script']['segments'][len(prior)-1] = segment
                state = self._save(state,'verify_current_segment',state['revision'])
            state['stage'] = 'review'
            state = self._save(state,'verified_current_draft',state['revision'])
            return self._remember(state)

    def generate_verified(self, expected_revision, style="faithful", max_events=None, progress=None, polish_options=None):
        """Single draft -> one polish call -> independent image verification."""
        with self.store.lock:
            state = self.generate(expected_revision,style,max_events,progress,polish_options)
            return self.verify(state['revision'])

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
                    segment.pop('human_review', None)
                    state.pop('chapter_memory', None)
            elif action == "regenerate":
                beat = next(b for b in state["plan"]["beats"] if b["event_id"] == segment["event_id"])
                idx = script["segments"].index(segment)
                candidate = ScriptWriter(self.model).write_segment(
                    state["bible"], state["plan"], beat,
                    script["segments"][idx - 1]["text"] if idx else "", fresh=True,
                    feedback=segment.get("fact_check"),original_text=segment['text'],
                    facts=current_fact_sheet(state['bible'],state['plan'],script))
                if candidate['text'] != segment['text']:
                    candidate.pop('human_review', None)
                    script["segments"][idx] = candidate
                    state.pop('chapter_memory', None)
            elif action == "check":
                segment["fact_check"] = self._check(state["bible"], segment, fresh=True)
            elif action == "approve":
                if not context_current(self.store,state['bible']):
                    raise StoryConflict('前话资料已变化，请重新分析核对后再审核本话')
                approve_segment(state['bible'], segment)
            elif action == "reject":
                segment["review_status"] = "rejected"
            else:
                raise StoryError("不支持的审核操作")
            complete = len(script["segments"]) == len(state["plan"]["beats"])
            state["stage"] = "approved" if complete and all(
                segment_approved(state['bible'], state['plan'], s)
                for s in script["segments"]) else "review"
            state = self._save(state, f"{action}:{segment_id}", expected_revision)
            return state

    def revise(self, expected_revision, polish_options=None):
        """Explicit fourth-stage request, optionally preceded by brief analysis."""
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            require((state.get('script') or {}).get('segments'), '请先生成稿件')
            require(context_current(self.store,state['bible']), '前话资料已变化，请重新分析本话后再润色')
            state['script']['polish_options'] = normalize_polish_options(
                polish_options if polish_options is not None else saved_polish_options(state['script']))
            return self._polish(state,fresh=True)

    def export(self, state=None):
        state = deepcopy(state or self.store.read())
        if not state.get("script"):
            raise StoryError("尚未生成解说稿")
        return {"revision": state["revision"], "stage": state["stage"],
                "bible_id": state["bible"]["bible_id"], **state["script"],
                'chapter_memory':state.get('chapter_memory')}
