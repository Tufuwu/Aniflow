"""Text-only chapter workflow; deliberately has no audio/video entry point."""
from copy import deepcopy

from pipeline.fact_checker import FactChecker
from pipeline.narrative_director import NarrativeDirector
from pipeline.script_writer import ScriptWriter
from pipeline.story_analyzer import StoryAnalyzer
from pipeline.story_bible import StoryConflict, StoryError


class StoryPipeline:
    def __init__(self, store, model):
        self.store, self.model = store, model
        self.checker = FactChecker(store, model)

    def _current(self, expected_revision):
        state = self.store.read()
        if state["revision"] != expected_revision:
            raise StoryConflict("内容已变化，请刷新后重试")
        return state

    @staticmethod
    def _unlocked(state):
        if any(s["review_status"] == "approved" for s in (state.get("script") or {}).get("segments", [])):
            raise StoryConflict("已审核文案已锁定；不能重新分析或整稿覆盖，请创建独立草稿项目")

    def analyze(self, panels, expected_revision, progress=None):
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            bible = StoryAnalyzer(self.store, self.model).analyze(panels, progress)
            state.update(bible=bible, plan=None, script=None, stage="analyzed")
            return self.store.save(state, "analyze", expected_revision)

    def generate(self, expected_revision, style="faithful", max_events=12, progress=None):
        with self.store.lock:
            state = self._current(expected_revision)
            self._unlocked(state)
            if not state["bible"]:
                raise StoryError("请先完成剧情分析")
            plan = NarrativeDirector(self.model).plan(state["bible"], style, max_events)
            state.update(plan=plan, script={"bible_id": state["bible"]["bible_id"],
                                           "style": style, "segments": []}, stage="writing")
            state = self.store.save(state, "plan", state["revision"])
            writer = ScriptWriter(self.model)
            for index, beat in enumerate(plan["beats"]):
                prior = state["script"]["segments"]
                segment = writer.write_segment(state["bible"], plan, beat, prior[-1]["text"] if prior else "")
                segment["fact_check"] = self.checker.check(state["bible"], segment)
                for _ in range(2):
                    if segment["fact_check"]["status"] != "issues":
                        break
                    segment = writer.write_segment(state["bible"], plan, beat, prior[-1]["text"] if prior else "",
                                                   feedback=segment["fact_check"])
                    segment["fact_check"] = self.checker.check(state["bible"], segment)
                prior.append(segment)
                # Commit each paragraph so interruptions don't erase successful work.
                state = self.store.save(state, "write_segment", state["revision"])
                if progress:
                    progress(index + 1, len(plan["beats"]), f"已写稿并核验 {index + 1} 段")
            state["stage"] = "review"
            return self.store.save(state, "ready_for_review", state["revision"])

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
                segment.update(text=narration.strip(), review_status="pending", fact_check=None)
                segment["fact_check"] = self.checker.check(state["bible"], segment)
            elif action == "regenerate":
                beat = next(b for b in state["plan"]["beats"] if b["event_id"] == segment["event_id"])
                idx = script["segments"].index(segment)
                candidate = ScriptWriter(self.model).write_segment(
                    state["bible"], state["plan"], beat,
                    script["segments"][idx - 1]["text"] if idx else "", fresh=True,
                    feedback=segment.get("fact_check"))
                candidate["fact_check"] = self.checker.check(state["bible"], candidate)
                script["segments"][idx] = candidate
            elif action == "check":
                segment["fact_check"] = self.checker.check(state["bible"], segment, fresh=True)
            elif action == "approve":
                if not segment.get("fact_check") or segment["fact_check"]["status"] != "passed":
                    raise StoryConflict("事实核验尚未通过，请先修正文案或补充剧情证据")
                segment["review_status"] = "approved"
            elif action == "reject":
                segment["review_status"] = "rejected"
            else:
                raise StoryError("不支持的审核操作")
            complete = len(script["segments"]) == len(state["plan"]["beats"])
            state["stage"] = "approved" if complete and all(
                s["review_status"] == "approved" and (s.get("fact_check") or {}).get("status") == "passed"
                for s in script["segments"]) else "review"
            return self.store.save(state, f"{action}:{segment_id}", expected_revision)

    def export(self, state=None):
        state = deepcopy(state or self.store.read())
        if not state.get("script"):
            raise StoryError("尚未生成解说稿")
        return {"revision": state["revision"], "stage": state["stage"],
                "bible_id": state["bible"]["bible_id"], **state["script"]}
