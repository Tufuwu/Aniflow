"""Select important beats without dropping their causal prerequisites."""
from pipeline.story_model import array, prompt_json, require
from pipeline.story_bible import StoryError

STYLES = {"faithful": "忠实叙事", "payoff": "爽点解说", "suspense": "高悬念"}


class NarrativeDirector:
    def __init__(self, model=None):
        self.model = model

    def plan(self, bible, style="faithful", max_events=12):
        require(style in STYLES, "不支持的解说风格")
        require(type(max_events) is int and 2 <= max_events <= 60, "关键事件数量应在2到60之间")
        events = [e for e in bible["events"] if e.get("narrative_eligible", True)]
        require(bool(events), "没有可用于叙事的剧情事件")
        if self.model:
            return self._model_plan(bible, events, style, max_events)
        by_id = {e["event_id"]: e for e in events}
        chosen = {events[0]["event_id"], events[-1]["event_id"]}
        # Keep chapter opening/ending, a conflict and a peak even on short budgets.
        for role in ("conflict", "climax"):
            candidates = [e for e in events if e["role"] == role]
            if candidates:
                chosen.add(max(candidates, key=lambda e: e["importance"])["event_id"])
        for e in sorted(events, key=lambda e: (-e["importance"], e["order"])):
            if len(chosen) >= max_events:
                break
            chosen.add(e["event_id"])

        def dependencies(eid):
            for cause in by_id[eid]["causes"]:
                if cause["event_id"] not in chosen:
                    chosen.add(cause["event_id"])
                    dependencies(cause["event_id"])

        for eid in list(chosen):
            dependencies(eid)
        selected = [e for e in events if e["event_id"] in chosen]
        peak = max(selected, key=lambda e: (e["importance"], e["role"] == "climax"))
        beats = []
        for i, e in enumerate(selected):
            role = "opening" if i == 0 else "ending" if i == len(selected) - 1 else (
                "climax" if e == peak else "conflict" if e["role"] == "conflict" else "development")
            beats.append({"event_id": e["event_id"], "source_panel_ids": e["source_panel_ids"],
                          "section": role, "reason": f"剧情重要性 {e['importance']}/5；保留因果前提",
                          "target_chars": 75 if role == "climax" else 50})
        return {"bible_id": bible["bible_id"], "style": style, "beats": beats,
                "opening_event_id": beats[0]["event_id"], "climax_event_id": peak["event_id"],
                "ending_event_id": beats[-1]["event_id"],
                "omitted_event_ids": [e["event_id"] for e in events if e["event_id"] not in chosen],
                "budget_note": "因果前提及首尾冲突优先，必要时允许超出事件预算",
                "ending_rule": "只在本话已出现的未知处留悬念，不预告后续剧情"}

    def _model_plan(self, bible, events, style, max_events):
        prompt = f"""Narrative Director v4。为抖音中文解说选择关键剧情事件，风格 {STYLES[style]}。
事件按原作顺序：{prompt_json([{k:e[k] for k in ('event_id','summary','importance','role','causes')} for e in events])}
选最多 {max_events} 个主事件，通常6到10个足够。不要把同一冲突的连续表情/碎片特写逐一复述。
开场直接进入具体异常或冲突；保留必要设定、主冲突、高潮、真正的本话结尾。
不要只选前半章，必须覆盖本话后半章的重要变化。严格按原作时间顺序排列，不得杜撰事件。
如果发生答题、任务或选择，必须选出必要的题目/规则及代价，不能直接跳到“选B”让人不知所云。
优先选人物关系变化与关键行动，不为单纯表情、颜色或特效单列事件。
里程碑必须覆盖主冲突、高潮、明确的规则、人物目标或失败代价。结尾选择最后一次剧情推进/新任务/悬念，
不要用片尾的表情或拟声字代替真正结尾。若没有规则或目标，对应字段用 null。
因果前提由程序附入主事件上下文，不必为每个前提单独写一段。
返回 {{"event_ids":["E00001"],"milestones":{{"conflict":"事件ID","climax":"事件ID",
"rules":null,"goal_or_stakes":null,"ending":"事件ID"}},"rationale":"选择这些事件的节奏考虑"}}。"""
        by_id = {e["event_id"]:e for e in events}
        retry_prompt=prompt
        for attempt in range(3):
            try:
                raw = self.model.ask("direct", retry_prompt)
                ids = array(raw.get("event_ids"), "event_ids")
                require(all(isinstance(eid, str) and eid in by_id for eid in ids), "导演引用未知或非剧情事件")
                require(bool(ids) and len(ids) <= max_events and len(set(ids)) == len(ids), "导演选择的事件数量无效")
                milestones=raw.get('milestones')
                require(isinstance(milestones,dict),'缺少剧情里程碑')
                for field in ('conflict','climax','rules','goal_or_stakes','ending'):
                    require(field in milestones and (milestones[field] in by_id if isinstance(milestones[field],str) else
                            milestones[field] is None and field in {'rules','goal_or_stakes'}),'剧情里程碑无效')
                break
            except StoryError as exc:
                if attempt==2:
                    raise
                retry_prompt=prompt+f"\n上一版未通过校验：{exc}。event_ids 最多只能有 {max_events} 项，请压缩后重新返回完整 JSON。"
        chosen=set(ids)|{eid for eid in milestones.values() if eid is not None}
        ending_order=by_id[milestones['ending']]['order']
        selected = sorted((by_id[eid] for eid in chosen if by_id[eid]['order'] <= ending_order), key=lambda e:e["order"])
        all_events = {e["event_id"]:e for e in bible["events"]}
        def prerequisites(event, seen):
            for cause in event["causes"]:
                eid = cause["event_id"]
                if eid not in seen:
                    seen.add(eid)
                    prerequisites(all_events[eid], seen)
        peak = by_id[milestones['climax']]
        beats=[]
        for i,event in enumerate(selected):
            context=set()
            prerequisites(event,context)
            sources=set(event["source_panel_ids"])
            for eid in context:
                sources.update(all_events[eid]["source_panel_ids"])
            section="opening" if i==0 else "ending" if i==len(selected)-1 else (
                "climax" if event == peak else "conflict" if event["role"]=="conflict" else "development")
            beats.append({"event_id":event["event_id"],"context_event_ids":sorted(context,key=lambda eid:all_events[eid]["order"]),
                          "source_panel_ids":sorted(sources,key=lambda pid:next(p['order'] for p in bible['panels'] if p['source_panel_id']==pid)),
                          "section":section,"target_chars":60,"reason":raw.get("rationale", "剧情关键节点")})
        chosen={e["event_id"] for e in selected}
        return {"bible_id":bible["bible_id"],"style":style,"beats":beats,
                "opening_event_id":beats[0]["event_id"],"climax_event_id":peak["event_id"],"ending_event_id":beats[-1]["event_id"],
                "omitted_event_ids":[e["event_id"] for e in bible["events"] if e["event_id"] not in chosen],
                "milestones":milestones,
                "budget_note":"因果前提并入主事件；关键规则和目标优先，必要时可超出事件预算",
                "ending_rule":"只在本话已出现的未知处留悬念，不预告后续剧情"}
