"""Chapter-level story planning: motivation, change and payoff determine length."""
from pipeline.story_model import array, prompt_json, require
from pipeline.story_bible import StoryError
from pipeline.story_context import context_ids, source_ids
from pipeline.character_identity import identity_context
from pipeline.chapter_memory import prompt_context
from pipeline.prompt_material import character_index

STYLES = {"faithful": "忠实叙事", "payoff": "爽点解说", "suspense": "高悬念"}


class NarrativeDirector:
    def __init__(self, model=None):
        self.model = model

    def plan(self, bible, style="faithful", max_events=None):
        require(style in STYLES, "不支持的解说风格")
        require(max_events is None or type(max_events) is int and 2 <= max_events <= 60,
                "兼容参数 max_events 应在2到60之间")
        events = [e for e in bible["events"] if e.get("narrative_eligible", True)]
        require(bool(events), "没有可用于叙事的剧情事件")
        if self.model:
            return self._model_plan(bible, events, style)
        selected = [e for e in events if e["importance"] >= 3 or e["role"] in {"conflict", "climax", "resolution"}]
        chosen = {events[0]["event_id"], events[-1]["event_id"], *(e["event_id"] for e in selected)}
        for eid in list(chosen):
            chosen.update(context_ids(bible, [eid]))
        selected = [e for e in events if e["event_id"] in chosen]
        peak = max(selected, key=lambda e: (e["importance"], e["role"] == "climax"))
        beats = [{"event_id": e["event_id"], "source_panel_ids": e["source_panel_ids"],
                  "section": "opening" if i == 0 else "ending" if i == len(selected)-1 else
                  "climax" if e == peak else "conflict" if e["role"] == "conflict" else "development",
                  "reason": "保留行动、转折与因果；没有目标时长"} for i,e in enumerate(selected)]
        return {"bible_id": bible["bible_id"], "style": style, "beats": beats,
                "opening_event_id": beats[0]["event_id"], "climax_event_id": peak["event_id"],
                "ending_event_id": beats[-1]["event_id"],
                "omitted_event_ids": [e["event_id"] for e in events if e["event_id"] not in chosen],
                "budget_note": "故事推进决定篇幅；不按时长、事件数或字数压缩",
                "ending_rule": "用本话已发生的下一步目标或未解决风险收束"}

    def _model_plan(self, bible, events, style):
        facts = [{k:e.get(k, []) for k in ("event_id", "summary", "character_ids", "actions", "dialogue",
                                           "importance", "role", "causes", "source_panel_ids",
                                           "kind", "time", "facets", "status", "pending_causes", "internal_causes",
                                           "reveal_panel_id", "reveal_order")} for e in events]
        prompt = f"""Narrative Director v5。为连续观看组织整话故事，风格 {STYLES[style]}。
没有目标时长，没有固定段数、事件上限或字数配额。只删没有新信息的重复，不删动机、转折和结果。
整话证据事件（按原作顺序）：{prompt_json(facts)}
生成前背景、主要人物资料及上一话大体剧情：{prompt_json(prompt_context(bible))}
历史用于理解当前选择，不机械重讲；用户参考资料须原图核对。historical 事件不能当作本话新剧情。
人物档案及关系：{prompt_json({'characters':character_index(bible.get('characters',[])),'relationships':bible.get('relationships',[])})}
逐事件身份可用位置：{prompt_json(identity_context(bible,[{'event_id':e['event_id']} for e in events]))}
先理解整合事件中的人物目标、阻碍、改变决定的具体消息/条件、行动和兑现，再选择讲述重点、组织 beats 与承接。
整合事件是与后续事实整理共用的本话基础；kind/time 区分当前动作、设定、回忆与历史说明，不把它们全当当前行动。
facets/status/pending_causes 保留必要设定、未知及待补历史，不能为连贯升级；不另压成背景或人物摘要附送。
开场让观众理解具体处境、当前问题和目标，保留环境压力；不设必须几秒进入冲突的规则。
首章或历史资料不可用时仍沿用当前流程，从已有事件与临时人物建立理解，不要求补齐姓名/性别/关系。
必要背景应说明环境、基本异常、危险限制和当前问题，避免设定堆积不等于省掉这些信息。
从背景中寻找具体后果、鲜明反差、危险、代价或待解决问题，解释怎样影响人物处境与选择。
开头建立什么值得继续关注的具体问题，后续哪些行动承接它，须写进 viewer_question/payoff/reason。
可用有吸引力的世界异常/背景事实开场，也可随场景行动交代，不强制先冲突后背景或固定开场结构。
设定区分现在理解选择所需、可在稍后需要时解释、只供核验无需朗读的背景；已启动的任务不被无关历史打断。
背景可以较长，但调整位置须保持原作明确的时间、因果及叙事信息揭示顺序，不得提前泄露揭晓。
严格保留行动因果顺序。不能因一个事件 importance 低，就漏掉人物从拒绝变为同意的原因。
检查每次决定变化：之前态度是什么、什么原文/动作触发改变、之后做了什么，必须在同一或相邻节点讲清。
不要逐格复述表情、衣服、颜色或对白框。安静场景有作用时保留。结尾兑现本话目标，再接已有的下一步目标。
救援后的哭喊、安慰及情绪停顿有叙事作用，不能被办公用品、外貌和站位动作清单淹没。
若有选择/任务，保留规则与代价。人物不知姓名不影响用有证据的稳定称呼；未知细节不单列悬念。
姓名/关系/身份的内部确认不能提前用于其揭晓位置之前；同一节点内保留先揭晓再使用的次序。
首次外貌识别可保留，之后减少反复以发色衣服姿态代名；中性称呼不能暗示未经确认的职业/关系/行为身份。
返回 {{"beats":[{{"event_ids":["E00001","E00002"],"section":"opening|development|conflict|climax|ending",
"title":"这一段推进什么","viewer_question":"观众此刻关心的已有问题","payoff":"本节点给出的已有推进",
"reason":"为何需要这一段"}}],"essential_event_ids":["不能漏掉的动机/转折/结果事件ID"],
"milestones":{{"conflict":"事件ID","climax":"事件ID","rules":null,"goal_or_stakes":null,"ending":"事件ID"}},
"rationale":"整体叙事链和信息揭示方式"}}。
每个事件只在一个 beat 的 event_ids 中出现；因果前提由程序另附供核验，不需要反复讲。
所有ID必须来自上方。不要补造事件。essential_event_ids 和每个非空里程碑必须进入 beats。
"""
        by_id = {e["event_id"]:e for e in events}
        feedback = ""
        for attempt in range(3):
            try:
                raw = self.model.ask("direct", prompt+feedback, fresh=attempt > 0)
                modern = "beats" in raw
                require(modern or getattr(self.model,'planning_version',4) < 5,
                        '新版规划必须返回多事件 beats，不能退回旧的事件清单')
                specs = array(raw.get("beats"), "beats") if modern else [
                    {"event_ids":[eid]} for eid in array(raw.get("event_ids"), "event_ids")]
                require(bool(specs), "剧情规划不能为空")
                milestones = raw.get("milestones")
                require(isinstance(milestones, dict), "缺少剧情里程碑")
                for key in ("conflict", "climax", "rules", "goal_or_stakes", "ending"):
                    require(key in milestones and (milestones[key] in by_id if isinstance(milestones[key], str)
                            else milestones[key] is None and key in {"rules", "goal_or_stakes"}), "剧情里程碑无效")
                seen = set()
                for spec in specs:
                    require(isinstance(spec,dict), '叙事节点格式无效')
                    ids = array(spec.get("event_ids"), "beat.event_ids")
                    require(ids and all(isinstance(eid,str) and eid in by_id and eid not in seen for eid in ids)
                            and len(ids) == len(set(ids)), "导演引用未知、重复或非剧情事件")
                    seen.update(ids)
                essential = raw.get("essential_event_ids", [])
                require(isinstance(essential,list) and all(isinstance(eid,str) and eid in by_id for eid in essential), "关键转折引用无效")
                mandatory = set(essential) | {eid for eid in milestones.values() if eid}
                if modern:
                    require(mandatory <= seen, "缺少关键动机、转折、规则或结果")
                    ranges = [(min(by_id[e]['order'] for e in s['event_ids']),
                               max(by_id[e]['order'] for e in s['event_ids'])) for s in specs]
                    require(all(a[1] < b[0] for a,b in zip(ranges,ranges[1:])),
                            '叙事节点不能颠倒或交错原作事件及信息揭示顺序')
                    audit_prompt = """Story Plan Review v2。独立检查故事链完整性，不评价时长或段数。
仅依据整话事件检查：目标→阻碍→改变决定的触发条件→行动→结果是否讲通。
同时检查进入故事必需的背景、背景带来的具体关注点及后续行动承接，允许背景开场及有作用的长背景、氛围和情绪停顿。
不能为了吸引力隐藏原作已经明确的信息或提前泄露身份；不要求固定结构或几秒进入冲突。
重点寻找被遗漏的态度变化原因、交换条件、物资线索、选择规则与结果；不要把可省略的表情当关键事件。
已在任一 beat.event_ids 内的事件已经进入规划，绝不能列入 missing_event_ids。
不在 essential_event_ids 标签里但已在 beats 内，不是剧情遗漏；不能仅因标签不够多判失败。
原作没有显示的开枪者、因果或清理结果不是可补造的缺失事件；规划保持未知或不做断言即可。
只阻止实际遗漏、歪曲或前后矛盾，不要求把全部表情、装备和每个细节列为必需事件。
返回 {"passed":true,"missing_event_ids":[],"issues":[]}。passed=false 时列真实缺失事件ID及具体问题。
资料：""" + prompt_json({"events":facts,"plan":raw,"already_planned_event_ids":sorted(seen)})
                    correction = ''
                    for audit_attempt in range(2):
                        audit = self.model.ask('story_plan_review',audit_prompt+correction,
                                               fresh=attempt > 0 or audit_attempt > 0)
                        require(type(audit.get('passed')) is bool and isinstance(audit.get('missing_event_ids'),list)
                                and isinstance(audit.get('issues'),list),'剧情完整性复核格式无效')
                        require(all(isinstance(eid,str) and eid in by_id for eid in audit['missing_event_ids']),
                                '完整性复核引用未知事件')
                        false_missing = set(audit['missing_event_ids']) & seen
                        if not false_missing:
                            break
                        correction = '\n上次复核有确定的误报：'+prompt_json(sorted(false_missing))+' 已在 beats.event_ids 内。请重新核对规划正文，不以 essential 标签或来源未说明的结果为缺口。上次复核：'+prompt_json(audit)
                    require(type(audit.get("passed")) is bool and isinstance(audit.get("missing_event_ids"),list)
                            and isinstance(audit.get("issues"),list), "剧情完整性复核格式无效")
                    require(all(eid in by_id for eid in audit["missing_event_ids"]), "完整性复核引用未知事件")
                    require(not (set(audit['missing_event_ids']) & seen),'完整性复核仍把已规划事件误报为缺失')
                    require(audit["passed"] and not audit["missing_event_ids"] and not audit["issues"],
                            "剧情链缺口："+prompt_json(audit))
                else:
                    for eid in mandatory - seen:
                        specs.append({"event_ids":[eid]})
                    specs.sort(key=lambda spec: min(by_id[eid]["order"] for eid in spec["event_ids"]))
                    audit = None
                break
            except StoryError as exc:
                if attempt == 2:
                    raise
                feedback = f"\n上一版需修正：{exc}。保留已有正确内容，返回完整规划。不得通过删减必要剧情来缩短。"
        beats = []
        for i,spec in enumerate(specs):
            ids = sorted(spec["event_ids"], key=lambda eid:by_id[eid]["order"])
            context = context_ids(bible, ids)
            section = "opening" if i == 0 else "ending" if i == len(specs)-1 else (
                "climax" if milestones["climax"] in ids else spec.get("section", "development"))
            require(section in {"opening","ending","climax","conflict","development"}, "叙事职能无效")
            beat = {"event_id":ids[0], "context_event_ids":context,
                    "source_panel_ids":source_ids(bible, ids+context), "section":section,
                    "reason":spec.get("reason",raw.get("rationale","剧情推进"))}
            if modern:
                beat.update(event_ids=ids,title=spec.get("title",""),viewer_question=spec.get("viewer_question",""),
                            payoff=spec.get("payoff",""))
            beats.append(beat)
        chosen = {eid for spec in specs for eid in spec["event_ids"]}
        result = {"bible_id":bible["bible_id"], "style":style, "beats":beats,
                  "opening_event_id":beats[0]["event_id"], "climax_event_id":milestones["climax"],
                  "ending_event_id":milestones["ending"], "milestones":milestones,
                  "omitted_event_ids":[e["event_id"] for e in events if e["event_id"] not in chosen],
                  "budget_note":"故事推进决定篇幅；不按时长、事件数或字数压缩",
                  "ending_rule":"兑现本话目标，用已有的下一步目标或未解决风险收束"}
        if modern:
            result.update(version=5, objective="continuous_story_engagement", essential_event_ids=essential,
                          completeness_review=audit)
        return result
