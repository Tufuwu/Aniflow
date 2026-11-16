"""Chapter-level story planning: motivation, change and payoff determine length."""
from copy import deepcopy
from pipeline.story_model import array, prompt_json, require
from pipeline.story_bible import StoryError, digest
from pipeline.story_context import context_ids, source_ids
from pipeline.planning_material import planning_material, planning_json

STYLES = {"faithful": "忠实叙事", "payoff": "爽点解说", "suspense": "高悬念"}
PLANNING_CONTRACT_VERSION = 5  # Director selection/beat duties only; independent of integration.


class NarrativeDirector:
    def __init__(self, model=None, settings=None):
        self.model = model
        self.settings = deepcopy(settings)
        if settings is not None:
            require(isinstance(settings,dict) and all(isinstance(settings.get(k),str)
                    for k in ('revision','background','characters')), '规划设定格式无效')

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
        material = planning_material(bible, self.settings)
        prompt = f"""Narrative Director v8。为连续观看选择值得讲的内容，组织自然、连贯、有吸引力的整话讲述路线，风格 {STYLES[style]}。
围绕人物处境、目标、阻碍、选择、行动、反应和结果，发挥原作已有的反差、拌嘴、危险、期待及情绪兑现。
没有目标时长，没有固定段数、事件上限、字数或几秒进入冲突的配额；不默认选入全部事件，也不以尽量少选为目标。
规划输入省略本地取证编号及空列表，来源按event_id回查；省略的列表按空处理。actions与facets动作完全一致时只保留facets；状态标签保留作参考，不等于局部理解的最终结论。
整话证据事件（按原作顺序）：{planning_json(material['events'])}
作品世界背景与主要人物设定：{planning_json(material['settings'])}
上述设定用于理解作品和规划叙事，不是上一话剧情或本话新事件；不能据此新增本话事件、因果、能力、人物归属或未来剧情。
内部知道主角姓名、性别、职业和关系，不等于本话已经揭晓或已对应某个C编号；与本话证据冲突时保留待核，不覆盖原分析。
未在本话揭示的身份、关系及后续线索不得提前进入节点标题、关注点或兑现，不为设定里的未来信息补造事件ID。
上一话剧情与已固定版本的历史资料：{planning_json(material['history'])}
历史用于理解当前选择，不机械重讲；用户参考资料须原图核对。historical 事件不能当作本话新剧情。
人物档案及关系：{planning_json(material['characters'])}
逐事件身份可用位置：{planning_json(material['identity_positions'])}
身份表no_confirmed_reveals表示尚无已确认的姓名/关系揭示记录，不自行确认人物归属；这是内部记录状态，不能把程序未识别清楚的人物当成观众悬念。
先通读整话，理解人物目标、阻碍、改变决定的具体消息/条件、行动、反应与结果，再选择讲述重点、组织 beats 与承接。
整合事件是已保存的本话基础；kind/time 区分当前动作、设定、回忆与历史说明，不把它们全当当前行动。
unknown、uncertain、facets/status/pending_causes 和旧分析结论供参考。允许结合整话上下文、对白衔接、人物反应及后续行动，重新理解局部动作、交流归属、情绪及事件承接。
选择与前后情节一致、附加假设较少的合理解释，不因局部记录不完整放弃组织故事；仍无法判断且不影响主线的细节可以省略。推演不回写成原始确认事实，不补齐待缺历史，不改动原作明确的重要事件、人物关系、关键因果和结果。
按讲述作用取舍：即使提供新信息，对剧情理解、人物辨认、危险、氛围、趣味或情绪没有实际作用的细节也可不采用，不限于删除重复。
importance、旧role和旧模型判断仅供参考，不能机械决定删留；信息独立出现不等于值得讲，更不等于必保。不要用表情、衣服、颜色、办公用品或站位动作清单淹没人物行动与反应，有辨认、氛围或趣味作用时则保留。
每个beat承担一个连续的推进作用，可以涵盖多个事件、动作和对白。让处境如何影响选择、行动怎样引出反应与结果自然衔接，不按每格或每个新细节单独设节点，不将原始摘要换个字段继续罗列。
节点字段各有分工：title写本节点的重要变化；viewer_question写观众继续关注的具体理由，可以是陈述句，不必强造疑问；payoff写本节点实际推进、解释或兑现了什么；reason写为什么保留，以及怎样承接前后。避免四个字段同义重复，不为制造高潮新增危险、反转或空泛提问。
开场让观众理解具体处境、当前问题和目标，保留有作用的环境压力。可用已有的世界异常或背景事实开场，也可随行动交代，不强制固定开场结构。
背景解释眼前的危险、人物选择与行动代价。区分现在理解选择所需、稍后需要时解释、只供核验无需讲述的内容；可以保留有作用的长背景，已启动的任务不被无关历史打断。
背景安排仍保持原作明确的重要时间、关键因果和信息揭示顺序，不把回忆、历史或设定改成当前发生的动作，不借调整位置提前泄露揭晓。
检查每次决定变化：之前态度是什么、什么原文/动作触发改变、之后做了什么，在同一或相邻节点讲清；不能因触发事件importance低而漏掉。若有选择/任务，保留理解行动必需的规则与代价。
保留有作用的配角反应、救援后的哭喊、安慰、沉默及情绪停顿，让已有期待获得兑现；不要求每个停顿都带来新客观事件。
首章或历史资料不可用时，从已有事件与临时人物建立理解，不要求补齐姓名/性别/关系。称呼稳定、能分清人物即可；中性称呼不暗示未经确认的职业、关系或行为身份。
姓名/关系/身份的内部理解不能提前用于原作揭晓位置之前；同一节点内保留先揭晓再使用的次序，标题、关注理由和兑现也不能提前泄露。
结尾先兑现本话已有的行动或情绪结果，再承接原有的下一步目标或未解决风险，不能只留悬念而跳过本话收束。
essential_event_ids只标记缺失后会破坏剧情理解、重要转折、必要背景或情绪兑现的事件。必要触发、重要行动及结果按其作用保护，不因每条记录独立、有新信息或旧importance/role就设为必保；不设必保数量、比例或删减目标。
milestones对应本话已有的阻碍、行动、规则、目标/代价、情绪兑现和收束；conflict/climax沿用字段名，不要求额外制造戏剧冲突，rules、goal_or_stakes没有对应内容时填null。
返回 {{"beats":[{{"event_ids":["E00001","E00002"],"section":"opening|development|conflict|climax|ending",
"title":"本节点的重要变化","viewer_question":"继续关注的具体理由，可用陈述句","payoff":"实际推进、解释或兑现的内容",
"reason":"保留理由及与前后的承接"}}],"essential_event_ids":["缺失会破坏理解、转折、必要背景或情绪兑现的事件ID"],
"milestones":{{"conflict":"事件ID","climax":"事件ID","rules":null,"goal_or_stakes":null,"ending":"事件ID"}},
"rationale":"整体叙事链和信息揭示方式"}}。
每个采用的事件只在一个beat的event_ids中出现；按原作顺序组织，节点之间不得颠倒或交错事件范围。因果前提由程序另附供核验，不等于已讲述，理解选择所必需的触发仍须进入beats，不需要重复引用。
所有ID必须来自上方剧情事件，不得引用非剧情内容或补造事件。essential_event_ids和每个非空里程碑必须进入beats；普通非必保事件允许不进入beats，遗漏列表由程序生成，不必为覆盖全部ID而选入。
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
                    require(mandatory <= seen, "缺少关键动机、转折、规则或结果：" + prompt_json(sorted(mandatory - seen)))
                    ranges = [(min(by_id[e]['order'] for e in s['event_ids']),
                               max(by_id[e]['order'] for e in s['event_ids'])) for s in specs]
                    require(all(a[1] < b[0] for a,b in zip(ranges,ranges[1:])),
                            '叙事节点不能颠倒或交错原作事件及信息揭示顺序')
                else:
                    for eid in mandatory - seen:
                        specs.append({"event_ids":[eid]})
                    specs.sort(key=lambda spec: min(by_id[eid]["order"] for eid in spec["event_ids"]))
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
                          planning_validation={'version':PLANNING_CONTRACT_VERSION, 'mode':'program_constraints'})
        if self.settings:
            result['settings_basis'] = {'revision':self.settings['revision'],
                'hash':digest({k:self.settings[k] for k in ('background','characters')})}
        return result
