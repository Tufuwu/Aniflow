"""Chinese Douyin narration constrained to a director-selected event."""
import re
from pipeline.story_model import prompt_json, require, text, validated_result
from pipeline.story_bible import StoryError
from pipeline.narrative_director import STYLES
from pipeline.narration_review import repeat_candidates, information_issues, content_issues, review_basis
from pipeline.character_identity import premature_name_issues
from pipeline.semantic_grounding import grounded_units
from pipeline.fact_material import fact_material, writing_material, FACT_CONTRACT_VERSION
from pipeline.input_compaction import compact_json
from pipeline.opening_recap import with_opening_recap
from pipeline.prompt_material import (script_text, revision_feedback,
                                     text_evidence, bind_context_usage,
                                     editor_material, fact_sheet_view)


NARRATIVE_RULES = '''
开场尽早讲清有证据的具体处境、当前问题与目标；保留环境压力，不按几秒进入冲突判断。
开篇提供理解环境、基本异常、危险限制和当前问题所必需的背景；不能把避免堆设定理解为省掉必要背景。
从原作背景中寻找有具体后果、反差、危险、代价或待解决问题的信息，讲清它怎样影响此刻的选择。
背景清楚之外，还须建立值得继续关注的具体问题，并由后续行动承接，不能用空泛提问代替。
允许以世界异常或背景事实开场，也可随场景和行动逐步交代；不强制先冲突后背景，不套固定开场结构。
坐下、拿手机、查看消息、反复催促只在改变理解或压力时保留，不累加同一过程。
态度转折须讲通“原先拒绝/警告—触发消息—改变决定—行动—兑现”；只使用原作实际存在的环节。
不得替人物解释贪婪、善良、责任感等心理；没有转折的故事不制造转折。
背景按用途区分：现在必须知道才能理解选择、可在稍后需要时解释、仅供核验无需进入正文。
检查已经启动的任务是否被设定介绍打断；有作用的长背景可以保留，不集中朗读全部历史，也不按长度全删。
调整解释位置须保持原作明确的时间、因果及叙事信息揭示顺序；不能提前泄露尚未揭晓的事实。
人物称呼要让只听声音的人持续分辨角色，不能只数外貌词；首次必要的外貌识别可以保留。
姓名必须有身份对应证据；未知姓名、性别、关系或说话者不补齐。中性称呼也不能暗示未经证实的职业、关系或行为身份。
内部读完整章已知不等于观众此刻应知；对照 identity_availability 的揭晓位置，节点内也须先揭晓再使用姓名/关系。
available_by_node_end 只是本节点末尾的可用上界，不能自动用于该节点开头；旧资料缺少揭晓位置时先核验，不默认已知。
稳定称呼优先采用有据 label_claims；人物换称呼时让听者自然接上，未知记录和内部编号不进入正文。
行动场面围绕危险、行动、反应和局势变化组织，姿态、手、鞋、拟声只在帮助理解时保留。
不擅自确定开枪者、受击者、死亡结果或未确认的因果；非必要且无法核实的细节可以省略。
哭喊、安慰、迟疑、沉默和情绪停顿可构成推进；让兑现有空间，不被办公用品、外貌或站位动作清单淹没。
跨段比较实际目标、行动和结果；有效回扣须指出新的理解、情绪或兑现，不能凭换词判为新信息。
不设目标总时长、固定段数、每段字数或切镜频率；不要把每句话都新增客观事件作为标准。
像直接说给观众听一样围绕人物问题、行动、反应和局势变化表达，避免取证报告、逐格说明和动作清单语气。
长短句结合，重要消息、态度变化、哭喊、安慰与沉默可留必要空间；不机械追求短句。
减少重复过程交代和生硬连接，不按词语黑名单批量删改，也不把所有对白改成旁白转述。
不依靠网络热词、夸张感叹、空泛提问和套话吸引观众；不能为顺口补造心理、主体、结果或因果，不能丢转折触发与兑现。
不以句长、感叹号数量、外貌词次数代替口语自然度判断。
prior_context 是生成前提供的背景、主要人物资料及上一话摘要；沿用有证据的历史连续性，不机械复述上一话。
plan.opening_recap 存在时，只在 anchor_event_id 对应的全稿首段，第一句简要回顾与本话相关的上一话内容，随后自然转入本话。
回顾仅从 opening_recap.event_ids 对应事件及上一话摘要中取有据内容，保留其确定性；不复述整话、不另加段落，后续段落不再重复开场回顾。
回顾是历史，不当作本话新发生，不为过渡补造时间、因果或人物关系；没有 opening_recap 时不写填充前情。
series_notes 属于参考线索，不能代替原图归属与事实核验；历史事实只作必要补充，不升级为本话新发生，也不凭摘要补因果。
'''

STYLE_RULES = {
    "faithful": "按原作顺序，清楚交代人物和动作，不替角色编内心。",
    "payoff": "突出原作已有的压迫、反击、反差和结果；没有反转就不制造反转，长短句服从情节。",
    "suspense": "用本话已知危险或尚未解答的问题吸引读者，保持原作信息揭示顺序，不隐藏已经明确的信息，不虚构谜团。",
}


def narration_lint(segments):
    issues = []
    pattern = r'一段.{0,8}(?:对白|对话框)|(?:红色|蓝色|浅蓝色)(?:对白|对话框)(?:里|内)?写着|画面(?:中|显示|出现)|背对(?:画面|镜头)|旁白说|(?:是谁|身份|同伴是谁).{0,8}(?:没交代|未交代|没有交代|没看清|不明)'
    for segment in segments:
        internal = r'(?<![A-Za-z0-9])C\d+(?![A-Za-z0-9])|未知人物(?:\d+|C\d+)'
        match = re.search(pattern+'|'+internal,segment['text'])
        if match:
            issues.append({'event_id':segment['event_id'],'detail':'读图说明或无意义的未知叙述：'+match.group()})
    return issues


class ScriptWriter:
    def __init__(self, model, store=None):
        self.model = model
        self.store = store or getattr(model,"store",None)

    def write_segment(self, bible, plan, beat, previous="", fresh=False, feedback=None, original_text=None, facts=None):
        plan = with_opening_recap(bible, plan)
        beat = next((b for b in plan['beats'] if b['event_id'] == beat['event_id']), beat)
        event = next(e for e in bible["events"] if e["event_id"] == beat["event_id"])
        prompt = f"""Script Writer v2。生成适合口播的中文故事解说，风格：{STYLES[plan['style']]}。
{STYLE_RULES[plan['style']]} 当前段落职能：{beat['section']}。篇幅由必要的剧情推进决定，没有目标字数或时长。
上段已说：{previous}
全稿叙事路线：{prompt_json(editor_material(bible,plan,facts))}
当前节点：{prompt_json({k:beat[k] for k in ('event_id','event_ids','context_event_ids','section') if k in beat})}
本段事实依据：{prompt_json(text_evidence(bible,[beat],facts))}
用自然的长短句推进故事，使用中文标点。开场可从有具体后果的背景异常、场景或行动进入，结尾仅保留本话已有悬念。
不得编造身份、亲属、动机、数值、能力、未来、动作结果；不要使用图中/分镜/画面等词。
未确认身份时使用有依据的稳定中性称呼，不把 C001、未知人物1 等内部标识读进文案；性别、亲属称呼也需要依据。
不要重复上段。不要把对白的猜测当旁白事实。因果仅使用事件 causes 的明确依据。
未知身份留在证据库，不要把“是谁还没交代”“身份还没看清”写进口播；可以省去非必要细节。
把消息写成有依据的人物交流，不要写“一段红色对白写着”“旁白说”等读图说明。
只写可核对的事实；不要为凑字数补写“一切看似平常”“无人知晓”等无依据评价。
用自然叙事串起人物的关键行动，少写衣服颜色、表情和发光特效。关系有原文依据时可以称呼父母或哥哥。
结尾围绕原作已有的未完成目标或具体风险收束，不用“画面到这里”“仍没有答案”等空泛句子。
返回 {{"text":"中文解说稿"}}。"""
        prompt += NARRATIVE_RULES
        if feedback:
            prompt += f"\n上一版核验问题（必须修正，不可补造证据）：{prompt_json(revision_feedback(feedback))}"
        if original_text is not None:
            prompt += '\n当前段原稿：'+prompt_json(original_text)+'''\n直接针对原稿修正上述核验指出的事实。
保留正确句子和本段剧情推进，不从取证记录重写整段，不新增外貌或姿态清单。
无法核实的非必要细节可以删除或用不指认身份的自然说法表达；不能补造说话者或因果。'''
        output = self.model.ask("write", prompt, fresh=fresh)
        narration = text(output.get("text"), "解说稿")
        require(len(narration) <= 10000, "单次模型输出异常过长")
        require(any("\u4e00" <= c <= "\u9fff" for c in narration), "解说稿必须使用中文")
        result = {"segment_id": f"S_{event['event_id']}", "event_id": event["event_id"],
                "source_panel_ids": list(beat["source_panel_ids"]), "context_event_ids":beat.get("context_event_ids", []), "section": beat["section"],
                "text": narration, "review_status": "pending", "fact_check": None}
        if "event_ids" in beat:
            result["event_ids"] = list(beat["event_ids"])
        return result

    def _distill_facts(self, bible, plan, feedback=None, fresh=False):
        """Compose all beats together before independent factual checks."""
        material = fact_material(bible, plan)
        digest_prompt = '''Story Digest v3。按当前规划将取证记录整理为共用事实简表，不写正式口播。
输入省略本地定位编号、完全重复的动作/同源对白证据副本及空列表，省略列表按空处理，事件来源由程序按event_id回查。
identity_availability为no_confirmed_reveals表示没有已确认的姓名/关系揭示，仍须保留人物未知状态；节点标题、关注点与兑现是规划意图，不是新增事实，事实和确定性以events原文为准。
本话整合事件是共同基础；按当前规划整理世界设定、环境异常、危险限制、人物行动及转折，不另写重复背景或人物摘要。
保留整合事件 kind/time/facets/status/pending_causes 的原作时间、揭示位置、未知与待补历史，不能把说明或回忆改写为当前动作。
保留背景对当前选择的作用、转折触发、行动与兑现，保持原作时间、因果和身份揭晓顺序；未知保持未知。
原始 summary 含外貌、颜色、拟声和画面位置，是取证记录，不能照单全收。
每个节点整理与目标、阻碍、决定变化、行动、结果有作用的事实；连续表情、取手机、脚步、装备品牌通常无需单独保留。
保留改变决定的触发信息、规则、代价和兑现；没有目标时长或字数要求。未知说话人不要指名，原作未说明的因果不得补造。
跨节点为同一人物使用有据的稳定称呼；青年、同伴、被困男子也分别需要年龄、关系或处境依据，不能当成通用替代词。
返回 {"nodes":[{"event_id":"节点锚点","facts":[{"text":"剧情事实","event_ids":["依据ID"],
"usage":"narrate|defer|verification_only","reason":"为何现在讲/在本节点稍后讲/仅供核验"}]}]}。
defer 只能在不改变原作时间、因果和揭示顺序的前提下使用；verification_only 不送入口播依据，不计为关键剧情已讲述。
每个 plan.beats 对应且仅对应一个 node，锚点与顺序必须相同。
facts.event_ids 只能来自当前节点 event_ids 或 context_event_ids。所有 essential_event_ids 必须有对应事实。
每条事实应有可核对的事件依据，不把关系猜测或未知身份写成事实。
'''
        if material['context_catalog']:
            digest_prompt += '''context_catalog 是已有跨话记忆或用户参考的来源索引，不是本话额外事实。按当前规划逐条决定用途，
在相关 node 的 context_usage 中各记录一次 {context_id,usage:"supplement|covered|irrelevant|uncertain",reason}。
supplement 仅用于简表缺少且影响理解当前选择、限制或因果的历史规则、人物状态、已确认关系或未完成目标。
covered 必须说明已由哪些事实涵盖，且来源章节、事件、版本与原图哈希一致；换个摘要仍是重复，不能作为 supplement。
同一事件可能含多条事实，不能仅因事件已引用就丢掉尚缺的必要信息。来源不足以判重时用 uncertain，不猜测、不静默丢弃。
irrelevant 说明为何与本规划理解无关。用户参考、模型推测、未知关系与未揭晓身份不升级为确认事实。
本话事实已涵盖的内容无需补充；第一话无跨话历史时前情为空，不生成填充前情。
'''
        else:
            digest_prompt += 'context_catalog为空，本次没有可补充资料，不生成填充前情或context_usage。\n'
        digest_prompt += '资料：'+compact_json(material)
        if feedback:
            digest_prompt += '\n上一轮需修正的问题（仅据原文修正，不补造）：'+prompt_json(revision_feedback(feedback))
        distilled = self.model.ask('story_digest',digest_prompt,fresh=fresh)
        nodes = distilled.get('nodes')
        require(isinstance(nodes,list) and len(nodes)==len(plan['beats']),'剧情事实整理漏节点')
        covered = set()
        for node,beat in zip(nodes,plan['beats']):
            require(isinstance(node,dict) and node.get('event_id')==beat['event_id'],'剧情事实节点顺序无效')
            facts = node.get('facts')
            require(isinstance(facts,list) and facts,'剧情事实不能为空')
            allowed = set(beat['event_ids']+beat.get('context_event_ids',[]))
            for fact in facts:
                require(isinstance(fact,dict) and isinstance(fact.get('text'),str) and fact['text'].strip(), '剧情事实文字无效')
                ids = fact.get('event_ids')
                require(isinstance(ids,list) and ids,
                        f"剧情事实节点 {node['event_id']} 的 event_ids 必须是非空数组，收到："+prompt_json(ids))
                require(all(isinstance(eid,str) and eid in allowed for eid in ids),
                        f"剧情事实节点 {node['event_id']} 引用越界；允许："+prompt_json(sorted(allowed)))
                usage = fact.get('usage','narrate')
                require(usage in {'narrate','defer','verification_only'}, '剧情事实用途无效')
                if usage != 'verification_only':
                    covered.update(ids)
        require(set(plan.get('essential_event_ids',[]))<=covered,'剧情事实遗漏关键转折或结果')
        return bind_context_usage(bible,nodes)

    def prepare_facts(self, bible, plan):
        def compute():
            feedback = None
            for attempt in range(3):
                try:
                    return self._distill_facts(bible,plan,feedback=feedback,fresh=attempt>0)
                except StoryError as exc:
                    if attempt == 2:
                        raise
                    feedback = {'error':str(exc)}
        if self.store is None:
            return compute()
        return validated_result(self.store,self.model,'story_digest',
            {'contract':FACT_CONTRACT_VERSION,'bible':bible,'plan':plan},compute)

    def write_story(self, bible, plan, feedback=None, fresh=False, facts=None, variant=None):
        plan = with_opening_recap(bible, plan)
        skeleton = {'segments':[{'event_id':b['event_id'],'text':'在这里填写本节点的连贯口播'} for b in plan['beats']]}
        if facts is None:
            material = writing_material(bible, plan)
        else:
            material = {**writing_material(bible, plan), 'nodes': fact_sheet_view(facts, spoken=True)}
        prompt = """Story Writer v3。为完整章节一次写出连贯、吸引人继续听的中文解说。
所有分析、规划及旧整理稿都是待核材料，经过模型处理不等于内容正确。
在本次写作中完成必要事实取舍、相邻事件归并和口语表达；不把全部原事件都当作必须逐项朗读的清单。
events是剧情依据，节点title/viewer_question/payoff/reason只是组织意图；与原事件或未知状态冲突时以原事件为准。
essential_event_ids中的必要动机、触发和结果须在正文实际表达，不能只靠来源ID覆盖。
identity_availability为no_confirmed_reveals表示尚无确认揭示，不确认人物归属。
有旧nodes整理稿时可参考其内容，但须对照events的具体证据与未知状态，不能因模型曾整理或确认就采信。
篇幅由讲清故事决定，没有目标时长、固定字数或段数配额。每个规划节点对应一个段落，不能漏节点。
这里的节点是 plan.beats，不是原始 events。一个节点内可有多个来源事件，不能为每个原始事件另写一段。
输出的 event_id 只能使用下面结构中的节点锚点，并保持节点顺序。只填 text，不增加、拆分或删除输出对象。
自然讲述人物的处境、选择、行动、反应和变化。必要背景可在适当位置交代，可从背景、场景或行动进入，不强制背景开场或先冲突后背景。
背景应解释具体后果、危险或代价，形成可由后续行动承接的关注点。消息、迟疑、哭喊、安慰与沉默可留必要空间。
保留人物目标、阻碍、态度变化的触发条件、行动与兑现。不要为了制造悬念隐藏已经明确的信息。
人物用有身份对应证据的稳定称呼；不知名字也能开始写作，青年、同伴、被困男子等称呼各自需要依据，不能给未知说话人擅自指名。
不朗读人物外貌清单、颜色、文字框、画面或分析过程。未知细节留在元数据，不写“是谁没交代”。
只讲原作证据支持的事实；对白中的推测必须保留为说法。绝不编造动机、关系、因果或后续。
每段事件必须严格来自该节点 event_ids 与 context_event_ids；context 只补必要信息，不反复复述。
原始 summary 是取证说明，不能照搬成口播。正文围绕节点 title/payoff 的推进，不逐条翻译全部来源事件。
外貌识别只在首次区分或确有必要时使用，此后沿用有依据的稳定称呼；不把衣服/发色反复当名字。
以人物行动讲故事，省去无作用的姿态、颜色、拟声字和装备品牌；也不要改成“一声喊出/有人说明/旁白提出疑问”的说明清单。
结尾先兑现本话已发生的结果，再承接已有的下一步目标，不增加空泛提问或预告。
返回 {"segments":[{"event_id":"节点的 event_id","text":"可直接配音的段落"}]}，覆盖全部节点，顺序一致。
identity_availability 保留逐节点揭晓限制，available_by_node_end 不是节点开头即可使用；同节点也须先揭晓后使用。
prior_context 仅为必要且非重复的跨话前情，首话无前情时为空；reference_context 为按需参考，两者不重复本话简表。
有 plan.opening_recap 时，首段第一句简要回顾与本话相关的上一话内容，随后自然转入本话；只用所列历史事件和摘要的有据内容，保留确定性。
回顾只占一句，放在原首段，不新增段落，不复述整话，不把历史当作本话新发生，不为过渡补造因果；没有 opening_recap 时直接进入本话。
uncertain 和 unverified_reference 不能作为确认事实；不得扩展本话范围。
候选可改变口语表达与组织方式，不能改变事实、时间、因果和揭晓顺序。不靠热词、夸张、空泛提问或套话补足吸引力。
必须严格填写的输出结构："""+prompt_json(skeleton)+"\n叙事依据：" + compact_json({
            **material,"style":STYLES[plan['style']]})
        if feedback:
            prompt += "\n须修正的全稿问题："+prompt_json(revision_feedback(feedback))
        if variant:
            prompt += '\n本候选的表达任务（不得改事实/时间/因果/身份揭晓顺序）：'+prompt_json(variant)
        raw = self.model.ask("write_story", prompt, fresh=fresh)
        rows = raw.get("segments")
        require(isinstance(rows,list) and len(rows) == len(plan['beats']),
                '整稿节点数量错误：必须严格按 plan.beats 锚点 '+prompt_json([b['event_id'] for b in plan['beats']])+
                ' 输出，不能按原始events逐一分段；文字可长可短，不要求缩短故事')
        result = []
        for row,beat in zip(rows,plan['beats']):
            require(isinstance(row,dict) and row.get('event_id') == beat['event_id'], "整稿节点顺序或引用无效")
            narration = text(row.get('text'), '解说稿')
            require(len(narration) <= 10000 and any('\u4e00' <= c <= '\u9fff' for c in narration), '整稿文字无效')
            result.append({"segment_id":f"S_{beat['event_id']}","event_id":beat['event_id'],
                           "event_ids":list(beat['event_ids']),"context_event_ids":beat.get('context_event_ids',[]),
                           "source_panel_ids":list(beat['source_panel_ids']),"section":beat['section'],
                           "text":narration,"review_status":"pending","fact_check":None})
        return result

    def review_story(self, bible, plan, segments, feedback=None, facts=None):
        candidates = repeat_candidates(segments)
        prompt = """Narration Review v5。通读全稿，检查故事推进、口头讲述与事实范围，不评价长度或要求压缩。
对照原事件分析检查正文取舍、动机、触发与结果，不能因事件ID存在就认为已讲清。
有fact_sheet时也只作待核整理稿，须与events的原有证据及未知状态对照；模型整理或确认不等于正确。
事实简表仍可能识别有误，未知及未揭晓限制不得升级；需要更多依据时定向指出事件和疑点，原图事实由选中稿的严格核验处理。
prior_context 是非重复前情，reference_context 是按需参考；uncertain 与 unverified_reference 不得升级为确认事实，也不能扩大本话范围。
有 plan.opening_recap 时，在 opening 项检查全稿是否以一句有据的上一话回顾开头、随后自然转入本话；遗漏或展开多句时定向修改首段，后文重复回顾时指出实际重复节点。
回顾不得把历史当作本话新发生，不得为衔接编造因果或升级确定性；没有 opening_recap 时不要求填充前情。
本环节没有提供原图，不能声称看见气泡指向、构图或人物动作；无法确认的疑点指向来源，留给严格逐句原图核验。
已经有原图核验通过的句子，不得仅凭对画面的猜测再次判错；仍可阻止文本本身明确存在的主线缺失、重复、矛盾或分析腔。
检查动机和转折是否被讲清、行动是否跳跃、信息是否重复、称呼是否前后一致、结尾是否兑现已有目标。
禁止读图说明（红色对白写着/画面显示/旁白说）、外貌清单和“是谁还没交代”等分析腔。
影响理解的转折触发、关键行动与兑现须在正文讲通，不接受仅有 event_id 的形式覆盖，不能把推测心理当动机补写。
原始事件是证据范围，不是每条都必须朗读的清单；自然叙述已讲清节点推进时，不要求补每个取手机、脚步、表情或外貌。
有作用的长背景、氛围和停顿可保留，不按长度全删，也不要求每句话都新增客观事件。
稳定自然称呼已能指代的人物，不要要求补姓名、性别或未经证实的关系。不要为未知说话者强行指定身份。
禁止要求补造原作未证实的开枪者、心理、区域是否被雾波及或因果。原作未显示的结果应省去断言，而不是要求编齐。
可建议省去无用的未知信息，不可要求把未知写成已知。反馈必须可依据已有证据执行。
会破坏理解的必要背景缺失、开篇没有具体关注点、人物难辨、取证或动作清单语气、动机缺失、矛盾、重复可作为有证据的定向问题。
没有逐个说明脚步、取手机、地址细节、未知说话人身份，不单独构成阻断。自然称呼已清楚时不要求补身份。
可选的措辞润色和原作本身的未知信息放入 suggestions，不要求凭空补齐，不能为追求完美反复返工。
没有有效问题时保留原稿，不无目标重写。候选表达可不同，但事实范围、时间、因果和揭晓顺序不可改变。
返回 {"passed":true,"blocking_issues":[],"suggestions":[]}；存在真实阻断才 passed=false。
blocking_issues 每项列 event_id、detail 和可依据现有证据执行的修正建议。
另返回 information_flow，逐段按原顺序列 {event_id,new_information:["逐字引用本段的新信息/行动/情绪变化/有作用的安静停顿"],
repeats:[{earlier_event_id,purpose:"redundant|callback|emphasis|payoff",detail:"比较两段实际讲述的目标、行动和结果",
new_information:"有效回扣必须逐字引用本段 new_information 中体现其新理解/情绪/兑现的一项"}]}。
即使用词不同也要检查连续段反复解释同一目标、行动或结果。不能仅因关键词相同判重复。
外貌称呼重复、姿态和构图清单若没有推进作用，应给出定向修改意见，不强行补姓名或心理。
对 repeat_candidates 每项返回 repeat_assessments:[{repeat_id,purpose:"redundant|callback|emphasis|payoff",
reason:"作用",new_information:"逐字引用后段新增推进，不能再引用重复文本"}]。
有作用的回扣、强调、伏笔兑现可保留；简单重说食物/工具/目标/搬运结果等已交代内容应标 redundant。
另返回 content_checks，按 opening,turns,exposition,references,action,emotion,ending 顺序逐项复核。
每项为 {aspect,verdict:"passed|issues|not_applicable",detail:"比较正文与证据后作出判断的具体理由",
citations:[{event_id:"稿件节点锚点",text:"当前正文连续原文"}],evidence_event_ids:["原始事件ID"],
issues:[{event_id:"仅需修改的节点锚点",detail:"具体影响",suggestion:"依据现有证据如何修正"}]}。
opening/exposition/references/action 另含 assessments 对象，各子项均须给出基于当前正文及证据的具体理由：
opening: {entry_context:"进入故事需要哪些背景，已交代/缺少什么",concrete_focus:"背景的后果/反差/危险/代价建立什么具体关注点",
follow_through:"后续哪些行动承接这个关注点"}；
exposition: {purpose:"必要背景、可后解释、仅核验分别怎样影响当前处境与选择；长背景与氛围有无作用",
placement:"是否保持原作时间、因果和揭示顺序，是否隐藏已明确的信息或提前泄露"}；
references: {listener_tracking:"听者不看画面能否持续分清人物并理解选择",identity_timing:"逐一对照姓名/关系/身份对应证据与揭晓位置",
grounded_labels:"稳定称呼的依据；中性词有无暗示未经确认的职业、关系、行为身份"}；
action: {oral_delivery:"引用哪些句子适合直接讲述，哪些仍像取证/逐格说明/动作清单；给出不补事实的改法，结合长短句、对白、消息与反应判断"}。
缺少这些检查不能当作复核通过。子项依据引用沿用本项 citations/evidence_event_ids，不增加分数或句长指标。
passed/issues 都须引用正文和事件依据；缺失内容可引用附近正文来定位。issues 必须给出定向修正项。
只有原作没有相关场面时才可 not_applicable，并说明原因；opening/references/ending 不能跳过。
opening 不机械计时或要求固定开场顺序；turns 核对实际转折的原态度、触发信息、决定、行动与兑现。
emotion 检查反应、哭喊、安慰或有作用的沉默是否有空间；ending 检查已有结果兑现与真正的新目标，避免重复收尾。
资料："""+compact_json({**editor_material(bible,with_opening_recap(bible,plan),facts,segments),"script":script_text(segments),
                       **text_evidence(bible,segments,facts)})
        prompt += '\nrepeat_candidates：'+prompt_json(candidates)
        if feedback:
            prompt += '\n上一份复核响应未通过校验：'+prompt_json(revision_feedback(feedback))+'。重新返回完整复核；citations.text 必须逐字复制当前正文连续片段，不能缩写、省略号代替或改标点；不要改写正文。'
        raw = self.model.ask('script_review',prompt,fresh=bool(feedback))
        return self.validate_review(bible,plan,segments,raw,candidates)

    def validate_review(self, bible, plan, segments, raw, candidates=None):
        """Same fail-closed contract for a full response or model quote repair."""
        candidates = repeat_candidates(segments) if candidates is None else candidates
        if 'blocking_issues' in raw:
            require(isinstance(raw['blocking_issues'],list) and isinstance(raw.get('suggestions',[]),list),'整稿复核格式无效')
            raw = {**raw,'issues':raw['blocking_issues']}
        require(type(raw.get('passed')) is bool and isinstance(raw.get('issues'),list), '整稿复核格式无效')
        flow_issues, information_review = information_issues(segments, raw, candidates)
        raw['information_review'] = information_review
        content_findings, content_review = content_issues(segments,bible,raw)
        raw['content_review'] = content_review
        raw['basis_hash'] = review_basis(bible,plan,segments)
        local_issues = narration_lint(segments) + flow_issues + content_findings + premature_name_issues(bible,segments)
        if local_issues:
            raw = {**raw,'passed':False,'issues':raw['issues']+local_issues}
        if information_review['status']!='reviewed' or content_review['status']!='reviewed' or raw['issues']:
            raw['passed'] = False
        raw['blocking_issues'] = raw['issues']
        return raw

    def revise_story(self,bible,plan,segments,feedback,facts=None):
        """Edit the actual rejected draft instead of recreating it from the evidence ledger."""
        targets = {r.get('event_id') for r in feedback.get('issues',[]) if isinstance(r,dict)}
        targets &= {s['event_id'] for s in segments}
        require(targets, '返修意见缺少具体节点，不能无目标地重写正确内容')
        require(not any(s.get('review_status')=='approved' for s in segments if s['event_id'] in targets),
                '已审核文案已锁定，不能定向覆盖')
        from copy import deepcopy
        writing_plan = with_opening_recap(bible, plan)
        scoped = deepcopy(segments)
        if writing_plan.get('opening_recap'):
            opening = writing_plan['beats'][0]
            for segment in scoped:
                if segment['event_id'] == opening['event_id'] and segment['event_id'] in targets:
                    segment.update({k:deepcopy(opening[k]) for k in
                                    ('event_ids','context_event_ids','source_panel_ids')})
        prompt = '''Story Revision v4。你是中文口播编辑，直接修改下面的当前整稿，解决具体审核问题。
当前稿件已经组织成叙事节点。保留节点锚点及顺序，不另按原始事件拆段，不从取证记录重新写一份。
把读图说明改为自然叙述：删除“画面/旁白说/字样/对话框”等编辑口吻，保留其有依据的情节内容。
用稳定自然的称呼串联同一人物；减少外貌姿态、颜色、脚步、拟声及装备清单。必要的安静场景可以保留。
未知说话者无需强行指名。证据没显示的动机、因果、身份和后果不能补造；非必要的未知细节可以省略。
只修改表达与衔接，不补新事实、不换动作发起者和承受者、不改变行动顺序。
保留真实的目标、拒绝/接受、改变决定的消息、行动和兑现。篇幅由故事决定，不按时长或字数压缩。
审核建议也可能要求无依据补全：这种建议不要照做，保持原稿有依据的说法。
只修改 revision_event_ids 中的节点，其他节点逐字保留。不能为消除重复而删掉后段真正的新目标。
返回 {"segments":[{"event_id":"原节点锚点","text":"修订后可直接配音的正文"}]}，可只返回需修改的节点。
资料：'''+compact_json({'current_script':[{'event_id':s['event_id'],'text':s['text']} for s in segments],
                       'review_feedback':revision_feedback(feedback),'revision_event_ids':sorted(targets),
                       **editor_material(bible,writing_plan,facts,scoped),
                       **text_evidence(bible,[s for s in scoped if s['event_id'] in targets],facts)}) + NARRATIVE_RULES
        raw = self.model.ask('rewrite_story',prompt,fresh=True)
        rows = raw.get('segments')
        require(isinstance(rows,list) and rows and all(isinstance(r,dict) and r.get('event_id') in
                {s['event_id'] for s in segments} for r in rows),'整稿返修锚点或顺序无效')
        require(len({r['event_id'] for r in rows})==len(rows) and targets <= {r['event_id'] for r in rows}, '整稿返修漏节点或重复节点')
        result = []
        for old, scope in zip(segments, scoped):
            row = next((r for r in rows if r['event_id']==old['event_id']),old)
            if old['event_id'] not in targets:
                require(row['text']==old['text'],'定向返修不能改写未被指出问题的段落')
            narration = text(row.get('text'),'返修稿')
            require(len(narration)<=10000 and any('\u4e00'<=c<='\u9fff' for c in narration),'返修稿文字无效')
            result.append(deepcopy(old) if narration==old['text'] and scope==old else
                          {**scope,'text':narration,'fact_check':None,'review_status':'pending'})
            check = result[-1].get('fact_check') or {}
            if (check.get('grounding') or {}).get('status')=='mapped' and not grounded_units(bible,result[-1]):
                result[-1]['fact_check'] = None
        return result

    def write(self, bible, plan):
        plan = with_opening_recap(bible, plan)
        segments = []
        for beat in plan["beats"]:
            segments.append(self.write_segment(bible, plan, beat,
                                              segments[-1]["text"] if segments else ""))
        return {"bible_id": bible["bible_id"], "style": plan["style"], "segments": segments}
