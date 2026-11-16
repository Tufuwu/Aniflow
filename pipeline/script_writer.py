"""Chinese Douyin narration constrained to a director-selected event."""
import re
from pipeline.story_model import prompt_json, require, text, validated_result
from pipeline.story_bible import StoryError
from pipeline.narrative_director import STYLES
from pipeline.narration_review import (polish_basis, ENGAGEMENT_POLISH_VERSION,
                                        ENGAGEMENT_POLISH_PROMPT_VERSION, ENGAGEMENT_POLISH_STYLE)
from pipeline.semantic_grounding import grounded_units
from pipeline.fact_material import fact_material, writing_material, FACT_CONTRACT_VERSION
from pipeline.input_compaction import compact_json
from pipeline.opening_recap import with_opening_recap
from pipeline.prompt_material import (script_text, revision_feedback,
                                     text_evidence, bind_context_usage,
                                     editor_material, fact_sheet_view)


NARRATIVE_RULES = '''
允许结合给定整话上下文、相邻事件、对白衔接、人物反应及后续行动，合理推演动作、说话者、语气、情绪和过渡，补足漫画留白，不要求每个表达都有单独对应的画面。
unknown和uncertain是已有分析的参考状态，可综合上下文重新判断；选择与前后情节一致、附加假设较少的解释，不因局部记录缺项退回逐格说明。难以判断且不影响主线的细节可省略。
推演须保持原作明确的重要事件、人物关系、关键因果和结果一致，不为刺激新增死亡、背叛、能力、隐藏身份或未来剧情。
开场尽早讲清具体处境、当前问题与目标；保留环境压力，不按几秒进入冲突判断。
开篇提供理解环境、基本异常、危险限制和当前问题所必需的背景；不能把避免堆设定理解为省掉必要背景。
从原作背景中寻找有具体后果、反差、危险、代价或待解决问题的信息，讲清它怎样影响此刻的选择。
背景清楚之外，还须建立值得继续关注的具体问题，并由后续行动承接，不能用空泛提问代替。
允许以世界异常或背景事实开场，也可随场景和行动逐步交代；不强制先冲突后背景，不套固定开场结构。
坐下、拿手机、查看消息、反复催促只在改变理解或压力时保留，不累加同一过程。
态度转折围绕原作已有的态度、触发信息、选择和行动讲通，可以合理补足局部反应，不按固定环节制造原作没有的转折。
可结合交流和行动表达人物当下情绪与想法，不凭空增加改变人物动机的深层解释。
背景按用途区分：现在必须知道才能理解选择、可在稍后需要时解释、仅供核验无需进入正文。
检查已经启动的任务是否被设定介绍打断；有作用的长背景可以保留，不集中朗读全部历史，也不按长度全删。
调整解释位置须保持原作明确的重要时间、关键因果及揭示顺序，不提前泄露重要揭晓。
人物称呼要让只听声音的人持续分辨角色，不能只数外貌词；首次必要的外貌识别可以保留。
说话者及交流关系可以结合上下文判断；姓名、重要身份关系以原作已有信息为准，不为衔接新增身份、职业或亲属关系。
内部读完整章已知不等于观众此刻应知；对照 identity_availability 的揭晓位置，节点内也须先揭晓再使用姓名/关系。
available_by_node_end 只是本节点末尾的可用上界，同节点也须先揭晓再使用姓名/重要关系；旧资料缺少位置时参照原作已有揭示，不因内部读完整话而提前使用。
稳定称呼优先采用有据 label_claims；人物换称呼时让听者自然接上，未知记录和内部编号不进入正文。
行动场面围绕危险、行动、反应和局势变化组织，姿态、手、鞋、拟声只在帮助理解时保留。
局部动作承接和因果可综合上下文理解；涉及关键行为主体、死亡和重大结果时，不凭一个模糊细节增加或改变剧情。
哭喊、安慰、迟疑、沉默和情绪停顿可构成推进；让兑现有空间，不被办公用品、外貌或站位动作清单淹没。
跨段比较实际目标、行动和结果；有效回扣须指出新的理解、情绪或兑现，不能凭换词判为新信息。
不设目标总时长、固定段数、每段字数或切镜频率；不要把每句话都新增客观事件作为标准。
像直接说给观众听一样围绕人物问题、行动、反应和局势变化表达，避免取证报告、逐格说明和动作清单语气。
长短句结合，重要消息、态度变化、哭喊、安慰与沉默可留必要空间；不机械追求短句。
减少重复过程交代和生硬连接，不按词语黑名单批量删改，也不把所有对白改成旁白转述。
发挥原作已有的反差、拌嘴、处境中的趣味和情绪变化，不依靠网络热词、夸张感叹、空泛提问和硬加段子；保留重要动机、转折触发与兑现。
不以句长、感叹号数量、外貌词次数代替口语自然度判断。
prior_context 是生成前提供的背景、主要人物资料及上一话摘要；沿用有证据的历史连续性，不机械复述上一话。
plan.opening_recap 存在时，只在 anchor_event_id 对应的全稿首段，第一句简要回顾与本话相关的上一话内容，随后自然转入本话。
回顾仅从 opening_recap.event_ids 对应事件及上一话摘要中取有据内容，保留其确定性；不复述整话、不另加段落，后续段落不再重复开场回顾。
回顾是历史，不当作本话新发生，不为过渡补造时间、因果或人物关系；没有 opening_recap 时不写填充前情。
series_notes 属于参考线索，不能凭它新增本话事件、重要身份或关键因果；历史只作必要补充，不升级为本话新发生。
'''

STYLE_RULES = {
    "faithful": "按原作顺序讲清人物选择与行动，允许结合上下文补足局部动作、交流和情绪，保持重要剧情一致。",
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
        prompt = f"""Script Writer v3。生成自然、连贯、有趣、适合口播的中文故事解说，风格：{STYLES[plan['style']]}。
{STYLE_RULES[plan['style']]} 当前段落职能：{beat['section']}。篇幅由必要的剧情推进决定，没有目标字数或时长。
上段已说：{previous}
全稿叙事路线：{prompt_json(editor_material(bible,plan,facts))}
当前节点：{prompt_json({k:beat[k] for k in ('event_id','event_ids','context_event_ids','section') if k in beat})}
本段事实依据：{prompt_json(text_evidence(bible,[beat],facts))}
用自然的长短句推进故事，使用中文标点。开场可从有具体后果的背景异常、场景或行动进入，结尾仅保留本话已有悬念。
允许结合上下文补足动作、说话者、语气、情绪和过渡，保持重要剧情、人物关系及结果一致；不要使用图中/分镜/画面等编辑说明。
称呼稳定，让听者分清人物，不把 C001、未知人物1 等内部标识读进文案；不为衔接新增姓名、职业或亲属关系。
不要重复上段。原作对白中明确的猜测和传闻保留其说法属性。causes供参考，结合事件、交流及反应理解合理承接，不受该字段是否完整限制。
未知身份留在证据库，不要把“是谁还没交代”“身份还没看清”写进口播；可以省去非必要细节。
把消息写成有依据的人物交流，不要写“一段红色对白写着”“旁白说”等读图说明。
让原作已有的危险、反差、拌嘴和情绪自然发挥，不为凑字数补写“一切看似平常”“无人知晓”等套话。
用自然叙事串起人物的关键行动，少写衣服颜色、表情和发光特效。关系有原文依据时可以称呼父母或哥哥。
结尾围绕原作已有的未完成目标或具体风险收束，不用“画面到这里”“仍没有答案”等空泛句子。
返回 {{"text":"中文解说稿"}}。"""
        prompt += NARRATIVE_RULES
        if feedback:
            prompt += f"\n上一版核验反馈（结合上下文复查真实矛盾和表达问题，不补造核验凭据）：{prompt_json(revision_feedback(feedback))}"
        if original_text is not None:
            prompt += '\n当前段原稿：'+prompt_json(original_text)+'''\n针对原稿解决真实矛盾、重要剧情偏离或表达问题。
保留正确句子和本段剧情推进，不从取证记录重写整段，不新增外貌或姿态清单。
合理补足不必因缺少单独画面就删除；结合上下文判断说话者及动作衔接，难以判断且非必要的细节可省略。'''
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

    def write_story(self, bible, plan, feedback=None, fresh=False, facts=None):
        plan = with_opening_recap(bible, plan)
        skeleton = {'segments':[{'event_id':b['event_id'],'text':'在这里填写本节点的连贯口播'} for b in plan['beats']]}
        if facts is None:
            material = writing_material(bible, plan)
        else:
            material = {**writing_material(bible, plan), 'nodes': fact_sheet_view(facts, spoken=True)}
        prompt = """Story Writer v5。为完整章节一次写出自然、连贯、有趣、吸引人继续听的中文解说。
所有分析、规划和旧nodes整理稿均为参考材料，经过模型处理不等于内容正确；综合events及给定整话上下文理解故事。
允许结合相邻事件、对白衔接、人物反应和后续行动，合理推演动作、说话者、语气、情绪和过渡，补足漫画留白，不要求每个表达都有单独对应的画面。
unknown和uncertain是已有分析的参考状态，可综合上下文重新判断；选择与前后情节一致、附加假设较少的解释，难以判断且不影响主线的细节可省略。
events中facets/evidence的数组条目按event_format同名字段顺序读取；对象条目保留额外信息，不能忽略。
省略的time按event_format.omitted_time理解，不表示本话当下或时间已确认；具体历史时间以显式记录为准，不把回忆和说明改成当前新发生的事件。
在本次写作中完成必要事实取舍、相邻事件归并和口语表达；原始summary是取证说明，不照搬，也不逐条朗读全部事件。
plan节点的title/viewer_question/payoff/reason只作组织意图，不是事实；与原作明确内容冲突时以原事件为准，分析中的未知可综合上下文判断。
按plan.beats顺序，每个规划节点写一个段落，一个节点可涵盖多个事件；严格填写下方输出结构，只填text，不增删或拆分对象。
每段实际讲述的事件限于该节点event_ids和context_event_ids，允许用整话上下文判断衔接，不把后续事件提前讲进本段；context只补必要信息。essential_event_ids中的必要动机、触发和结果须实际讲清，不能只靠来源ID覆盖。
以人物处境、目标、阻碍、选择、行动、反应和变化推进，保留态度转折的触发与兑现，不隐藏已经明确的信息。
必要背景可从开场或随场景行动交代，不强制背景开场或先冲突后背景；讲清具体后果、危险或代价，并由后续行动承接。
称呼自然稳定，让听者分清人物；说话者与当下情绪可结合上下文判断，姓名和重要关系以原作揭示及身份对应证据为准。
identity_availability为no_confirmed_reveals表示尚无姓名/重要关系的确认揭示；available_by_node_end只是节点末尾上界，同节点也须先揭晓后使用，不因内部已读完整话而提前使用。
保持原作明确的重要事件、人物关系、关键因果和结果一致，不为刺激新增死亡、背叛、能力、隐藏身份或未来剧情；原作对白中明确的猜测和传闻保留其说法属性。
外貌仅在首次区分或确有必要时使用，不反复当名字；省去无作用的姿态、颜色、拟声和装备品牌，不写外貌清单、画面说明、“是谁没交代”或“一声喊出/有人说明/旁白提出疑问”等分析清单。
prior_context为非重复跨话前情，首话为空；reference_context和unverified_reference为按需参考，不据此新增本话事件或重要身份关系。
有plan.opening_recap时，仅在原首段第一句，用所列历史事件和摘要的有据内容简要回顾相关前情，随后自然转入本话；保留确定性，不新增段落、不复述整话、不把历史当作本话新发生，不为过渡补造因果。没有opening_recap时直接进入本话。
结尾先兑现本话已有结果，再承接原有下一步目标，不增加空泛提问或预告。
篇幅由讲清故事决定，没有目标时长、固定字数或段数配额；长短句结合，消息、迟疑、哭喊、安慰与沉默可留必要空间。不靠热词、夸张感叹或套话补足吸引力。
让原作已有的反差、拌嘴、危险和情绪兑现成为讲述重点，不把外貌姿态和装备信息逐项朗读。
必须严格填写的输出结构："""+prompt_json(skeleton)+"\n叙事依据：" + compact_json({
            **material,"style":STYLES[plan['style']]})
        if feedback:
            prompt += "\n须修正的全稿问题："+prompt_json(revision_feedback(feedback))+"\n结合上下文判断反馈，修正真实矛盾及表达问题，合理补足不必因缺少单独画面就删除。"
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

    def polish_story(self, bible, plan, segments, fresh=False, facts=None, options=None):
        """Turn the whole draft into a performed recap, preserving event bindings."""
        from pipeline.narration_polish import (polish_options, script_view, polish_prompt,
                                               analyze_script, merge_polish, length_report)
        options = polish_options(options)
        script = script_view(segments)
        analysis = analyze_script(self.model,script,options,fresh) if options['analyze_first'] else None
        raw = self.model.ask('script_polish',polish_prompt(script,options,analysis),fresh=fresh)
        result,changed,unchanged = merge_polish(raw,segments,script)
        from pipeline.story_bible import digest
        model_identity = self.model.cache_identity('script_polish') if hasattr(self.model,'cache_identity') else None
        # A completion receipt is not semantic verification or human approval.
        completion = {'passed':True,'issues':[], 'basis_hash':polish_basis(bible,plan,result,options),
            'engagement_polish':{'version':ENGAGEMENT_POLISH_VERSION,'status':'completed',
                'style':ENGAGEMENT_POLISH_STYLE,'prompt_version':ENGAGEMENT_POLISH_PROMPT_VERSION,
                'options':options,'analysis':analysis,
                'model_basis':digest(model_identity) if model_identity is not None else None,
                'model':{k:model_identity[k] for k in ('model','backend','host') if k in model_identity}
                        if model_identity is not None else None,
                'validation':{'structure':'passed','event_order':'passed','locked_text':'passed',
                              'mode':'performance_rewrite',
                              'length':length_report(script,result,options)},
                'ignored_unchanged_event_ids':unchanged,
                'input_hash':polish_basis(bible,plan,segments,options),'changed_event_ids':changed}}
        return {'segments':result,'completion':completion}

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
        prompt = '''Story Revision v5。你是中文口播编辑，直接修改下面的当前整稿，解决具体审核问题，让故事自然、连贯、有趣。
当前稿件已经组织成叙事节点。保留节点锚点及顺序，不另按原始事件拆段，不从取证记录重新写一份。
把读图说明改为自然叙述：删除“画面/旁白说/字样/对话框”等编辑口吻，保留其有依据的情节内容。
用稳定自然的称呼串联同一人物；减少外貌姿态、颜色、脚步、拟声及装备清单。必要的安静场景可以保留。
允许结合整话上下文合理补足动作、说话者、语气、情绪和过渡，不要求每个表达都有单独对应的画面；unknown和uncertain供参考，可综合上下文重新判断。
保持原作明确的重要事件、人物关系、关键因果和结果一致；不为刺激新增死亡、背叛、能力、隐藏身份或未来剧情。
保留真实的目标、拒绝/接受、改变决定的消息、行动和兑现。篇幅由故事决定，不按时长或字数压缩。
审核意见也可能过于保守或误读：结合上下文判断真实矛盾、重要剧情偏离及表达问题，不因缺少单独画面就删除合理补足；难以判断且非必要的细节可省略。
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
