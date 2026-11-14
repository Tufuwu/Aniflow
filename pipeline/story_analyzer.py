"""Panel-grounded extraction. IDs and source links are assigned by code."""
from copy import deepcopy

from pipeline.story_bible import StoryError, digest
from pipeline.story_model import array, prompt_json, require, text, EvidenceImage, validated_result
from pipeline.character_identity import confirm_claim, record_name
from pipeline.prompt_material import character_index
from pipeline.chapter_memory import prompt_context

ANALYSIS_VERSION = 6


class StoryAnalyzer:
    def __init__(self, store, model):
        self.store, self.model = store, model

    def analyze(self, panels, progress=None, history=None):
        require(bool(panels), "请先载入本话并确认分镜")
        history = deepcopy(history or {})
        evidence, events = history.pop('panels', []), history.pop('events', [])
        characters = {c['character_id']:c for c in history.pop('characters', [])}
        relationships = history.pop('relationships', [])
        ids = {p['source_panel_id'] for p in evidence}
        for index, panel in enumerate(panels):
            pid = str(panel.get("source_panel_id", panel.get("panel_id", index + 1)))
            require(pid not in ids, "分镜 ID 重复")
            ids.add(pid)
            # Never analyze an inpainted image if the pre-cleaning crop is present.
            source = panel.get("raw_path") or panel.get("path")
            require(bool(source), "分镜缺少原图路径")
            entry = {"source_panel_id": pid, "order": index,
                     **self.store.snapshot(source),
                     "source_page": panel.get("source_page"),
                     "bbox": {k: panel[k] for k in ("x", "y", "width", "height") if k in panel},
                     "extracted_text": panel.get("text", "")}
            entry["boundary_uncertain"] = panel.get("boundary_uncertain", False)
            if panel.get('derivation'):
                entry['derivation'] = deepcopy(panel['derivation'])
            self.store.evidence_path(entry)
            # Full causal history is retained; don't truncate character identity context.
            context = {"history_available": bool(characters), "characters": character_index(list(characters.values())),
                       'prior_context':prompt_context({'prior_context':history}),
                       'identity_reference_panel_ids':[p['source_panel_id'] for p in evidence[-1:]],
                       "previous_events": [{k: e[k] for k in ("event_id", "summary", "character_ids", "causes")}
                                           for e in events if not e.get("historical")],
                       "ocr_hint_unverified": entry["extracted_text"]}
            prompt = f"""Story Analyzer v{ANALYSIS_VERSION}。按阅读顺序分析当前来源 {pid}，图片是首要依据。
首图是当前来源，后附上一来源原图仅供人物连续性核对，不能从参考图重复提取事件。
已有背景（仅供身份/连续性核对）：{prompt_json(context)}
prior_context 是生成前资料：上一话已核对摘要和历史设定可用于连续性；series_notes 是用户提供的参考线索，仍须与原图核对，不能直接确认身份。
历史事件和原图有 H 前缀及 historical 标记，只用于核验和有明确依据的因果前提；不重复生成上一话事件，不把历史当作本话新发生。
本来源区域可能含多个相邻分镜；如与上个区域有重叠，同一事件不要重复提取。裁切边界不确定：{entry['boundary_uncertain']}。
返回 {{"analyzed_source_panel_id":"{pid}","characters":[{{"character_id":"C001","name":null,
"description":"可确认的识别特征；无法确认可写未知","aliases":[],"identity_evidence":"当前画面中的识别依据",
"identity_match":null,"name_evidence":null,"gender":null,
"narrative_label":null,"label_evidence":null}}],
"dialogue":[{{"speaker_id":null,"text":"逐字保留原文，含繁体中文"}}],
"visible_text":["胸牌/说明文字等逐字原文"],
"identity_clues":[{{"kind":"name|address|relationship|correspondence","quote":"逐字原文；纯行动对应可为空",
"character_ids":["C001"],"needs_review":true,"detail":"姓名提及/称呼变化/人物对应中待核对之处"}}],
"panel_type":"story|cover|blank|advertisement","scene":"可见场景位置；无法确认则写未知",
"events":[{{"summary":"中文客观事件","character_ids":["C001"],"actions":["人物做了什么"],
"importance":3,"role":"setup|conflict|climax|resolution",
"evidence":[{{"kind":"visual|dialogue|caption","detail":"图中可核对的具体细节或原文引句"}}],
"causes":[{{"event_id":"E00001","evidence":"明确因果的画面或对白依据"}}]}}],"uncertainties":[]}}。
沿用已有角色 ID；新角色使用未占用的 C 编号，不能凭发色相似就合并身份。
历史资料为空时从零建立临时记录，之后出现的新人物也使用同一流程，不要求先补姓名、性别或关系。
已有角色再次出现须附 identity_match={{"method":"continuous_action|explicit_address|dialogue_context|original_image_comparison",
"basis":"连续行动、明确称呼、对白或原图对应的具体依据，不能只有发色/衣服相似"}}。
对应不确定就新建临时ID，并在 identity_clues 标记待核对，不强行合并。
original_image_comparison 只能用于实际附上的参考原图；更早未附图的人物对应若有疑点，留待整话原图核对。
姓名即使出现在对白中，也不等于对应此人。首次及后续确认都须 name_evidence={{"source_panel_id":"{pid}",
"kind":"dialogue|visual_text","name_kind":"primary|alias","quote":"含姓名的逐字原文","attribution":{{"character_id":"C001",
"method":"self_introduction|direct_address|name_tag|explicit_caption|dialogue_context",
"basis":"独立说明原文为何指向此人，不能重复姓名出现这一事实","source_panel_ids":["{pid}"]}}}}。
缺少姓名归属依据时 name 留 null，候选姓名留在 identity_clues。姓名和人物对应的疑点不限亲属称呼。
明确昵称或称呼变化也须逐字引文及独立归属证据，以 name_kind=alias 保存，不能覆盖已确认姓名；aliases 数组不能代替证据。
必要时建议稳定 narrative_label，label_evidence 说明有据的称呼依据；中性称呼也不能补职业、关系或行为身份。
未确认姓名/昵称不能通过 narrative_label 绕过归属核对。
不要把后文揭晓的姓名/关系提前当成本图已知信息；未知状态只留内部资料。
只列当前画面出现的人物；未知说话人用 null。不得用目录名或常识补剧情。
封面、广告、人物拼贴不属于剧情事件，events 必须为空；不能漏掉真正剧情中的无对白动作。importance 为1到5。
causes 仅能指向前文已存在事件；时间先后不等于因果，无明确依据时留空。
每个事件必须有具体证据。不要自行生成 event_id 或 source_panel_ids。"""
            prompt += '\n只提取标为当前来源的图。analyzed_source_panel_id 必须等于当前来源编号；参考图和参考资料里的文字不能抄进当前图。画外说明逐字放 visible_text，证据用 caption；人物发言逐字放 dialogue。'
            def extract():
                retry_prompt = prompt
                for attempt in range(3):
                    try:
                        images = [str(self.store.evidence_path(p)) for p in [entry]+evidence[-1:]]
                        raw = self.model.ask("analyze", retry_prompt, images, fresh=attempt > 0)
                        require(isinstance(raw,dict) and 'identity_clues' in raw, '分镜分析缺少身份线索核对项；无疑点也须返回空数组')
                        parsed, chars, extracted = self._consume(raw,deepcopy(entry),deepcopy(characters),deepcopy(events),pid,evidence)
                        return {'entry':parsed,'characters':chars,'events':extracted}
                    except StoryError as exc:
                        if attempt == 2: raise
                        retry_prompt = prompt + f"\n上次返回的结构未通过校验：{exc}。请修正后重新返回完整 JSON。"
            result = validated_result(self.store,self.model,'analyze',
                {'prompt':prompt,'entry':entry,'prior_evidence':evidence,'characters':characters,
                 'events':events,'history':history},extract)
            entry, characters, events = result['entry'], result['characters'], result['events']
            evidence.append(entry)
            if progress:
                progress(index + 1, len(panels), f"已分析分镜 {pid}")
        require(any(not e.get('historical') for e in events), "本话没有提取到剧情事件，请检查分镜和视觉模型")
        bible = {"schema_version": 1, "analysis_version": ANALYSIS_VERSION,
                 "panels": evidence, "characters": list(characters.values()), "events": events,
                 'relationships':relationships}
        if history:
            bible['prior_context'] = history
        bible = validated_result(self.store,self.model,'reconcile',
            {'version':ANALYSIS_VERSION,'bible':bible},lambda:self.review_boundaries(deepcopy(bible)))
        try:
            bible = validated_result(self.store,self.model,'identity',
                {'version':ANALYSIS_VERSION,'bible':bible},lambda:self.reconcile_identity(deepcopy(bible)))
        except Exception as exc:
            # Invalid or unavailable attribution must not erase the extracted
            # chapter or partially commit a batch of newly confirmed names.
            bible['identity_review'] = {'status':'needs_review','error':str(exc)}
            bible.setdefault('relationships', [])
        bible["bible_id"] = digest(bible)
        return bible

    def reconcile_identity(self, bible):
        current_ids = {p['source_panel_id'] for p in bible['panels'] if not p.get('historical')}
        candidate_ids = {p['source_panel_id'] for p in bible['panels'] if not p.get('historical')
                         and any(c.get('needs_review') for c in p.get('identity_clues', []))}
        candidate_ids.update(p['source_panel_id'] for c in bible['characters']
                             for p in c.get('pending_names', []) if p['source_panel_id'] in current_ids)
        candidates = [p for p in bible['panels'] if p['source_panel_id'] in candidate_ids]
        if not candidates:
            bible.setdefault('relationships', [])
            return bible
        # Include original appearances for correspondence checks, not every page
        # merely because this is a new chapter.
        involved = {cid for p in candidates for clue in p.get('identity_clues', [])
                    for cid in clue.get('character_ids', [])}
        involved.update(c['character_id'] for c in bible['characters'] if c.get('pending_names'))
        related = {pid for c in bible['characters'] if c['character_id'] in involved
                   for pid in c['source_panel_ids']}
        candidates = [p for p in bible['panels'] if p['source_panel_id'] in candidate_ids | related]
        prompt=f"""Character Identity Review v2。整话已经读完，核对普通姓名、称呼变化、人物对应及关系。
人物档案：{prompt_json(bible['characters'])}
事件：{prompt_json([{'summary':e['summary'],'character_ids':e['character_ids'],'source_panel_ids':e['source_panel_ids']} for e in bible['events']])}
附图及原文：{prompt_json([{k:p.get(k) for k in ('source_panel_id','dialogue','visible_text','identity_clues')} for p in candidates])}
图片顺序与附图列表一致。综合连续行动、明确称呼、对白及原图，不能只凭发色或衣服相似合并人物。
引文逐字正确与归属正确是两个条件；有人喊姓名不等于已知在叫谁。没有归属依据时保持未确认。
返回 {{"names":[{{"character_id":"C001","name":"确切姓名或昵称","name_kind":"primary|alias","source_panel_id":"1","kind":"dialogue|visual_text",
"quote":"逐字原文","attribution":{{"character_id":"C001","method":"self_introduction|direct_address|name_tag|explicit_caption|dialogue_context",
"basis":"为何确切对应此人；不能只重复姓名出现","source_panel_ids":["1"]}}}}],
"relationships":[{{"from_id":"C001","to_id":"C002","relation":"原文明确称呼","source_panel_id":"1","quote":"逐字原文",
"attribution":{{"character_id":"C001","method":"dialogue_context","basis":"为何这段称呼确切连接这两人",
"source_panel_ids":["1"]}}}}],"unresolved":[{{"detail":"未确认的对应及缺少的依据"}}]}}。
不得用新姓名偷偷合并临时ID；不能确认的对应留在 unresolved，仍使用原ID。
确有依据的昵称以 name_kind=alias 追加，保留已有正式姓名；昵称本身及其人物归属也须分别成立。
source_panel_ids 须包括建立人物对应所需的所有原图，尤其是后文才确认对应的图，不能只填较早提到姓名的图。
程序按最后一张必要证据记录揭晓位置。内部读完整章后知道的姓名或关系不能提前给开头使用。
from_id 是被描述关系的主体，to_id 是关系另一端，不能把方向写反。"""
        raw=self.model.ask('identity',prompt,[EvidenceImage(self.store.evidence_path(p),p['source_panel_id']) for p in candidates])
        by_id={c['character_id']:c for c in bible['characters']}
        sources={p['source_panel_id']:p for p in candidates}
        for item in array(raw.get('names'),'names'):
            require(isinstance(item,dict), '身份证据格式无效')
            require(item.get('character_id') in by_id,'身份证据人物无效')
            name=text(item.get('name'),'确认姓名')
            claim=confirm_claim(item,sources,item['character_id'],name)
            character=by_id[item['character_id']]
            record_name(character,claim)
            character['identity_evidence'].append({**claim,'detail':claim['quote'],'kind':'confirmed_name'})
        relationships=deepcopy(bible.get('relationships', []))
        for item in array(raw.get('relationships'),'relationships'):
            require(isinstance(item,dict), '关系格式无效')
            require(item.get('from_id') in by_id and item.get('to_id') in by_id and item['from_id']!=item['to_id'],'关系人物无效')
            relation=text(item.get('relation'),'relation')
            claim=confirm_claim(item,sources,item['from_id'],relation)
            relationships.append(claim)
        bible['relationships']=relationships
        bible['identity_review']={**raw, 'status':'needs_review' if raw.get('unresolved') else 'reviewed'}
        return bible

    def review_boundaries(self, bible):
        # End-card quizzes can look like in-story system questions in isolation.
        # Recheck the opening/ending together after reading the complete chapter.
        current = [p for p in bible['panels'] if not p.get('historical')]
        candidates = current[:3]+current[-5:]
        candidates = list({p["source_panel_id"]:p for p in candidates}.values())
        ids = {p["source_panel_id"] for p in candidates}
        prompt = f"""Chapter Boundary Review v3。已完成整话分析，现在只核对封面、片尾和广告边界。
按顺序附上以下来源图：{prompt_json([p['source_panel_id'] for p in candidates])}。
本话事件摘要：{prompt_json([{'event_id':e['event_id'],'summary':e['summary'],'source_panel_ids':e['source_panel_ids']} for e in bible['events']])}
区别剧情内系统界面/答题与作者在片尾让读者点赞关注评论的互动题；后者不属于剧情结尾。
只排除有明确画面证据的非剧情内容，不确定时不要排除。不得补造或修改剧情。
返回 {{"panels":[{{"source_panel_id":"图号","classification":"story|non_story","reason":"具体画面依据"}}]}}。
覆盖每张附图且不重复。剧情内画面 classification=story，即使是夸张反应或系统问答也不能排除。
只有明确的封面、制作信息、更新预告、作者读者互动才分类 non_story。只能列附图 ID。"""
        feedback = ''
        for attempt in range(2):
            raw = self.model.ask("reconcile", prompt+feedback,
                [EvidenceImage(self.store.evidence_path(p),p['source_panel_id']) for p in candidates],fresh=attempt>0)
            excluded,seen = {},set()
            try:
                for item in array(raw.get("panels"), "panels"):
                    require(isinstance(item, dict) and item.get("source_panel_id") in ids, "片尾核对来源无效")
                    pid=item["source_panel_id"]
                    require(pid not in seen and item.get("classification") in {"story", "non_story"}, "片尾分类无效")
                    seen.add(pid)
                    reason=text(item.get("reason"), "分类依据")
                    if item["classification"] == "non_story":
                        excluded[pid]=reason
                require(seen == ids, "片尾核对漏图")
                break
            except StoryError as exc:
                if attempt==1:raise
                feedback='\n上次结构核对失败：'+str(exc)+'。只返回这些真实来源ID：'+prompt_json([p['source_panel_id'] for p in candidates])+'。不要返回1到附图数量的序号。'
        for event in bible["events"]:
            if all(pid in ids for pid in event["source_panel_ids"]):
                event.pop("narrative_eligible",None)
                event.pop("exclusion_reason",None)
            if any(pid in excluded for pid in event["source_panel_ids"]):
                event["narrative_eligible"] = False
                event["exclusion_reason"] = ";".join(excluded[pid] for pid in event["source_panel_ids"] if pid in excluded)
        bible["boundary_review"] = raw
        return bible

    @staticmethod
    def _consume(raw, entry, characters, events, pid, previous_panels=None):
        require(isinstance(raw, dict), "分镜分析结果无效")
        require(raw.get('analyzed_source_panel_id',pid)==pid, '分析对象混入参考图：来源编号不等于当前图')
        panel_type = raw.get("panel_type", "story")
        entry.setdefault('order', len(previous_panels or []))
        entry['source_panel_id'] = pid
        dialogue = array(raw.get('dialogue'), 'dialogue')
        for line in dialogue:
            require(isinstance(line, dict), '对白格式无效')
            text(line.get('text'), 'dialogue.text')
        entry['dialogue'] = dialogue
        entry['visible_text'] = [text(t, 'visible_text') for t in array(raw.get('visible_text', []), 'visible_text')]
        clues = array(raw.get('identity_clues', []), 'identity_clues')
        for clue in clues:
            require(isinstance(clue, dict) and clue.get('kind') in {'name','address','relationship','correspondence'}
                    and type(clue.get('needs_review')) is bool, '身份线索格式无效')
            text(clue.get('detail'), '身份线索依据')
            quote = clue.get('quote', '')
            require(isinstance(quote,str) and (not quote or any(quote in t for t in
                    [d['text'] for d in dialogue]+entry['visible_text'])), '身份线索必须逐字来自原文')
        entry['identity_clues'] = clues
        visible_characters = array(raw.get("characters"), "characters")
        # A cover collage is not evidence that these people entered the story.
        for char in visible_characters if panel_type == "story" else []:
            require(isinstance(char, dict), "人物格式无效")
            cid = text(char.get("character_id"), "character_id")
            name = char.get('name')
            require(name is None or isinstance(name,str), '姓名格式无效')
            proposed = name if name and not name.startswith('未知') else None
            basis = text(char.get("identity_evidence"), "identity_evidence")
            description = text(char.get('description') or '未知', 'description')
            if cid in characters:
                match = char.get('identity_match')
                require(isinstance(match,dict) and match.get('method') in {
                    'continuous_action','explicit_address','dialogue_context','original_image_comparison'},
                    '复用人物编号需要连续行动、称呼或原图对应依据；不确定应新建临时人物')
                text(match.get('basis'), '人物对应依据')
                characters[cid]["source_panel_ids"].append(pid)
                characters[cid]["identity_evidence"].append({"source_panel_id": pid, "detail": basis,
                                                           'identity_match':deepcopy(match),
                                                           'reveal_order':entry['order']})
            else:
                characters[cid] = {"character_id": cid, "name": '未知人物'+cid, 'name_status':'unknown',
                                   'gender':None, 'name_claims':[], 'label_claims':[], 'pending_names':[],
                                   "description": description, "aliases": [], "source_panel_ids": [pid],
                                   "identity_evidence": [{"source_panel_id": pid, "detail": basis}]}
            character = characters[cid]
            if proposed:
                proof = char.get('name_evidence')
                if proof:
                    claim = confirm_claim({**proof,'name':proposed},
                        {p['source_panel_id']:p for p in (previous_panels or [])+[entry]},cid,proposed)
                    record_name(character,claim)
                elif character['name_status']=='unknown' or character['name']!=proposed:
                    character['pending_names'].append({'name':proposed,'source_panel_id':pid,
                                                       'reason':'姓名提及尚无独立归属证据'})
            label = char.get('narrative_label')
            if label:
                text(label,'叙述称呼')
                require(label != cid and not label.startswith('未知人物'), '内部编号或未知状态不能作为叙述称呼')
                if label == proposed or any(n['name']==label for n in character['pending_names']):
                    require(any(n['name']==label for n in character['name_claims']), '叙述称呼不能绕过姓名归属核对')
                label_basis = text(char.get('label_evidence'),'叙述称呼依据')
                if not character['label_claims'] or character['label_claims'][-1]['label']!=label:
                    character['label_claims'].append({'label':label,'basis':label_basis,
                        'source_panel_id':pid,'reveal_panel_id':pid,'reveal_order':entry['order']})
        require(all(isinstance(clue.get('character_ids'),list) and
                    all(cid in characters for cid in clue['character_ids']) for clue in clues), '身份线索人物无效')
        dialogue = array(raw.get("dialogue"), "dialogue")
        for line in dialogue:
            require(isinstance(line, dict), "对白格式无效")
            text(line.get("text"), "dialogue.text")
            require(panel_type != "story" or line.get("speaker_id") is None or line["speaker_id"] in characters, "对白人物不存在")
        entry["dialogue"] = dialogue
        scene = raw.get("scene", "")
        require(isinstance(scene, str) and len(scene) <= 300, "场景描述无效")
        entry["scene"] = scene
        entry["panel_type"] = raw.get("panel_type", "story")
        require(entry["panel_type"] in {"story", "cover", "blank", "advertisement"}, "分镜类型无效")
        if entry["panel_type"] != "story":
            require(raw.get("events") == [], "封面/空白/广告不应生成剧情事件")
        entry["uncertainties"] = array(raw.get("uncertainties", []), "uncertainties")
        for event in array(raw.get("events"), "events"):
            require(isinstance(event, dict), "事件格式无效")
            event = deepcopy(event)
            event["summary"] = text(event.get("summary"), "event.summary")
            refs = array(event.get("character_ids"), "character_ids")
            require(all(isinstance(c, str) and c in characters for c in refs), "事件人物不存在")
            actions = array(event.get("actions"), "actions")
            for action in actions:
                text(action, "action")
            require(type(event.get("importance")) is int and 1 <= event["importance"] <= 5, "重要性应为1到5")
            require(event.get("role") in {"setup", "conflict", "climax", "resolution"}, "事件角色无效")
            proof = array(event.get("evidence"), "evidence")
            require(bool(proof), "事件缺少证据")
            for item in proof:
                require(isinstance(item, dict) and item.get("kind") in {"visual", "dialogue", 'caption'}, "证据类型无效")
                text(item.get("detail"), "evidence.detail")
                require(item.get('source_panel_id',pid)==pid, '事件证据混入参考图来源')
                item["source_panel_id"] = pid
                if item['kind'] in {'dialogue','caption'}:
                    quote=item['detail'].strip('“”「」『』"')
                    spoken=any(quote in line['text'] for line in dialogue)
                    caption=any(quote in line for line in entry['visible_text'])
                    if item['kind']=='dialogue' and not spoken and caption:
                        item.update(kind='caption',original_kind='dialogue')
                    require(caption if item['kind']=='caption' else spoken,
                            f"对白证据必须逐字来自原文：引用 {prompt_json(item['detail'])}，"
                            f"本区域原文为 {prompt_json([line['text'] for line in dialogue])}。"
                            "evidence.detail 只填写原文片段，不附加解释；画面说明应使用 visual")
            for cause in array(event.get("causes"), "causes"):
                require(isinstance(cause, dict) and cause.get("event_id") in {e["event_id"] for e in events},
                        "因果引用不存在或指向未来事件")
                text(cause.get("evidence"), "cause.evidence")
                cause["source_panel_id"] = pid
            local_index = sum(not e.get('historical') for e in events)
            event.update(event_id=f"E{local_index + 1:05d}", source_panel_ids=[pid],
                         order=local_index, dialogue=dialogue)
            events.append(event)
        return entry, characters, events
