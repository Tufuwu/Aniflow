"""Panel-grounded extraction. IDs and source links are assigned by code."""
from copy import deepcopy
import re

from pipeline.story_bible import StoryError, digest
from pipeline.story_model import array, prompt_json, require, text

ANALYSIS_VERSION = 2


class StoryAnalyzer:
    def __init__(self, store, model):
        self.store, self.model = store, model

    def analyze(self, panels, progress=None):
        require(bool(panels), "请先载入第一话并确认分镜")
        evidence, characters, events = [], {}, []
        ids = set()
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
            # Full causal history is retained; don't truncate character identity context.
            context = {"characters": [{k: c[k] for k in ("character_id", "name", "description", "aliases")}
                                      for c in characters.values()],
                       "previous_events": [{k: e[k] for k in ("event_id", "summary", "character_ids", "causes")}
                                           for e in events],
                       "ocr_hint_unverified": entry["extracted_text"]}
            prompt = f"""Story Analyzer v{ANALYSIS_VERSION}。按阅读顺序分析当前唯一分镜，图片是首要依据。
已有背景（仅供身份/连续性核对）：{prompt_json(context)}
本来源区域可能含多个相邻分镜；如与上个区域有重叠，同一事件不要重复提取。裁切边界不确定：{entry['boundary_uncertain']}。
返回 {{"characters":[{{"character_id":"C001","name":"原文姓名或未知人物1",
"description":"外貌特征","aliases":[],"identity_evidence":"当前画面中的身份依据"}}],
"dialogue":[{{"speaker_id":null,"text":"逐字保留原文，含繁体中文"}}],
"panel_type":"story|cover|blank|advertisement",
"events":[{{"summary":"中文客观事件","character_ids":["C001"],"actions":["人物做了什么"],
"importance":3,"role":"setup|conflict|climax|resolution",
"evidence":[{{"kind":"visual|dialogue","detail":"图中可核对的具体细节或原文引句"}}],
"causes":[{{"event_id":"E00001","evidence":"明确因果的画面或对白依据"}}]}}],"uncertainties":[]}}。
沿用已有角色 ID；新角色使用未占用的 C 编号，不能凭发色相似就合并身份。
只列当前画面出现的人物；未知说话人用 null。不得用目录名或常识补剧情。
封面、广告、人物拼贴不属于剧情事件，events 必须为空；不能漏掉真正剧情中的无对白动作。importance 为1到5。
causes 仅能指向前文已存在事件；时间先后不等于因果，无明确依据时留空。
每个事件必须有具体证据。不要自行生成 event_id 或 source_panel_ids。"""
            retry_prompt = prompt
            for attempt in range(3):
                try:
                    raw = self.model.ask("analyze", retry_prompt, [str(self.store.evidence_path(entry))])
                    entry, new_characters, new_events = self._consume(
                        raw, deepcopy(entry), deepcopy(characters), deepcopy(events), pid)
                    characters, events = new_characters, new_events
                    break
                except StoryError as exc:
                    if attempt == 2:
                        raise
                    retry_prompt = prompt + f"\n上次返回的结构未通过校验：{exc}。请修正后重新返回完整 JSON。"
            evidence.append(entry)
            if progress:
                progress(index + 1, len(panels), f"已分析分镜 {pid}")
        require(bool(events), "本话没有提取到剧情事件，请检查分镜和视觉模型")
        bible = {"schema_version": 1, "analysis_version": ANALYSIS_VERSION,
                 "panels": evidence, "characters": list(characters.values()), "events": events}
        bible = self.review_boundaries(bible)
        bible = self.reconcile_identity(bible)
        bible["bible_id"] = digest(bible)
        return bible

    def reconcile_identity(self, bible):
        candidates=[p for p in bible['panels'] if re.search(
            r'爸|妈|父|母|哥哥|姐姐|用户.{1,12}触发', ' '.join(d['text'] for d in p['dialogue']))]
        if not candidates:
            bible['relationships']=[]
            return bible
        prompt=f"""Character Identity Review v1。整话已经读完，只确认原文明确出现的人名和亲属关系。
人物档案：{prompt_json(bible['characters'])}
事件：{prompt_json([{'summary':e['summary'],'character_ids':e['character_ids'],'source_panel_ids':e['source_panel_ids']} for e in bible['events']])}
附图及原文：{prompt_json([{'source_panel_id':p['source_panel_id'],'dialogue':p['dialogue']} for p in candidates])}
图片顺序与附图列表一致。不能凭衣服或外貌推定亲属；姓名须在原文逐字出现且能明确归于角色。
返回 {{"names":[{{"character_id":"C001","name":"确切姓名","source_panel_id":"1","quote":"逐字原文"}}],
"relationships":[{{"from_id":"C001","to_id":"C002","relation":"母亲/父亲/哥哥等","source_panel_id":"1","quote":"逐字原文"}}]}}。
未确认的身份保持未知。from_id 是被描述关系的主体，to_id 是该主体的亲属。"""
        raw=self.model.ask('identity',prompt,[str(self.store.evidence_path(p)) for p in candidates])
        by_id={c['character_id']:c for c in bible['characters']}
        sources={p['source_panel_id']:p for p in candidates}
        def verify(item):
            require(isinstance(item,dict) and item.get('source_panel_id') in sources,'身份核对来源无效')
            quote=text(item.get('quote'),'身份证据原文')
            require(any(quote in d['text'] for d in sources[item['source_panel_id']]['dialogue']),'身份依据必须逐字来自原文')
            return quote
        for item in array(raw.get('names'),'names'):
            quote=verify(item)
            require(item.get('character_id') in by_id,'身份证据人物无效')
            name=text(item.get('name'),'确认姓名')
            require(name in quote,'姓名没有出现在身份证据中')
            character=by_id[item['character_id']]
            if character['name'] != name:
                require(character['name'].startswith('未知'),'不能覆盖已经确认的人名')
                character['aliases']=list(dict.fromkeys(character['aliases']+[character['name']]))
                character['name']=name
                character['identity_evidence'].append({'source_panel_id':item['source_panel_id'],'detail':quote,'kind':'confirmed_name'})
        relationships=[]
        for item in array(raw.get('relationships'),'relationships'):
            verify(item)
            require(item.get('from_id') in by_id and item.get('to_id') in by_id and item['from_id']!=item['to_id'],'关系人物无效')
            text(item.get('relation'),'relation')
            relationships.append(item)
        bible['relationships']=relationships
        bible['identity_review']=raw
        return bible

    def review_boundaries(self, bible):
        # End-card quizzes can look like in-story system questions in isolation.
        # Recheck the opening/ending together after reading the complete chapter.
        candidates = (bible["panels"][:3] + bible["panels"][-5:])
        candidates = list({p["source_panel_id"]:p for p in candidates}.values())
        ids = {p["source_panel_id"] for p in candidates}
        prompt = f"""Chapter Boundary Review v2。已完成整话分析，现在只核对封面、片尾和广告边界。
按顺序附上以下来源图：{prompt_json([p['source_panel_id'] for p in candidates])}。
本话事件摘要：{prompt_json([{'event_id':e['event_id'],'summary':e['summary'],'source_panel_ids':e['source_panel_ids']} for e in bible['events']])}
区别剧情内系统界面/答题与作者在片尾让读者点赞关注评论的互动题；后者不属于剧情结尾。
只排除有明确画面证据的非剧情内容，不确定时不要排除。不得补造或修改剧情。
返回 {{"panels":[{{"source_panel_id":"图号","classification":"story|non_story","reason":"具体画面依据"}}]}}。
覆盖每张附图且不重复。剧情内画面 classification=story，即使是夸张反应或系统问答也不能排除。
只有明确的封面、制作信息、更新预告、作者读者互动才分类 non_story。只能列附图 ID。"""
        raw = self.model.ask("reconcile", prompt, [str(self.store.evidence_path(p)) for p in candidates])
        excluded = {}
        seen=set()
        for item in array(raw.get("panels"), "panels"):
            require(isinstance(item, dict) and item.get("source_panel_id") in ids, "片尾核对来源无效")
            pid=item["source_panel_id"]
            require(pid not in seen and item.get("classification") in {"story", "non_story"}, "片尾分类无效")
            seen.add(pid)
            reason=text(item.get("reason"), "分类依据")
            if item["classification"] == "non_story":
                excluded[pid]=reason
        require(seen == ids, "片尾核对漏图")
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
    def _consume(raw, entry, characters, events, pid):
        require(isinstance(raw, dict), "分镜分析结果无效")
        panel_type = raw.get("panel_type", "story")
        visible_characters = array(raw.get("characters"), "characters")
        # A cover collage is not evidence that these people entered the story.
        for char in visible_characters if panel_type == "story" else []:
            require(isinstance(char, dict), "人物格式无效")
            cid = text(char.get("character_id"), "character_id")
            name = text(char.get("name"), "name")
            basis = text(char.get("identity_evidence"), "identity_evidence")
            description = text(char.get("description"), "description")
            aliases = [text(a, "alias") for a in array(char.get("aliases", []), "aliases")]
            if cid in characters:
                old = characters[cid]["name"]
                if old != name:
                    require(old.startswith("未知") and name in basis,
                            "已有角色姓名变化，请先人工确认身份")
                    characters[cid]["name"] = name
                    characters[cid]["aliases"] = list(dict.fromkeys(characters[cid]["aliases"] + [old] + aliases))
                characters[cid]["source_panel_ids"].append(pid)
                characters[cid]["identity_evidence"].append({"source_panel_id": pid, "detail": basis})
            else:
                characters[cid] = {"character_id": cid, "name": name, "description": description,
                                   "aliases": aliases, "source_panel_ids": [pid],
                                   "identity_evidence": [{"source_panel_id": pid, "detail": basis}]}
        dialogue = array(raw.get("dialogue"), "dialogue")
        for line in dialogue:
            require(isinstance(line, dict), "对白格式无效")
            text(line.get("text"), "dialogue.text")
            require(panel_type != "story" or line.get("speaker_id") is None or line["speaker_id"] in characters, "对白人物不存在")
        entry["dialogue"] = dialogue
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
                require(isinstance(item, dict) and item.get("kind") in {"visual", "dialogue"}, "证据类型无效")
                text(item.get("detail"), "evidence.detail")
                item["source_panel_id"] = pid
                if item["kind"] == "dialogue":
                    require(any(item["detail"].strip('“”「」『』"') in line["text"] for line in dialogue), "对白证据必须逐字来自原文")
            for cause in array(event.get("causes"), "causes"):
                require(isinstance(cause, dict) and cause.get("event_id") in {e["event_id"] for e in events},
                        "因果引用不存在或指向未来事件")
                text(cause.get("evidence"), "cause.evidence")
                cause["source_panel_id"] = pid
            event.update(event_id=f"E{len(events) + 1:05d}", source_panel_ids=[pid],
                         order=len(events), dialogue=dialogue)
            events.append(event)
        return entry, characters, events
