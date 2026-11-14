"""Fail-closed structural checks plus sentence-by-sentence semantic grounding."""
import re

from pipeline.story_model import array, prompt_json, require, text, EvidenceImage
from pipeline.story_context import narrated_ids, context_ids as causal_context_ids, source_ids
from pipeline.semantic_grounding import make_grounding
from pipeline.character_identity import identity_context, premature_name_issues, identity_source_ids
from pipeline.chapter_memory import prompt_context
from pipeline.prompt_material import character_index

ISSUE_TYPES = {"unsupported", "character_confusion", "event_contradiction", "wrong_causality"}


def sentences(narration):
    return [s.strip() for s in re.findall(r'[^。！？!?\n]+[。！？!?]*[”’」』"。！？!?]*', narration) if s.strip()]


class FactChecker:
    def __init__(self, store, model):
        self.store, self.model = store, model

    def check(self, bible, segment, fresh=False, feedback=None):
        parts = sentences(segment["text"])
        require(bool(parts), "旁白不能为空")
        events = {e["event_id"]: e for e in bible["events"]}
        event = events.get(segment["event_id"])
        panels = {p["source_panel_id"]: p for p in bible["panels"]}
        ids = narrated_ids(segment)
        allowed_context = set(causal_context_ids(bible, ids)) if all(eid in events for eid in ids) else set()
        context_ids=segment.get("context_event_ids", [])
        expected_sources=set(source_ids(bible,ids)) if event and all(eid in events for eid in ids) else set()
        for eid in context_ids:
            if eid in events:
                expected_sources.update(events[eid]["source_panel_ids"])
        if not event or any(eid not in events for eid in ids) or not segment.get("source_panel_ids") or not set(context_ids).issubset(allowed_context) or set(segment["source_panel_ids"]) != expected_sources:
            return {"status": "issues", "sentences": [
                {"sentence_index": i, "text": s, "verdict": "issues", "issues": [
                    {"type": "unsupported", "detail": "事件或来源关联无效"}]} for i, s in enumerate(parts)]}
        early_names = premature_name_issues(bible, [segment])
        if early_names:
            return {'status':'issues', 'sentences':[
                {'sentence_index':i, 'text':s, 'verdict':'issues', 'issues':[
                    {'type':'character_confusion','detail':issue['detail']} for issue in early_names]}
                for i,s in enumerate(parts)]}
        primary_sources = source_ids(bible,ids)
        ordered_sources = primary_sources+[pid for pid in segment['source_panel_ids'] if pid not in primary_sources]
        causes = sorted((events[eid] for eid in allowed_context),key=lambda e:e['order'])
        image_panel_ids = list(ordered_sources)
        for cause in causes:
            for pid in cause["source_panel_ids"]:
                if pid not in image_panel_ids:
                    image_panel_ids.append(pid)
        for pid in identity_source_ids(bible,segment['text'],ids):
            if pid not in image_panel_ids:
                image_panel_ids.append(pid)
        current_panels = [p for p in bible['panels'] if not p.get('historical')]
        first = next((i for i,p in enumerate(current_panels) if p['source_panel_id'] in primary_sources),0)
        continuity = [p['source_panel_id'] for p in current_panels[max(0,first-2):first]]
        for pid in continuity:
            if pid not in image_panel_ids:
                image_panel_ids.append(pid)
        images = [EvidenceImage(self.store.evidence_path(panels[pid]),pid) for pid in image_panel_ids]
        prompt = f"""Fact Checker v2。独立逐句审核漫画解说，图片为原始证据（当前事件在前，因果前提在后）。
不能因给了 event_id 就判通过；Story Bible 也可能识别错，要与原图对照。
逐句检查四类问题：unsupported 无来源依据、character_confusion 人物混淆、
event_contradiction 事件矛盾、wrong_causality 错误因果。
角色想法、对白说法与客观事实分开；先后出现不代表因果；疑问句的预设也必须有依据。
身份或细节看不清时 verdict=uncertain，绝不默认通过。
人物：{prompt_json(character_index(bible['characters']))}
生成前资料：{prompt_json(prompt_context(bible))}
前话摘要不能替代原图证据；series_notes 是参考，不得凭它确认姓名/关系/因果。本话引用历史事实仍须对应所附历史原图。
identity_availability：{prompt_json(identity_context(bible,[segment]))}
原文已确认的关系：{prompt_json(bible.get('relationships', []))}
事件：{prompt_json([events[eid] for eid in ids])}
允许的因果前提：{prompt_json(causes)}
前文连续性参考图（仅核验，不增加本段事实/展示范围）：{prompt_json(continuity)}
参考图只协助核对同一交谈/动作的指代与媒介，不能把参考图事件纳入当前 units.event_ids 或 display_panel_ids。
逐条区分发出和收到的消息，不因都在一次交流就归为同一人；综合连续操作、指代与原图，不只凭气泡颜色。
“问”也可以是文字询问，不限开口发声；阅读顺序可支持交谈/动作先后，不能仅因缺少时间戳就否认顺序，也不能把先后升级为因果。
附图顺序的来源ID：{prompt_json(image_panel_ids)}
除逐句核验，还将每句拆成自然语义单元 units，所有 units.text 拼接必须逐字等于该句。
每个单元只围绕当前讲述的动作或事实；句内换到另一个事件时可按自然分句切开，不能逐字切碎。
event_ids 只列正文实际讲述的事实（包括正文提及的上下文事件），不能只写节点主事件。
background_event_ids 只列用于核验的背景，不要求观众重看。不得把全部递归因果前提当作正文。
display_panel_ids 只列观众听到这一单元时应看到且实际支持对应事实的来源ID，每个讲述事件都需展示依据。
外貌、姿态和构图只是识别证据，不要求正文朗读；中性称呼不应被强制改成外貌标签，不能凭猜测指认说话人。
姓名只有在身份对应得到原图/原文证据确认后才能使用；未知性别、关系、开枪者、受击者和死亡结果不能为连贯补齐。
逐字引文与人物归属须分别核对，单有人喊姓名不证明在叫谁；连续行动、明确称呼、对白与原图综合判断，不能只凭发色/衣服合并。
核对姓名、关系及身份的 reveal_order/reveal_panel_id：内部后文已知不代表本句时观众已知，同节点内也不能先用后揭晓。
中性称呼也要核对依据，不能暗示未经证实的职业、关系或行为身份；未知但稳定可辨的称呼无需补齐姓名。
不能把拒绝后收到消息解释为贪婪、善良或责任感，除非原作明确说明。display_panel_ids 还须避免提前展示后文才揭晓的信息。
待核对句子：{prompt_json([{'sentence_index': i, 'text': s} for i,s in enumerate(parts)])}
返回 {{"sentences":[{{"sentence_index":0,"verdict":"supported|issues|uncertain",
"evidence":"具体画面细节/对白依据或不能确认的原因",
"units":[{{"text":"当前句内连续原文","event_ids":["实际讲述事件ID"],"display_panel_ids":["来源ID"],"background_event_ids":[]}}],
"issues":[{{"type":"unsupported|character_confusion|event_contradiction|wrong_causality","detail":"说明"}}]}}]}}。
必须覆盖每个句子且保持索引；supported 必须有证据且 issues 为空，issues 必须列出问题。"""
        if feedback:
            prompt += '\n上一份核验响应未通过结构/来源校验：'+prompt_json(feedback)+'。重新看图，修正核验JSON与逐句关联；不能默认通过，也不要改写待核对原句。'
        try:
            raw = self.model.ask("check", prompt, images, fresh=fresh)
            rows = array(raw.get("sentences"), "sentences")
            require(len(rows) == len(parts), "事实核验漏句")
            normalized = []
            for i, row in enumerate(rows):
                require(isinstance(row, dict) and type(row.get("sentence_index")) is int and row["sentence_index"] == i,
                        "事实核验句子索引无效")
                require(row.get("verdict") in {"supported", "issues", "uncertain"}, "核验状态无效")
                basis = text(row.get("evidence"), "核验依据")
                issues = array(row.get("issues"), "issues")
                for issue in issues:
                    require(isinstance(issue, dict) and issue.get("type") in ISSUE_TYPES, "问题类型无效")
                    text(issue.get("detail"), "issue.detail")
                require(not (row["verdict"] == "supported" and issues) and
                        not (row["verdict"] == "issues" and not issues), "核验状态与问题不一致")
                normalized.append({**row, "text": parts[i], "evidence": basis,
                                   "verification_panel_ids": image_panel_ids,
                                   "source_panel_ids": list(dict.fromkeys(pid for u in row.get('units', [])
                                                                         for pid in u.get('display_panel_ids', [])))})
            status = "issues" if any(r["verdict"] == "issues" for r in normalized) else (
                "uncertain" if any(r["verdict"] == "uncertain" for r in normalized) else "passed")
            grounding = make_grounding(bible, segment, normalized) if status == 'passed' else {
                'status': 'needs_review', 'units': [], 'issues': ['事实核验尚未通过']}
            return {"status": status, "sentences": normalized, "grounding": grounding}
        except Exception as exc:
            # The draft is useful for review, but a failed check can never approve it.
            return {"status": "unavailable", "error": str(exc), "sentences": [
                {"sentence_index": i, "text": s, "verdict": "uncertain", "issues": []}
                for i, s in enumerate(parts)]}
