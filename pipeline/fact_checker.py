"""Fail-closed structural checks plus sentence-by-sentence semantic grounding."""
import re

from pipeline.story_model import array, prompt_json, require, text

ISSUE_TYPES = {"unsupported", "character_confusion", "event_contradiction", "wrong_causality"}


def sentences(narration):
    return [s.strip() for s in re.findall(r'[^。！？!?\n]+[。！？!?]*[”’」』"。！？!?]*', narration) if s.strip()]


class FactChecker:
    def __init__(self, store, model):
        self.store, self.model = store, model

    def check(self, bible, segment, fresh=False):
        parts = sentences(segment["text"])
        require(bool(parts), "旁白不能为空")
        events = {e["event_id"]: e for e in bible["events"]}
        event = events.get(segment["event_id"])
        panels = {p["source_panel_id"]: p for p in bible["panels"]}
        allowed_context=set()
        def visit(e):
            for cause in e["causes"]:
                if cause["event_id"] not in allowed_context:
                    allowed_context.add(cause["event_id"])
                    visit(events[cause["event_id"]])
        if event:
            visit(event)
        context_ids=segment.get("context_event_ids", [])
        expected_sources=set(event["source_panel_ids"]) if event else set()
        for eid in context_ids:
            if eid in events:
                expected_sources.update(events[eid]["source_panel_ids"])
        if not event or not segment.get("source_panel_ids") or not set(context_ids).issubset(allowed_context) or set(segment["source_panel_ids"]) != expected_sources:
            return {"status": "issues", "sentences": [
                {"sentence_index": i, "text": s, "verdict": "issues", "issues": [
                    {"type": "unsupported", "detail": "事件或来源关联无效"}]} for i, s in enumerate(parts)]}
        images = [str(self.store.evidence_path(panels[pid])) for pid in segment["source_panel_ids"]]
        causes = [events[eid] for eid in allowed_context]
        for cause in causes:
            for pid in cause["source_panel_ids"]:
                image = str(self.store.evidence_path(panels[pid]))
                if image not in images:
                    images.append(image)
        prompt = f"""Fact Checker v1。独立逐句审核漫画解说，图片为原始证据（当前事件在前，因果前提在后）。
不能因给了 event_id 就判通过；Story Bible 也可能识别错，要与原图对照。
逐句检查四类问题：unsupported 无来源依据、character_confusion 人物混淆、
event_contradiction 事件矛盾、wrong_causality 错误因果。
角色想法、对白说法与客观事实分开；先后出现不代表因果；疑问句的预设也必须有依据。
身份或细节看不清时 verdict=uncertain，绝不默认通过。
人物：{prompt_json(bible['characters'])}
原文已确认的关系：{prompt_json(bible.get('relationships', []))}
事件：{prompt_json(event)}
允许的因果前提：{prompt_json(causes)}
待核对句子：{prompt_json([{'sentence_index': i, 'text': s} for i,s in enumerate(parts)])}
返回 {{"sentences":[{{"sentence_index":0,"verdict":"supported|issues|uncertain",
"evidence":"具体画面细节/对白依据或不能确认的原因",
"issues":[{{"type":"unsupported|character_confusion|event_contradiction|wrong_causality","detail":"说明"}}]}}]}}。
必须覆盖每个句子且保持索引；supported 必须有证据且 issues 为空，issues 必须列出问题。"""
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
                                   "source_panel_ids": segment["source_panel_ids"]})
            status = "issues" if any(r["verdict"] == "issues" for r in normalized) else (
                "uncertain" if any(r["verdict"] == "uncertain" for r in normalized) else "passed")
            return {"status": status, "sentences": normalized}
        except Exception as exc:
            # The draft is useful for review, but a failed check can never approve it.
            return {"status": "unavailable", "error": str(exc), "sentences": [
                {"sentence_index": i, "text": s, "verdict": "uncertain", "issues": []}
                for i, s in enumerate(parts)]}
