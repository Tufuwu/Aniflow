"""Chinese Douyin narration constrained to a director-selected event."""
from pipeline.story_model import prompt_json, require, text
from pipeline.narrative_director import STYLES

STYLE_RULES = {
    "faithful": "按原作顺序，清楚交代人物和动作，不替角色编内心。",
    "payoff": "突出原作已有的压迫、反击、反差和结果；没有反转就不制造反转。短句有力。",
    "suspense": "用本话已知危险或尚未解答的问题吸引读者，延迟解释但不改变因果，不虚构谜团。",
}


class ScriptWriter:
    def __init__(self, model):
        self.model = model

    def write_segment(self, bible, plan, beat, previous="", fresh=False, feedback=None):
        event = next(e for e in bible["events"] if e["event_id"] == beat["event_id"])
        context = [e for e in bible["events"] if e["event_id"] in beat.get("context_event_ids", [])]
        panels = [p for p in bible["panels"] if p["source_panel_id"] in beat["source_panel_ids"]]
        prompt = f"""Script Writer v1。生成适合抖音口播的中文解说，风格：{STYLES[plan['style']]}。
{STYLE_RULES[plan['style']]} 当前段落职能：{beat['section']}；目标约{beat['target_chars']}字，最多100字。
上段已说：{previous}
人物档案：{prompt_json(bible['characters'])}
原文已确认的人物关系：{prompt_json(bible.get('relationships', []))}
唯一允许叙述的事件：{prompt_json(event)}
该事件明确的因果前提（仅作必要补充，不重复上段）：{prompt_json(context)}
对应原文证据：{prompt_json([{k: p[k] for k in ('source_panel_id', 'dialogue', 'uncertainties')} for p in panels])}
只写1到3个短句，使用中文标点。开场直接进入已发生的异常或冲突，结尾仅保留本话已有悬念。
不得编造身份、亲属、动机、数值、能力、未来、动作结果；不要使用图中/分镜/画面等词。
未确认姓名时使用她/他/女孩等自然称呼，不把 C001、未知人物1 等内部标识读进文案；亲属称呼也需要依据。
不要重复上段。不要把对白的猜测当旁白事实。因果仅使用事件 causes 的明确依据。
只写可核对的事实；不要为凑字数补写“一切看似平常”“无人知晓”等无依据评价。
用自然叙事串起人物的关键行动，少写衣服颜色、表情和发光特效。关系有原文依据时可以称呼父母或哥哥。
结尾围绕原作已有的未完成目标或具体风险收束，不用“画面到这里”“仍没有答案”等空泛句子。
返回 {{"text":"中文解说稿"}}。"""
        if feedback:
            prompt += f"\n上一版核验问题（必须修正，不可补造证据）：{prompt_json(feedback)}"
        output = self.model.ask("write", prompt, fresh=fresh)
        narration = text(output.get("text"), "解说稿")
        require(len(narration) <= 200, "单段过长，请重新生成")
        require(any("\u4e00" <= c <= "\u9fff" for c in narration), "解说稿必须使用中文")
        return {"segment_id": f"S_{event['event_id']}", "event_id": event["event_id"],
                "source_panel_ids": list(beat["source_panel_ids"]), "context_event_ids":beat.get("context_event_ids", []), "section": beat["section"],
                "text": narration, "review_status": "pending", "fact_check": None}

    def write(self, bible, plan):
        segments = []
        for beat in plan["beats"]:
            segments.append(self.write_segment(bible, plan, beat,
                                              segments[-1]["text"] if segments else ""))
        return {"bible_id": bible["bible_id"], "style": plan["style"], "segments": segments}
