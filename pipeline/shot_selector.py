"""Evidence constrained shot retrieval and chapter-wide beam optimization.

The model may rerank known evidence, but cannot create panel/event associations.
Quality is a soft preference; required story coverage is a hard constraint.
"""
from copy import deepcopy
import json
import math
import re
from pathlib import Path
import tempfile

from PIL import Image, ImageFilter, ImageStat

from pipeline.story_bible import StoryError, digest
from pipeline.story_context import narrated_ids
from pipeline.semantic_grounding import grounded_units
from pipeline.story_model import EvidenceImage
from pipeline.shot_text_ranges import (text_units, validate_ranges, validate_context_ranges,
                                       attach_mapping, build_mapping)


def require(ok, message):
    if not ok:
        raise StoryError(message)


def tokens(value):
    text = json.dumps(value, ensure_ascii=False).lower()
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,2}", text))


def similarity(a, b):
    a, b = tokens(a), tokens(b)
    return len(a & b) / max(1, len(a | b))


def required_events(source, segment):
    events = {e["event_id"]: e for e in source["bible"]["events"]}
    units = grounded_units(source['bible'], segment)
    if units:
        return sorted({eid for u in units for eid in u['event_ids']}, key=lambda eid: events[eid]['order'])
    if 'event_ids' in segment:
        # Context proves the narration; it does not mandate replaying every old cause.
        ids = narrated_ids(segment)
        require(all(eid in events for eid in ids), '缺少叙事事件证据')
        return sorted(ids,key=lambda eid:events[eid]['order'])
    required = {segment["event_id"], *segment.get("context_event_ids", [])}
    beat = next((b for b in source["plan"]["beats"] if b["event_id"] == segment["event_id"]), {})
    required.update(beat.get("context_event_ids", []))
    pending = list(required)
    while pending:
        eid = pending.pop()
        require(eid in events, f"缺少事件证据：{eid}")
        for cause in events[eid].get("causes", []):
            if cause["event_id"] not in required:
                required.add(cause["event_id"])
                pending.append(cause["event_id"])
    return sorted(required, key=lambda eid: events[eid]["order"])


def adjacent_narrations(source, segment):
    """Immediate narrative neighbors only; their images are editorial context."""
    segments = source.get('script',{}).get('segments',[])
    beats = source.get('plan',{}).get('beats') or segments
    index = next((i for i,b in enumerate(beats) if b['event_id']==segment['event_id']),None)
    if index is None:
        return []
    by_event = {s['event_id']:s for s in segments}
    result = []
    for offset,direction in ((-1,'previous'),(1,'next')):
        if not 0 <= index+offset < len(beats):
            continue
        beat = beats[index+offset]
        neighbor = by_event.get(beat['event_id'],beat)
        ids = required_events(source,neighbor)
        result.append({'direction':direction,'event_id':beat['event_id'],
                       'narration_id':neighbor.get('segment_id','S_'+beat['event_id']),
                       'event_ids':ids,'text':neighbor.get('text'),'title':beat.get('title')})
    return result


def complete_context_group(group, library):
    """Add a model-proposed neighboring image only when it fills uncovered words."""
    rows = {r['panel_id']:r for r in group['decisions']}
    if not any('text_ranges' in r for r in rows.values()):
        return group
    text = group['text']
    uncovered = {i for i,c in enumerate(text) if not c.isspace()}
    selected = {s['panel_id'] for s in group['selected']}
    for shot in group['selected']:
        for r in rows[shot['panel_id']].get('text_ranges',[]):
            uncovered.difference_update(range(r['start_char'],r['end_char']))
    candidates = sorted((r for r in rows.values() if r.get('context_only') and r['eligible']),
                        key=lambda r:-r['score'])
    for row in candidates:
        if row['panel_id'] in selected:
            continue
        positions = {i for r in row.get('text_ranges',[]) for i in range(r['start_char'],r['end_char'])}
        if not uncovered.intersection(positions):
            continue
        group['selected'].append({'shot_id':'SH_'+digest([group['narration_id'],row['panel_id']])[:16],
            'panel_id':row['panel_id'],'locked':False,'crop':row.get('suggested_crop',[0,0,1,1]),
            'motion':'auto','transition':'cut','duration_frames':None})
        selected.add(row['panel_id'])
        uncovered.difference_update(positions)
        row.update(selected=True,reason='相邻镜头承接原文：'+'；'.join(r['reason'] for r in row['text_ranges']))
    order = {p['panel_id']:p['order'] for p in library}
    group['selected'].sort(key=lambda s:(min((r['start_char'] for r in rows[s['panel_id']].get('text_ranges',[])),
                                            default=len(text)),order[s['panel_id']]))
    return attach_mapping(group)


def build_library(store, source):
    library = []
    for panel in source["bible"]["panels"]:
        pid = panel["source_panel_id"]
        events = [e for e in source["bible"]["events"] if pid in e["source_panel_ids"]]
        path = store.evidence_path(panel)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            thumb = image.resize((17, 16)).convert("L")
            pixels = list(thumb.getdata())
            bits = [pixels[y * 17 + x] > pixels[y * 17 + x + 1] for y in range(16) for x in range(16)]
            contrast = ImageStat.Stat(image.resize((128, 128)).convert("L")).stddev[0] / 64
            edge = ImageStat.Stat(image.resize((128, 128)).filter(ImageFilter.FIND_EDGES)).mean[0] / 32
            quality = min(1., min(image.size) / 720) * .6 + min(1., contrast) * .4
            library.append({"panel_id": pid, "order": panel["order"], "width": image.width,
                            "height": image.height, "sha256": panel["sha256"],
                            "evidence_path":str(path),
                            "event_ids": [e["event_id"] for e in events],
                            "character_ids": sorted({c for e in events for c in e["character_ids"]}),
                            "scene": panel.get("scene", panel.get("location", next(
                                (e["scene"] for e in events if e.get("scene")), ""))),
                            "actions": [a for e in events for a in e.get("actions", [])],
                            "dialogue": panel.get("dialogue", []),
                            "panel_type": panel.get("panel_type", "story"),
                            "boundary_uncertain": panel.get("boundary_uncertain", False),
                            "boundary_review": panel.get("boundary_review"),
                            "quality": round(quality, 4), "visual": round(min(1., edge), 4),
                            "color_signature": list(image.resize((8, 8)).tobytes()),
                            "fingerprint": ''.join('1' if b else '0' for b in bits)})
    return library


def visually_similar(a, b):
    if a["sha256"] == b["sha256"]:
        return True
    # dHash alone collapses flat/low-information drawings; also compare aspect and visual statistics.
    return (abs(a["width"] / a["height"] - b["width"] / b["height"]) < .12
            and abs(a["quality"] - b["quality"]) < .08
            and abs(a["visual"] - b["visual"]) < .08
            and sum(abs(x - y) for x, y in zip(a.get("color_signature", []), b.get("color_signature", []))) / 192 < 12
            and sum(x != y for x, y in zip(a["fingerprint"], b["fingerprint"])) <= 10)


def same_information(a, b):
    return (set(a["event_ids"]) == set(b["event_ids"])
            and tokens(a["actions"]) == tokens(b["actions"])
            and tokens(a["dialogue"]) == tokens(b["dialogue"]))


class ShotSelectorAgent:
    def __init__(self, model=None):
        self.model = model

    def review_selected(self,source,library,groups):
        """Inspect the actual proposed crops; a rejected crop is restored, not hidden."""
        if self.model is None:
            return {'status':'not_run','reason':'未调用视觉模型'}
        by_id = {p['panel_id']:p for p in library}
        reports = []
        with tempfile.TemporaryDirectory(prefix='aniflow-shot-review-') as folder:
            for group in groups:
                for attempt in range(2):
                    checked = []
                    shots = group['selected']
                    for start in range(0,len(shots),12):
                        batch = shots[start:start+12]
                        images = []
                        for i,shot in enumerate(batch):
                            with Image.open(by_id[shot['panel_id']]['evidence_path']) as opened:
                                image = opened.convert('RGB')
                                x,y,w,h = shot['crop']
                                image = image.crop((round(x*image.width),round(y*image.height),
                                                    round((x+w)*image.width),round((y+h)*image.height)))
                                path = Path(folder)/f'{start+i}.png';image.save(path)
                                images.append(EvidenceImage(path,shot['panel_id']))
                        prompt = '''Selected Shot Review v4。附图是即将用于成片的实际裁切，不是完整原图。
panel_id 必须逐字复制 panels 内的真实来源ID；附图序号1、2等不是来源ID，不得替换。
核对主体、必要对白和动作是否完整，是否与本段旁白相符。允许有剧情作用的安静镜头或文字证据。
有 text_ranges 时，按每张图关联的精确原文复核；不能因整段提到该情节就判可用。
同时读取每个范围的 match_type 和 reason：direct 为直接展示，transition 为延续/过渡，commentary 为吐槽反应，callback 为回扣。
过渡可停留在上一镜，吐槽可承接相关反应或动作，回扣可重现前面相关物品；不要求每句评论都出现逐字对应的对白或新画面。
context_only 画面来自相邻段落，仅作上述承接，不是新增的当前事件证据。检查其人物、场景与情绪承接是否合理。
有 semantic_units 时须核对该图对应 unit_ids 的具体原句，不能凭整段提到过该事件就判可用。
对照单元顺序，不能提前展示后文才揭晓的身份、答案或结果。若同图含后续揭晓且裁切无法避开，判 unusable，不以完整来源覆盖为通过理由。
不要把只剩空白、截掉关键动作/对白的裁切判为可用。只使用附图ID，逐张覆盖，不得补造剧情。
返回 {"checks":[{"panel_id":"原始来源ID","usable":true,"reason":"实际可见依据"}]}。
资料：'''+json.dumps({'narration':group['text'],'panels':[s['panel_id'] for s in batch],
                                    'semantic_units':group.get('semantic_units',[]),
                                    'shot_units':[{k:s.get(k) for k in ('panel_id','unit_ids')} for s in batch],
                                    'shot_text_ranges':[{k:s.get(k) for k in ('panel_id','text_ranges')} for s in batch],
                                    'shot_context':[{k:s.get(k) for k in ('panel_id','context_only','context_origins')} for s in batch],
                                    'adjacent_narrations':adjacent_narrations(source,{'event_id':group['event_id']}),
                                    'events':[e for e in source['bible']['events'] if e['event_id'] in group['required_event_ids']]},ensure_ascii=False)
                        raw = self.model.ask('visual_review',prompt,images=images)
                        rows = raw.get('checks')
                        require(isinstance(rows,list) and len(rows)==len(batch), '裁切复核漏图')
                        ids = [r.get('panel_id') for r in rows if isinstance(r,dict)]
                        require(len(ids)==len(set(ids)) and set(ids)=={s['panel_id'] for s in batch},'裁切复核引用未知或重复分镜')
                        require(all(type(r.get('usable')) is bool and isinstance(r.get('reason'),str) and r['reason'].strip()
                                    for r in rows),'裁切复核结果无效')
                        checked.extend(rows)
                    restored = False
                    for row in checked:
                        shot = next(s for s in shots if s['panel_id']==row['panel_id'])
                        if not row['usable'] and shot['crop'] != [0,0,1,1]:
                            shot['crop'] = [0,0,1,1]; restored = True
                    if not restored or attempt == 1:
                        break
                report = {'narration_id':group['narration_id'],'checks':checked,
                          'status':'passed' if all(r['usable'] for r in checked) else 'needs_review'}
                group['visual_review'] = report
                for row in checked:
                    decision = next(r for r in group['decisions'] if r['panel_id']==row['panel_id'])
                    decision['reason'] += '；裁切复核：'+row['reason']
                reports.append(report)
        return {'status':'passed' if all(r['status']=='passed' for r in reports) else 'needs_review','groups':reports}

    def candidates(self, source, library, segment):
        required = required_events(source, segment)
        units = grounded_units(source['bible'], segment)
        display = {pid for u in units for pid in u['display_panel_ids']}
        adjacent = adjacent_narrations(source,segment) if self.model and not units else []
        panel_meta = {p['source_panel_id']:p for p in source['bible']['panels']}
        event = next(e for e in source["bible"]["events"] if e["event_id"] == segment["event_id"])
        rows = []
        for panel in library:
            overlap = set(panel["event_ids"]) & set(required)
            origins = [{k:n[k] for k in ('direction','event_id','narration_id','event_ids')}
                       for n in adjacent if set(panel['event_ids']) & set(n['event_ids'])]
            if panel.get('panel_type','story') != 'story' or panel_meta.get(panel['panel_id'],{}).get('historical'):
                origins = []
            direct = bool(overlap) and (not units or panel['panel_id'] in display)
            score = {
                "importance": max((e["importance"] / 5 for e in source["bible"]["events"]
                                   if e["event_id"] in overlap), default=0),
                "narration_match": .75 * bool(overlap) + .25 * similarity(
                    segment["text"], [panel["actions"], panel["dialogue"], panel["scene"]]),
                "visual": panel["visual"], "quality": panel["quality"],
                "continuity": len(set(event["character_ids"]) & set(panel["character_ids"])) /
                              max(1, len(event["character_ids"]))}
            total = sum(score[k] * w for k, w in
                        {"importance": .25, "narration_match": .35, "visual": .15,
                         "continuity": .15, "quality": .10}.items())
            rows.append({"panel_id": panel["panel_id"], "narration_id": segment["segment_id"],
                         "event_id": segment["event_id"], "covers": sorted(overlap),
                         "eligible": direct or bool(origins), 'context_only':bool(origins) and not direct,
                         'context_origins':origins, "scores": score, "score": round(total, 5),
                         "retrieval_reason": "事件证据命中；结合人物、场景、动作和对白匹配" if overlap else
                         "相邻段落画面，仅可用于过渡、吐槽反应或回扣" if origins else
                         "无此旁白事件的来源证据，不能仅凭外观替代剧情"})
        if self.model:
            eligible = [r for r in rows if r["eligible"]]
            panels = [p for p in library if p['panel_id'] in {r['panel_id'] for r in eligible}]
            ranked = []
            for start in range(0,len(panels),12):
                batch = panels[start:start+12]
                row_by_id = {r['panel_id']:r for r in rows}
                facts = [{**{k:p[k] for k in ('panel_id','event_ids','character_ids','scene','actions','dialogue')},
                          **{k:row_by_id[p['panel_id']][k] for k in ('context_only','context_origins')}} for p in batch]
                images = [EvidenceImage(p['evidence_path'],p['panel_id']) for p in batch if p.get('evidence_path')]
                require(len(images)==len(batch), '看图选镜缺少原始图片')
                raw = self.model.ask('shot_select', '''Visual Shot Selector v4。逐张看附图，为旁白选择实际有表现力的画面。
附图顺序与 panels 一致。只能使用列表ID，不得凭相似人物补造事件关联。
panel_id 必须逐字复制 panels 的真实来源ID，不能用附图序号1、2代替。逐张返回本批全部图片。
每张图片须返回 text_ranges，表示它应展示在哪些具体原文处；确实无匹配则返回空数组。
start_char/end_char 是 narration.text 内从0开始的Unicode字符位置，末端不包含；一个表情也算一个字符。
text 必须逐字等于 narration.text[start_char:end_char]，保留空白、换行与标点。不得改写原文。
text_units 已给出可直接复制的位置与原文，可以合并相邻片段；同图多段范围按原文顺序返回，不得交叠。
按真实讲述匹配动作、对白、反应及评论，不要为填满范围把所有图片都关联整段；多图可对应同一处原文。
这是剪辑展示关联，不是重新事实审稿。过渡、评价与吐槽可延续相邻相关的反应或环境画面；尽可能覆盖本段全文，确实无适合画面则保留缺口供人工审核。
每个 text_ranges 条目须给 match_type 和 reason：direct=直接展示，transition=过渡延续，commentary=吐槽反应，callback=画面回扣。
先匹配正文动作/对白，再为未覆盖的过渡、评价与吐槽补配承接画面；同一张图可以持续停留，也可以在多个不连续文字范围再次出现。
段首过渡优先延续上一镜；等关键消息说出后才展示揭晓。例如“对面又发来一条消息”可延用上一段看手机的画面，不能提前露出肉罐头消息。
吐槽可以停留在相关人物的反应、动作或物品上，回扣可以重现前面已经出现过的相关画面；结合全文保持人物、场景与情绪连贯。
context_only=true 的图片仅是相邻段落参考，不增加当前事实或必保事件。借用上一段时只能标 transition/commentary/callback 并说明理由。
仅来自下一段的图片最多用于本段最后一个 text_unit 的 transition，必须避开尚未讲述的身份、消息或结果；不能用作提前的反应或回扣。
本批图片不适用的范围保留空缺，后续合并各批检查覆盖；不要因为图片相邻就自动把整段都分给它。
优先具体动作、反应、环境和可读的完整对白组合。纯白、零散文字条、重复信息降分，但不能漏关键证据。
按当前原句选择展示，不能因整段稍后会提到，就提前展示答案、身份或结果；有作用的安静镜头可以停留，不要求固定频率切镜。
检查主体或对白是否截断。只在能保全当前事件动作和必要对白时建议裁切，否则用 [0,0,1,1]。
crop 为原图归一化 [x,y,w,h]。不要因画面更漂亮而切掉证据。对每张图给出具体可见理由。
返回 {"candidates":[{"panel_id":"原始来源ID","score":0.8,"visual_role":"action|reaction|setting|dialogue|text_only|blank",
"crop":[0,0,1,1],"reason":"实际画面依据"}]}。
每项另外必须包含 "text_ranges":[{"start_char":0,"end_char":5,"text":"对应的原文","match_type":"direct|transition|commentary|callback","reason":"这张图如何承接这段文字"}]；示例位置仅说明结构，实际复制 text_units。
资料：'''+json.dumps({'narration':segment,'text_units':text_units(segment['text']),
                        'text_offset_unit':'unicode_codepoint','adjacent_narrations':adjacent,'panels':facts},ensure_ascii=False), images=images)
                require(isinstance(raw.get('candidates'),list), '镜头模型返回格式无效')
                require(all(isinstance(x,dict) and x.get('panel_id') in {p['panel_id'] for p in batch}
                            for x in raw['candidates']), '镜头模型引用未知或无证据分镜')
                require(len(raw['candidates']) == len(batch) and
                        {r['panel_id'] for r in raw['candidates']} == {p['panel_id'] for p in batch}, '镜头模型漏图或重复图片')
                ranked.extend(raw['candidates'])
            seen = set()
            for item in ranked:
                require(isinstance(item, dict) and item.get("panel_id") in {r["panel_id"] for r in eligible}
                        and item["panel_id"] not in seen, "镜头模型引用未知、重复或无证据分镜")
                value = item.get("score")
                require(type(value) in (int, float) and 0 <= value <= 1
                        and isinstance(item.get("reason"), str) and bool(item["reason"].strip()),
                        "镜头模型评分或理由无效")
                seen.add(item["panel_id"])
                row = next(r for r in rows if r["panel_id"] == item["panel_id"])
                crop = item.get('crop',[0,0,1,1])
                require(isinstance(crop,list) and len(crop)==4 and all(type(x) in (int,float) and math.isfinite(x) for x in crop)
                        and crop[0]>=0 and crop[1]>=0 and crop[2]>0 and crop[3]>0
                        and crop[0]+crop[2]<=1.000001 and crop[1]+crop[3]<=1.000001, '镜头模型裁切无效')
                role = item.get('visual_role','unknown')
                require(role in {'unknown','action','reaction','setting','dialogue','text_only','blank'}, '画面类型无效')
                ranges = validate_ranges(segment['text'],item.get('text_ranges'),require_annotations=True)
                try:
                    validate_context_ranges(segment['text'],ranges,row)
                except StoryError as exc:
                    # Decline an unsafe editorial association without discarding
                    # valid current-event pictures from the same model response.
                    row.update(eligible=False,rejected_text_ranges=deepcopy(ranges),
                               context_validation_error=str(exc))
                    row['retrieval_reason'] += '；承接范围被拒绝：'+str(exc)
                    ranges = []
                penalty = .3 if role == 'blank' else .12 if role == 'text_only' else 0
                row.update(score=round(.35*row['score']+.65*value-penalty,5),model_reason=item['reason'],
                           suggested_crop=crop,visual_role=role,visual_reviewed=bool(images),text_ranges=ranges)
        return rows, required

    def fill_text_gaps(self, source, library, group):
        """Ask explicitly about remaining words, using current/previous images only."""
        if self.model is None or group.get('locked') or group.get('semantic_units'):
            return group
        rows = group['decisions']
        pool = [r for r in rows if r['covers'] or any(
            o['direction']=='previous' for o in r.get('context_origins',[]))]
        by_panel = {p['panel_id']:p for p in library}
        selected = {s['panel_id'] for s in group['selected']}
        # Prefer already selected shots, then nearby prior reactions/objects.
        pool.sort(key=lambda r:(r['panel_id'] not in selected,-r['score']))
        attempts, errors = 0, []
        for start in range(0,len(pool),12):
            mapping = build_mapping(group)
            gaps = [{k:u[k] for k in ('start_char','end_char','text')}
                    for u in mapping['units'] if not u['display_panel_ids'] and u['text'].strip()]
            if not gaps:
                break
            batch = pool[start:start+12]
            facts = [{**{k:by_panel[r['panel_id']][k] for k in ('panel_id','event_ids','scene','actions','dialogue')},
                      **{k:r.get(k) for k in ('context_only','context_origins','text_ranges')}} for r in batch]
            images = [EvidenceImage(by_panel[r['panel_id']]['evidence_path'],r['panel_id']) for r in batch]
            prompt = '''Shot Text Gap Fill v1。只为当前讲稿尚未配图的 missing_text_ranges 选择承接画面，不改讲稿、不重新评分。
panel_id 必须复制 panels 的真实来源ID，不能用附图序号代替。只返回本批内图片，只填给出的缺口；已有关联的正文不得覆盖。
过渡优先延续上一镜；吐槽停留在对应人物反应、动作或物品；回扣可重现之前相关画面。同图可以重复使用，不必每句话换图。
这是剪辑承接，不是逐字事实审稿。像“对面又发来消息”“看来救援条件没谈到位”“罐头也出了力”等过渡和评论，可由相邻相关画面承接。
借用上一段只可用 transition/commentary/callback 并说明关系；当前段画面还可 direct。不要提前展示尚未讲到的消息、身份或结果。
start_char/end_char 从0计数，按Unicode字符，末端不包含；text 必须逐字复制原文及其空白标点。可把缺口再拆分，但每个范围必须完整位于缺口内。
尽量补齐这些缺口。确实无合适画面时返回空 assignments，并保留人工审核，不编造解释或新增画面。
返回 {"assignments":[{"panel_id":"原始来源ID","text_ranges":[{"start_char":0,"end_char":5,"text":"对应的原文","match_type":"transition|commentary|callback|direct","reason":"画面如何承接"}]}]}。
资料：'''+json.dumps({'narration':{'text':group['text'],'event_id':group['event_id']},
                         'missing_text_ranges':gaps,'text_units':text_units(group['text']),
                         'adjacent_narrations':adjacent_narrations(source,{'event_id':group['event_id']}),
                         'panels':facts},ensure_ascii=False)
            attempts += 1
            try:
                raw = self.model.ask('shot_text_fill',prompt,images=images)
                assignments = raw.get('assignments')
                require(isinstance(assignments,list),'缺口补配格式无效')
                allowed = {r['panel_id']:r for r in batch}
                staged, seen = {}, set()
                for item in assignments:
                    require(isinstance(item,dict) and item.get('panel_id') in allowed
                            and item['panel_id'] not in seen,'缺口补配引用未知或重复图片')
                    pid=item['panel_id'];seen.add(pid)
                    ranges=validate_ranges(group['text'],item.get('text_ranges'),require_annotations=True)
                    require(all(any(g['start_char']<=r['start_char']<r['end_char']<=g['end_char'] for g in gaps)
                                for r in ranges),'缺口补配超出指定原文范围')
                    validate_context_ranges(group['text'],ranges,allowed[pid])
                    old=allowed[pid].get('text_ranges',[]) if pid in selected else []
                    staged[pid]=validate_ranges(group['text'],sorted(deepcopy(old)+ranges,key=lambda r:r['start_char']))
                # Apply only after the entire response has passed validation.
                for pid,ranges in staged.items():
                    row=allowed[pid]
                    row.setdefault('initial_text_ranges',deepcopy(row.get('text_ranges',[])))
                    row.update(text_ranges=ranges,eligible=True)
                group=complete_context_group(group,library)
                selected={s['panel_id'] for s in group['selected']}
            except Exception as exc:
                errors.append(str(exc))
        group['text_fill']={'attempts':attempts,'errors':errors,
                            'status':'completed' if group['text_mapping']['status']=='mapped' else 'needs_review'}
        return group


class GlobalShotOptimizer:
    """Beam search of alternative evidence covers across every narration segment."""
    @staticmethod
    def grounded_group(source, segment, library, selector, old):
        units = grounded_units(source['bible'], segment)
        if old and old.get('locked'):
            # Do not repair human choices behind their back. Timeline validation
            # will report missing/stale unit associations without altering them.
            return deepcopy(old)
        rows, required = selector.candidates(source, library, segment)
        by_id = {p['panel_id']: p for p in library}
        selected = {}
        for unit in units:
            uncovered = set(unit['event_ids'])
            candidates = [r for r in rows if r['eligible'] and r['panel_id'] in unit['display_panel_ids']]
            while uncovered:
                row = max(candidates, key=lambda r: (len(set(r['covers']) & uncovered), r['score']), default=None)
                require(row and set(row['covers']) & uncovered, '当前口播单元缺少展示来源')
                pid = row['panel_id']
                if pid not in selected:
                    selected[pid] = {'shot_id': 'SH_'+digest([segment['segment_id'], pid])[:16],
                                     'panel_id': pid, 'locked': False, 'crop': row.get('suggested_crop',[0,0,1,1]),
                                     'motion':'auto', 'transition':'cut', 'duration_frames':None, 'unit_ids':[]}
                selected[pid]['unit_ids'].append(unit['unit_id'])
                uncovered -= set(by_id[pid]['event_ids'])
                candidates.remove(row)
        for row in rows:
            row.update(selected=row['panel_id'] in selected,
                       reason='按当前口播事实选择展示来源' if row['panel_id'] in selected else '未用于当前语义单元')
        return attach_mapping({'narration_id': segment['segment_id'], 'event_id':segment['event_id'], 'text':segment['text'],
                'required_event_ids':required, 'semantic_units':deepcopy(units), 'selected':list(selected.values()),
                'decisions':rows, 'locked':False, 'review_status':'pending'})

    def optimize(self, source, library, selector, previous=None):
        by_id = {p["panel_id"]: p for p in library}
        previous = previous or {}
        groups = []
        mapped = {}
        for segment in source["script"]["segments"]:
            old = previous.get(segment["segment_id"])
            if grounded_units(source['bible'], segment):
                mapped[segment['segment_id']] = self.grounded_group(source, segment, library, selector, old)
                continue
            if old and old.get("locked"):
                rows, required = deepcopy(old["decisions"]), required_events(source, segment)
                selected = deepcopy(old["selected"])
                require(all(s["panel_id"] in by_id for s in selected), "锁定镜头来源已缺失")
                require(set(required) <= {e for s in selected for e in by_id[s["panel_id"]]["event_ids"]},
                        "锁定镜头未覆盖必要剧情，请人工补充")
                options = [selected]
            else:
                rows, required = selector.candidates(source, library, segment)
                def make_shot(pid):
                    row = next(r for r in rows if r['panel_id']==pid)
                    return {'shot_id':'SH_'+digest([segment['segment_id'],pid])[:16], 'panel_id':pid,
                            'locked':False,'crop':row.get('suggested_crop',[0,0,1,1]),
                            'motion':'auto','transition':'cut','duration_frames':None}
                eligible = sorted((r for r in rows if r["eligible"] and r['covers']), key=lambda r: -r["score"])
                require(bool(eligible), f"旁白 {segment['segment_id']} 没有来源画面")
                # Several greedy set covers with different seeds produce alternatives.
                options, signatures = [], set()
                for seed in eligible[:8]:
                    picked, uncovered = [], set(required)
                    pending = [seed] + [r for r in eligible if r is not seed]
                    while uncovered:
                        row = pending.pop(0) if not picked else max(
                            pending, key=lambda r: (len(set(r["covers"]) & uncovered), r["score"]), default=None)
                        require(row is not None and bool(set(row["covers"]) & uncovered),
                                f"必要剧情无可用画面：{','.join(sorted(uncovered))}")
                        picked.append(row)
                        uncovered -= set(row["covers"])
                        if row in pending:
                            pending.remove(row)
                    # Drop redundant seeds, but never remove novel dialogue/actions.
                    for row in picked[:]:
                        others = [r for r in picked if r is not row]
                        if (set(required) <= {e for r in others for e in r["covers"]}
                                and any(visually_similar(by_id[row["panel_id"]], by_id[r["panel_id"]])
                                        and same_information(by_id[row["panel_id"]], by_id[r["panel_id"]])
                                        for r in others)):
                            picked.remove(row)
                    picked.sort(key=lambda r: by_id[r["panel_id"]]["order"])
                    sig = tuple(r["panel_id"] for r in picked)
                    if sig not in signatures:
                        signatures.add(sig)
                        options.append([make_shot(pid) for pid in sig])
                # Retrieval budget only; actual timing is always measured from TTS.
                target = max(1, math.ceil(len(segment['text'])/24)) if 'event_ids' in segment else min(4,max(1,math.ceil(len(segment['text'])/22)))
                for option in options[:]:
                    richer = deepcopy(option)
                    for row in eligible:
                        panel = by_id[row["panel_id"]]
                        if len(richer) >= target:
                            break
                        if row["panel_id"] not in {s["panel_id"] for s in richer} and not any(
                            visually_similar(panel, by_id[s["panel_id"]]) and same_information(panel, by_id[s["panel_id"]])
                            for s in richer):
                            richer.append(make_shot(row['panel_id']))
                    richer.sort(key=lambda s: by_id[s["panel_id"]]["order"])
                    sig = tuple(s["panel_id"] for s in richer)
                    if sig not in signatures:
                        signatures.add(sig)
                        options.append(richer)
            groups.append((segment, rows, required, options, old))
        # State: score, selected options. Duplicate cost uses the whole prefix.
        beam = [(0., [])]
        for segment, rows, required, options, old in groups:
            scores = {r["panel_id"]: r["score"] for r in rows}
            next_beam = []
            for score, path in beam:
                prior = [by_id[s["panel_id"]] for option in path for s in option]
                for option in options:
                    panels = [by_id[s["panel_id"]] for s in option]
                    cost = sum(scores[p["panel_id"]] for p in panels) / len(panels) - .04 * len(panels)
                    target = max(1,math.ceil(len(segment['text'])/24)) if 'event_ids' in segment else min(4,max(1,math.ceil(len(segment['text'])/22)))
                    cost += .10 * min(len(panels), target)
                    if prior:
                        cost += .12 * bool(set(prior[-1]["character_ids"]) & set(panels[0]["character_ids"]))
                        cost -= .10 * (panels[0]["order"] < prior[-1]["order"])
                        cost += .08 * bool(prior[-1]["scene"] and prior[-1]["scene"] == panels[0]["scene"])
                    for panel in panels:
                        cost -= .8 * any(visually_similar(panel, p) and same_information(panel, p) for p in prior)
                    next_beam.append((score + cost, path + [deepcopy(option)]))
            beam = sorted(next_beam, key=lambda item: -item[0])[:32]
        result = []
        for (segment, rows, required, options, old), selected in zip(groups, beam[0][1]):
            if old and old.get("locked"):
                result.append(deepcopy(old))
                continue
            selected_ids = {s["panel_id"] for s in selected}
            for row in rows:
                panel = by_id[row["panel_id"]]
                duplicate = next((pid for pid in selected_ids if pid != row["panel_id"]
                                  and visually_similar(panel, by_id[pid]) and same_information(panel, by_id[pid])), None)
                row["selected"] = row["panel_id"] in selected_ids
                row["reason"] = ("覆盖必要事件 " + "、".join(row["covers"]) +
                                 f"；综合评分 {row['score']:.3f}；全局排序兼顾连续性，必要剧情不受质量过滤") if row["selected"] else (
                                 f"与已选 {duplicate} 相似且没有新增剧情信息，压缩重复" if duplicate else
                                 "已有更合适的证据镜头；" + row["retrieval_reason"])
                if row.get("model_reason"):
                    row["reason"] += "；模型复核：" + row["model_reason"]
            result.append(complete_context_group({"narration_id": segment["segment_id"], "event_id": segment["event_id"],
                           "text": segment["text"], "required_event_ids": required,
                           "selected": selected, "decisions": rows,
                           "locked": bool(old and old.get("locked")),
                           "review_status": old.get("review_status", "pending") if old and old.get("locked") else "pending"},library))
        by_narration = {g['narration_id']:g for g in result}
        by_narration.update(mapped)
        ordered = [by_narration[s['segment_id']] for s in source['script']['segments']]
        if getattr(selector,'model',None) is not None and hasattr(selector,'fill_text_gaps'):
            ordered = [selector.fill_text_gaps(source,library,g) if not g.get('locked') and
                       g.get('text_mapping',{}).get('status')=='needs_review' else g for g in ordered]
        return ordered


def coverage_report(groups, library):
    by_id = {p["panel_id"]: p for p in library}
    required = {e for g in groups for e in g["required_event_ids"]}
    covered, missing_by_narration, selected, repeated, intentional_reuse = set(), {}, [], 0, 0
    local_required, local_covered = 0, 0
    missing_semantic_units = []
    missing_text_ranges = []
    for group in groups:
        if 'text_mapping' in group:
            mapping = build_mapping(group)
            if mapping != group['text_mapping'] or mapping['status'] != 'mapped':
                missing_text_ranges.append({'narration_id':group['narration_id'],'issues':mapping['issues'] or ['文字关联已过期']})
        for unit in group.get('semantic_units', []):
            matching = [s for s in group['selected'] if unit['unit_id'] in s.get('unit_ids', [])
                        and s['panel_id'] in unit['display_panel_ids']]
            facts = {eid for s in matching for eid in by_id[s['panel_id']]['event_ids']}
            if not set(unit['event_ids']) <= facts:
                missing_semantic_units.append({'narration_id':group['narration_id'], 'unit_id':unit['unit_id']})
        local = {e for s in group["selected"] for e in by_id[s["panel_id"]]["event_ids"]}
        missing = set(group["required_event_ids"]) - local
        local_required += len(group["required_event_ids"])
        local_covered += len(set(group["required_event_ids"]) & local)
        if missing:
            missing_by_narration[group["narration_id"]] = sorted(missing)
        covered |= local
        for shot in group["selected"]:
            panel = by_id[shot["panel_id"]]
            duplicate = any(visually_similar(panel, p) and same_information(panel, p) for p in selected)
            deliberate = any(r.get('match_type') in {'transition','commentary','callback'} for r in shot.get('text_ranges',[]))
            intentional_reuse += bool(duplicate and deliberate)
            repeated += bool(duplicate and not deliberate)
            selected.append(panel)
    return {"coverage_rate": local_covered / max(1, local_required),
            'missing_semantic_units':missing_semantic_units,
            'missing_text_ranges':missing_text_ranges,
            "unique_event_coverage_rate": len(required & covered) / max(1, len(required)),
            "required_event_ids": sorted(required), "missing_event_ids": sorted(required - covered),
            "missing_by_narration": missing_by_narration, "selected_count": len(selected),
            "duplicate_rate": repeated / max(1, len(selected)), "duplicate_count": repeated,
            'intentional_reuse_count':intentional_reuse}
