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
                                path = Path(folder)/f'{start+i}.png';image.save(path);images.append(str(path))
                        prompt = '''Selected Shot Review v2。附图是即将用于成片的实际裁切，不是完整原图。
核对主体、必要对白和动作是否完整，是否与本段旁白相符。允许有剧情作用的安静镜头或文字证据。
有 semantic_units 时须核对该图对应 unit_ids 的具体原句，不能凭整段提到过该事件就判可用。
对照单元顺序，不能提前展示后文才揭晓的身份、答案或结果。若同图含后续揭晓且裁切无法避开，判 unusable，不以完整来源覆盖为通过理由。
不要把只剩空白、截掉关键动作/对白的裁切判为可用。只使用附图ID，逐张覆盖，不得补造剧情。
返回 {"checks":[{"panel_id":"1","usable":true,"reason":"实际可见依据"}]}。
资料：'''+json.dumps({'narration':group['text'],'panels':[s['panel_id'] for s in batch],
                                    'semantic_units':group.get('semantic_units',[]),
                                    'shot_units':[{k:s.get(k) for k in ('panel_id','unit_ids')} for s in batch],
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
        event = next(e for e in source["bible"]["events"] if e["event_id"] == segment["event_id"])
        rows = []
        for panel in library:
            overlap = set(panel["event_ids"]) & set(required)
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
                         "eligible": bool(overlap) and (not units or panel['panel_id'] in display), "scores": score, "score": round(total, 5),
                         "retrieval_reason": "事件证据命中；结合人物、场景、动作和对白匹配" if overlap else
                         "无此旁白事件的来源证据，不能仅凭外观替代剧情"})
        if self.model:
            eligible = [r for r in rows if r["eligible"]]
            panels = [p for p in library if p['panel_id'] in {r['panel_id'] for r in eligible}]
            ranked = []
            for start in range(0,len(panels),12):
                batch = panels[start:start+12]
                facts = [{k:p[k] for k in ('panel_id','event_ids','character_ids','scene','actions','dialogue')} for p in batch]
                images = [p['evidence_path'] for p in batch if p.get('evidence_path')]
                require(len(images)==len(batch), '看图选镜缺少原始图片')
                raw = self.model.ask('shot_select', '''Visual Shot Selector v2。逐张看附图，为旁白选择实际有表现力的画面。
附图顺序与 panels 一致。只能使用列表ID，不得凭相似人物补造事件关联。
优先具体动作、反应、环境和可读的完整对白组合。纯白、零散文字条、重复信息降分，但不能漏关键证据。
按当前原句选择展示，不能因整段稍后会提到，就提前展示答案、身份或结果；有作用的安静镜头可以停留，不要求固定频率切镜。
检查主体或对白是否截断。只在能保全当前事件动作和必要对白时建议裁切，否则用 [0,0,1,1]。
crop 为原图归一化 [x,y,w,h]。不要因画面更漂亮而切掉证据。对每张图给出具体可见理由。
返回 {"candidates":[{"panel_id":"1","score":0.8,"visual_role":"action|reaction|setting|dialogue|text_only|blank",
"crop":[0,0,1,1],"reason":"实际画面依据"}]}。
资料：'''+json.dumps({'narration':segment,'panels':facts},ensure_ascii=False), images=images)
                require(isinstance(raw.get('candidates'),list), '镜头模型返回格式无效')
                require(all(isinstance(x,dict) and x.get('panel_id') in {p['panel_id'] for p in batch}
                            for x in raw['candidates']), '镜头模型引用未知或无证据分镜')
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
                penalty = .3 if role == 'blank' else .12 if role == 'text_only' else 0
                row.update(score=round(.35*row['score']+.65*value-penalty,5),model_reason=item['reason'],
                           suggested_crop=crop,visual_role=role,visual_reviewed=bool(images))
        return rows, required


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
        return {'narration_id': segment['segment_id'], 'event_id':segment['event_id'], 'text':segment['text'],
                'required_event_ids':required, 'semantic_units':deepcopy(units), 'selected':list(selected.values()),
                'decisions':rows, 'locked':False, 'review_status':'pending'}

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
                eligible = sorted((r for r in rows if r["eligible"]), key=lambda r: -r["score"])
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
            result.append({"narration_id": segment["segment_id"], "event_id": segment["event_id"],
                           "text": segment["text"], "required_event_ids": required,
                           "selected": selected, "decisions": rows,
                           "locked": bool(old and old.get("locked")),
                           "review_status": old.get("review_status", "pending") if old and old.get("locked") else "pending"})
        by_narration = {g['narration_id']:g for g in result}
        by_narration.update(mapped)
        return [by_narration[s['segment_id']] for s in source['script']['segments']]


def coverage_report(groups, library):
    by_id = {p["panel_id"]: p for p in library}
    required = {e for g in groups for e in g["required_event_ids"]}
    covered, missing_by_narration, selected, repeated = set(), {}, [], 0
    local_required, local_covered = 0, 0
    missing_semantic_units = []
    for group in groups:
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
            repeated += any(visually_similar(panel, p) and same_information(panel, p) for p in selected)
            selected.append(panel)
    return {"coverage_rate": local_covered / max(1, local_required),
            'missing_semantic_units':missing_semantic_units,
            "unique_event_coverage_rate": len(required & covered) / max(1, len(required)),
            "required_event_ids": sorted(required), "missing_event_ids": sorted(required - covered),
            "missing_by_narration": missing_by_narration, "selected_count": len(selected),
            "duplicate_rate": repeated / max(1, len(selected)), "duplicate_count": repeated}
