"""Shared grounding for legacy single events and scene-based narration."""
from pipeline.story_model import require


def narrated_ids(segment):
    ids = segment.get("event_ids", [segment["event_id"]])
    require(isinstance(ids, list) and ids and all(isinstance(eid,str) for eid in ids) and len(ids) == len(set(ids))
            and segment["event_id"] in ids, "叙事事件关联无效")
    return ids


def context_ids(bible, ids):
    events = {e["event_id"]: e for e in bible["events"]}
    found, pending = set(), list(ids)
    while pending:
        eid = pending.pop()
        require(eid in events, "叙事引用未知事件")
        for cause in events[eid].get("causes", []):
            cid = cause["event_id"]
            require(cid in events, "因果引用未知事件")
            if cid not in found and cid not in ids:
                found.add(cid)
                pending.append(cid)
    return sorted(found, key=lambda eid: events[eid]["order"])


def source_ids(bible, ids):
    events = {e["event_id"]: e for e in bible["events"]}
    require(all(eid in events for eid in ids), "叙事引用未知事件")
    sources = {pid for eid in ids for pid in events[eid]["source_panel_ids"]}
    require(sources <= {p['source_panel_id'] for p in bible['panels']}, '事件引用未知原图')
    return [p["source_panel_id"] for p in bible["panels"] if p["source_panel_id"] in sources]
