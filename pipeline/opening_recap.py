"""Bind a one-sentence opening recap to the immediately preceding chapter."""
from copy import deepcopy

from pipeline.story_context import context_ids, source_ids


def recap_material(bible):
    prior = bible.get('prior_context') or {}
    chapter = prior.get('chapter')
    previous = prior.get('previous_chapter') or {}
    if (type(chapter) is not int or chapter <= 1 or prior.get('status') != 'available'
            or previous.get('chapter') != chapter-1 or not previous.get('revision')):
        return None
    events = {e['event_id']: e for e in bible['events']}
    panels = {p['source_panel_id']: p for p in bible['panels']}
    ids = []
    for row in prior.get('summary', []):
        if not isinstance(row, dict) or not isinstance(row.get('text'), str) or not row['text'].strip():
            continue
        refs = row.get('event_ids')
        if not isinstance(refs, list) or not refs or not all(isinstance(e, str) and e in events for e in refs):
            continue
        usable = True
        for eid in refs:
            event = events[eid]
            origin = event.get('history_origin') or {}
            if (not event.get('historical') or origin.get('chapter') != previous['chapter']
                    or origin.get('revision') != previous['revision'] or not event.get('source_panel_ids')
                    or any(pid not in panels or not panels[pid].get('historical')
                           for pid in event['source_panel_ids'])):
                usable = False
                break
        if not usable:
            continue  # Never keep half a sentence by discarding invalid source IDs.
        ids.extend(eid for eid in refs if eid not in ids)
    if not ids:
        return None
    return {'previous_chapter': {k: previous[k] for k in ('chapter','revision')},
            'event_ids': ids}


def with_opening_recap(bible, plan):
    """Keep the same node count and anchor; add historical spoken scope only."""
    if not plan.get('beats'):
        return plan
    recap = recap_material(bible)
    old = plan.get('opening_recap') or {}
    if not recap and not old:
        return plan
    result = deepcopy(plan)
    first = result['beats'][0]
    old_added = set(old.get('added_event_ids', [])) - {first['event_id']}
    old_context = set(old.get('added_context_event_ids', []))
    base_ids = [eid for eid in first.get('event_ids', [first['event_id']]) if eid not in old_added]
    base_context = [eid for eid in first.get('context_event_ids', []) if eid not in old_context]
    base_context = list(dict.fromkeys(base_context + context_ids(bible, base_ids)))
    result.pop('opening_recap', None)
    if recap:
        added = [eid for eid in recap['event_ids'] if eid not in base_ids]
        ids = added + base_ids
        context = list(dict.fromkeys(base_context + context_ids(bible, ids)))
        context = [eid for eid in context if eid not in ids]
        result['opening_recap'] = {**recap, 'anchor_event_id': first['event_id'], 'added_event_ids': added,
                                   'added_context_event_ids': [eid for eid in context if eid not in base_context]}
    else:
        ids, context = base_ids, base_context
    first.update(event_ids=ids, context_event_ids=context,
                 source_panel_ids=source_ids(bible, ids+context))
    return result


def recap_view(plan):
    recap = plan.get('opening_recap')
    return {k: deepcopy(recap[k]) for k in ('anchor_event_id','previous_chapter','event_ids')} if recap else None
