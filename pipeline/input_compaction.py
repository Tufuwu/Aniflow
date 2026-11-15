"""Shared structural text compaction; no semantic classification or truncation."""
from copy import deepcopy
import json


EVENT_LISTS = {'character_ids', 'actions', 'dialogue', 'causes', 'facets',
               'pending_causes', 'internal_causes', 'evidence'}
CHARACTER_LISTS = {'aliases', 'pending_names', 'name_claims', 'label_claims'}
IDENTITY_LISTS = {'available_by_node_end', 'not_yet_revealed', 'legacy_needs_verification'}


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def omit_empty_lists(value, fields):
    # Only explicit list defaults may disappear. None, False, 0 and unknown
    # states must not be interpreted as absent information.
    return {k: v for k, v in value.items() if not (k in fields and v == [])}


def compact_event(event, fields):
    result = {k: deepcopy(event[k]) for k in fields if k in event}
    facets = result.get('facets')
    if isinstance(facets, list):
        actions = [f['text'] for f in facets
                   if isinstance(f, dict) and f.get('role') == 'action' and 'text' in f]
        # Keep unmatched legacy actions and repeated occurrences in their order.
        if result.get('actions') == actions:
            result.pop('actions', None)
        result['facets'] = [
            {k: v for k, v in f.items() if k != 'fragment_ids'} if isinstance(f, dict) else f
            for f in facets]
    return omit_empty_lists(result, EVENT_LISTS)


def compact_characters(rows):
    result = deepcopy(rows)
    for row in result:
        row.pop('source_panel_ids', None)
    return [omit_empty_lists(row, CHARACTER_LISTS) for row in result]


def compact_identity(rows):
    if any(row.get(key) for row in rows for key in IDENTITY_LISTS):
        return [omit_empty_lists(deepcopy(row), IDENTITY_LISTS) for row in rows]
    return {'status': 'no_confirmed_reveals'}
