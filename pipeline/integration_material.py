"""Lossless semantic input views; raw provenance remains in the chapter store."""
from copy import deepcopy
import json

from pipeline.prompt_material import character_index

INPUT_VIEW_VERSION = 2
FRAGMENT_COLUMNS = ['id', 'kind', 'source', 'status', 'content']


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def fragment_kind(fragment):
    return 'event' if fragment['source_event_id'] else fragment['path'].split('/')[2]


def model_material(analysis, fragments, selected_ids=None):
    """One semantic occurrence per fragment, addressed by its existing ID.

    Events inherit dialogue only when it is exactly the same ordered data on
    their explicitly referenced panels. Conflicting text or attribution stays.
    Repair requests include those dialogue fragments as read-only context.
    """
    panels = {p['source_panel_id']: p for p in analysis['panels']}
    chosen = {f['fragment_id'] for f in fragments} if selected_ids is None else set(selected_ids)
    dialogue_by_panel = {}
    for f in fragments:
        if fragment_kind(f) == 'dialogue':
            dialogue_by_panel.setdefault(f['source_panel_ids'][0], []).append(f['fragment_id'])
    event_values = {}
    for f in fragments:
        if f['fragment_id'] not in chosen or fragment_kind(f) != 'event':
            continue
        value = deepcopy(f['value'])
        # The exact event ID already occupies the source column; table order
        # and source panel order retain reading order without another index.
        value.pop('event_id', None)
        canonical = [line for p in analysis['panels'] if p['source_panel_id'] in f['source_panel_ids']
                     for line in p.get('dialogue', [])]
        if 'dialogue' in value and value['dialogue'] == canonical:
            value.pop('dialogue')
            for pid in f['source_panel_ids']:
                chosen.update(dialogue_by_panel.get(pid, []))
        event_values[f['fragment_id']] = value
    selected = [f for f in fragments if f['fragment_id'] in chosen]
    pids = {pid for f in selected for pid in f['source_panel_ids']}
    event_ids = {f['source_event_id'] for f in selected if f['source_event_id']}
    character_ids = {cid for f in selected if f['source_event_id']
                     for cid in f['value'].get('character_ids', [])}
    character_ids.update(d.get('speaker_id') for pid in pids for d in panels[pid].get('dialogue', []) if d.get('speaker_id'))
    characters = [c for c in analysis['characters'] if selected_ids is None or c['character_id'] in character_ids
                  or set(c.get('source_panel_ids', [])) & pids]
    return {
        'input_view_version': INPUT_VIEW_VERSION,
        'source_revision': analysis['source_revision'],
        'analysis_version': analysis.get('analysis_version'),
        'panels': [{k: deepcopy(p[k]) for k in ('source_panel_id', 'order', 'panel_type', 'boundary_uncertain', 'identity_clues') if k in p}
                   for p in analysis['panels'] if p['source_panel_id'] in pids],
        'characters': character_index(characters),
        'relationships': [deepcopy(r) for r in analysis['relationships']
                          if selected_ids is None or r.get('source_panel_id') in pids],
        'identity_review': {k: deepcopy(v) for k, v in analysis.get('identity_review', {}).items()
                            if k in {'status', 'unresolved'}},
        'boundary_review': {'panels': [{k: deepcopy(p[k]) for k in ('source_panel_id', 'classification', 'reason') if k in p}
                                      for p in analysis.get('boundary_review', {}).get('panels', [])]},
        'pending_history': [deepcopy(r) for r in analysis['pending_history'] if r['event_id'] in event_ids],
        'fragment_columns': FRAGMENT_COLUMNS,
        'fragments': [[f['fragment_id'], fragment_kind(f), f['source_event_id'] or f['source_panel_ids'][0], f['status'],
                       event_values[f['fragment_id']] if f['source_event_id'] else deepcopy(f['value'])]
                      for f in selected],
    }
