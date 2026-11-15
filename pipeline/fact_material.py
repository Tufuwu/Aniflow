"""Compact source inputs for direct writing and an optional legacy digest."""
from copy import deepcopy

from pipeline.character_identity import identity_context
from pipeline.input_compaction import (compact_event, compact_characters,
                                       compact_identity, omit_empty_lists)
from pipeline.prompt_material import character_index, necessary_events, context_catalog, editor_material
from pipeline.story_bible import digest
from pipeline.story_model import require
from pipeline.opening_recap import with_opening_recap, recap_view


FACT_CONTRACT_VERSION = 4
SOURCE_MATERIAL_VERSION = 2
EVENT_FIELDS = ('event_id', 'summary', 'character_ids', 'actions', 'dialogue', 'causes',
                'kind', 'time', 'facets', 'status', 'pending_causes', 'internal_causes',
                'reveal_order', 'evidence', 'historical', 'history_origin')
PLAN_FIELDS = ('style', 'essential_event_ids', 'ending_rule', 'milestones')
BEAT_FIELDS = ('event_id', 'event_ids', 'context_event_ids', 'section', 'title',
               'viewer_question', 'payoff', 'reason')


def evidence_view(event, panels):
    """Omit only plain quote copies proven against the same original source.

    Unique descriptions, different types, uncertain annotations and conflicting
    sources stay. Quotes are never normalized, merged across events or truncated.
    """
    result = []
    for item in event.get('evidence', []):
        if not isinstance(item, dict):
            result.append(deepcopy(item))
            continue
        plain = set(item) <= {'kind', 'detail', 'source_panel_id'}
        pid = item.get('source_panel_id')
        local = pid in event.get('source_panel_ids', []) and pid in panels
        if plain and local and item.get('kind') == 'dialogue':
            originals = [line for line in panels[pid].get('dialogue', [])
                         if line.get('text') == item.get('detail')]
            if originals and all(line in event.get('dialogue', []) for line in originals):
                continue
        row = deepcopy(item)
        # Unmatched dialogue can indicate conflicting text or attribution;
        # preserve its explicit source as well as its unique wording.
        if plain and local and item.get('kind') != 'dialogue':
            row.pop('source_panel_id', None)
        result.append(row)
    return result


def fact_events(bible, beats):
    """Compact source events without making narration or verification decisions."""
    panels = {p['source_panel_id']: p for p in bible['panels']}
    events = []
    for event in necessary_events(bible, beats):
        row = compact_event(event, EVENT_FIELDS)
        if 'evidence' in event:
            row['evidence'] = evidence_view(event, panels)
        events.append(omit_empty_lists(row, {'evidence'}))
    return events


def fact_material(bible, plan):
    """Optional model digest input, including non-eligible historical context."""
    material = {
        'plan': {**{k: deepcopy(plan[k]) for k in PLAN_FIELDS if k in plan},
                 'beats': [omit_empty_lists({k: deepcopy(beat[k]) for k in BEAT_FIELDS if k in beat},
                                           {'context_event_ids'}) for beat in plan['beats']]},
        'events': fact_events(bible, plan['beats']),
        'characters': compact_characters(character_index(bible.get('characters', []))),
        'identity_availability': compact_identity(identity_context(bible, plan['beats'])),
        # Keep reference provenance intact, including conflicting versions and
        # hashes. It is input to the existing context_usage decisions.
        'context_catalog': context_catalog(bible),
    }
    if plan.get('opening_recap'):
        material['plan']['opening_recap'] = recap_view(plan)
    return material


def source_material_basis(bible, plan):
    """Track input version and dependencies, never a correctness rating."""
    plan = with_opening_recap(bible, plan)
    return {'version': SOURCE_MATERIAL_VERSION,
            'hash': digest({'version': SOURCE_MATERIAL_VERSION, 'bible': bible, 'plan': plan})}


def writing_material(bible, plan):
    """Package the director's route and raw source facts for one writing call."""
    plan = with_opening_recap(bible, plan)
    selected = {eid for beat in plan['beats'] for eid in beat.get('event_ids', [beat['event_id']])}
    context = {eid for beat in plan['beats'] for eid in beat.get('context_event_ids', [])}
    require(selected | context <= {e['event_id'] for e in bible['events']}, '写作规划引用未知事件')
    require(set(plan.get('essential_event_ids', [])) <= selected, '写作规划缺少必保事件')
    source = fact_material(bible, plan)
    continuity = editor_material(bible, plan)
    return {k: v for k, v in source.items() if k != 'context_catalog'} | {
        'prior_context': continuity['prior_context'],
        'reference_context': continuity['reference_context'],
    }
