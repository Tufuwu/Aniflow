"""Best-effort structural compaction for the director's text input only.

The full Bible remains the source for validation and later evidence checks.
No text is classified, summarized, or merged across events here.
"""
from copy import deepcopy

from pipeline.chapter_memory import prompt_context
from pipeline.character_identity import identity_context
from pipeline.prompt_material import character_index
from pipeline.input_compaction import (compact_json as planning_json, compact_event,
                                       compact_characters, compact_identity, omit_empty_lists)


EVENT_FIELDS = (
    'event_id', 'summary', 'character_ids', 'actions', 'dialogue', 'importance',
    'role', 'causes', 'kind', 'time', 'facets', 'status', 'pending_causes',
    'internal_causes', 'reveal_order',
)


def planning_material(bible, settings=None):
    """Return a fresh planning view, leaving all stored data untouched."""
    events = [e for e in bible['events'] if e.get('narrative_eligible', True)]
    return {
        'events': [compact_event(e, EVENT_FIELDS) for e in events],
        'settings': {k: settings[k] for k in ('background', 'characters')} if settings else {},
        'history': prompt_context(bible),
        'characters': omit_empty_lists({
            'characters': compact_characters(character_index(bible.get('characters', []))),
            'relationships': deepcopy(bible.get('relationships', [])),
        }, {'characters', 'relationships'}),
        'identity_positions': compact_identity(identity_context(
            bible, [{'event_id': e['event_id']} for e in events])),
    }
