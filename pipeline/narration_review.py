"""Completion tracking for direct interest polishing and read-only excerpt audits."""
from difflib import SequenceMatcher
import re

from pipeline.story_bible import digest
from pipeline.narration_polish import polish_options, script_view


ENGAGEMENT_POLISH_VERSION = 1
ENGAGEMENT_POLISH_PROMPT_VERSION = 5
ENGAGEMENT_POLISH_STYLE = '生动有趣的故事解说'


def polish_basis(bible, plan, segments, options=None):
    # Approving unchanged prose does not undo a completed edit. The actual model
    # request still includes review-derived locks and has its own transport key.
    script = [{**row,'locked':source.get('locked',False)}
              for row,source in zip(script_view(segments),segments)]
    return digest({'polish_contract':ENGAGEMENT_POLISH_VERSION,
                   'prompt_version':ENGAGEMENT_POLISH_PROMPT_VERSION,
                   'style':ENGAGEMENT_POLISH_STYLE,
                   'options':polish_options(options),
                   'script':script})


def polish_ready(record, bible, plan, segments, options=None, model_identity=None):
    operation = (record or {}).get('engagement_polish') or {}
    return bool(operation.get('status')=='completed'
                and operation.get('version')==ENGAGEMENT_POLISH_VERSION
                and (model_identity is None or operation.get('model_basis')==digest(model_identity))
                and record.get('basis_hash')==polish_basis(bible,plan,segments,
                    options if options is not None else operation.get('options')))


def repeat_candidates(segments):
    candidates = []
    for j, later in enumerate(segments):
        for earlier in segments[:j]:
            # Keep the original excerpts/offsets so a model must inspect the
            # actual duplication instead of comparing event IDs or keywords.
            matcher = SequenceMatcher(None, earlier['text'], later['text'], autojunk=False)
            for a,b,size in matcher.get_matching_blocks():
                excerpt = later['text'][b:b+size]
                if len(re.sub(r'\W', '', excerpt)) < 16:
                    continue
                candidates.append({'repeat_id':digest([earlier['event_id'],later['event_id'],a,b,size])[:16],
                                   'earlier_event_id':earlier['event_id'], 'event_id':later['event_id'],
                                   'earlier_start':a, 'start':b, 'text':excerpt})
    return candidates
