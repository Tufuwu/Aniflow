"""Manual script acceptance, bound to the reviewed text and source scope."""
from pipeline.semantic_grounding import evidence_hash, grounded_units, scope_hash
from pipeline.story_bible import digest
from pipeline.story_context import narrated_ids, source_ids
from pipeline.story_model import require


def approval_basis(bible, segment):
    return digest({'text': segment['text'], 'scope': scope_hash(segment),
                   'evidence': evidence_hash(bible, segment)})


def approve_segment(bible, segment):
    require(isinstance(segment.get('text'), str) and segment['text'].strip(), '旁白不能为空')
    ids = narrated_ids(segment)
    require(source_ids(bible, ids), '文案缺少事件来源')
    panels = {p['source_panel_id'] for p in bible['panels']}
    require(set(segment.get('source_panel_ids', [])) <= panels, '文案引用未知原图')
    segment['human_review'] = {'version': 1, 'mode': 'manual',
                               'basis_hash': approval_basis(bible, segment)}
    segment['review_status'] = 'approved'


def manually_approved(bible, segment):
    review = segment.get('human_review') or {}
    return (segment.get('review_status') == 'approved' and review.get('version') == 1
            and review.get('mode') == 'manual'
            and review.get('basis_hash') == approval_basis(bible, segment))


def segment_approved(bible, plan, segment):
    if segment.get('human_review'):
        return manually_approved(bible, segment)
    # Historical approvals keep their original verification requirements.
    return (segment.get('review_status') == 'approved'
            and (segment.get('fact_check') or {}).get('status') == 'passed'
            and (plan.get('version', 0) < 5 or bool(grounded_units(bible, segment))))
