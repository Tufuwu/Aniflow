"""Identity evidence and reveal positions within the existing chapter workflow.

These checks validate references, not a model's visual interpretation. Unknown
claims remain in the roster and never become narration merely through presence.
"""
from copy import deepcopy

from pipeline.story_model import require, text


def confirm_claim(item, sources, character_id, value):
    """Verbatim text and attribution are two separate mandatory conditions."""
    pid = item.get('source_panel_id')
    require(pid in sources, '身份核对来源无效')
    quote = text(item.get('quote'), '身份证据原文')
    panel = sources[pid]
    kind = item.get('kind', 'dialogue')
    originals = ([d['text'] for d in panel.get('dialogue', [])] if kind == 'dialogue'
                 else panel.get('visible_text', []) if kind == 'visual_text' else [])
    require(any(quote in original for original in originals), '身份依据必须逐字来自原文')
    require(value in quote, '姓名或关系没有出现在身份证据中')
    attribution = item.get('attribution')
    require(isinstance(attribution, dict) and attribution.get('character_id') == character_id,
            '身份缺少独立的人物归属证据')
    require(attribution.get('method') in {'self_introduction', 'direct_address', 'name_tag',
                                        'explicit_caption', 'dialogue_context'}, '人物归属方式无效')
    text(attribution.get('basis'), '人物归属依据')
    refs = attribution.get('source_panel_ids')
    require(isinstance(refs, list) and refs and pid in refs
            and all(isinstance(p, str) and p in sources for p in refs), '人物归属来源无效')
    # The latest supporting picture is when this association becomes available,
    # even if the literal name was mentioned much earlier.
    reveal = max(refs, key=lambda p: sources[p]['order'])
    return {**deepcopy(item), 'source_panel_id': pid, 'reveal_panel_id': reveal,
            'reveal_order': sources[reveal]['order']}


def record_name(character, claim):
    """A nickname is a separately evidenced claim, never a canonical overwrite."""
    name, kind = claim['name'], claim.get('name_kind', 'primary')
    require(kind in {'primary','alias'}, '姓名类型无效')
    if kind == 'primary':
        require(character.get('name_status')=='unknown' or character['name'].startswith('未知')
                or character['name']==name, '不能覆盖已经确认的人名')
        character.update(name=name, name_status='confirmed')
    else:
        character['aliases'] = list(dict.fromkeys(character.get('aliases', [])+[name]))
    if claim not in character.setdefault('name_claims', []):
        character['name_claims'].append(deepcopy(claim))


def identity_context(bible, beats):
    """Expose per-node availability alongside internal knowledge to all editors."""
    panel_order = {p['source_panel_id']:p.get('order',i) for i,p in enumerate(bible['panels'])}
    events = {e['event_id']: e for e in bible['events']}
    result = []
    for beat in beats:
        ids = beat.get('event_ids', [beat['event_id']])
        positions = [panel_order[p] for eid in ids for p in events[eid]['source_panel_ids']]
        # End is an upper bound only. Within a multi-event beat the original
        # reveal still precedes use; the model checks the precise sentence order.
        end = max(positions)
        known, later, unverified = [], [], []
        for char in bible.get('characters', []):
            claims = char.get('name_claims', []) + char.get('label_claims', [])
            if not claims and char.get('name_status') != 'unknown':
                unverified.append(deepcopy(char))  # Legacy data is readable, not promoted.
            for claim in claims:
                row = {'character_id': char['character_id'], **deepcopy(claim)}
                (known if claim['reveal_order'] <= end else later).append(row)
        for relation in bible.get('relationships', []):
            if 'reveal_order' not in relation:
                unverified.append(deepcopy(relation))
            else:
                (known if relation['reveal_order'] <= end else later).append(deepcopy(relation))
        result.append({'event_id': beat['event_id'], 'available_by_node_end': known,
                       'not_yet_revealed': later, 'legacy_needs_verification': unverified})
    return result


def identity_source_ids(bible, narration, event_ids):
    """Original name/relationship proof is verification context, not a shot list."""
    panels = {p['source_panel_id']:p for p in bible['panels']}
    positions = {p['source_panel_id']:p.get('order',i) for i,p in enumerate(bible['panels'])}
    events = {e['event_id']:e for e in bible['events']}
    end = max(positions[p] for eid in event_ids for p in events[eid]['source_panel_ids'])
    found = set()
    claims = [n for c in bible.get('characters', []) for n in c.get('name_claims', [])]
    claims += bible.get('relationships', [])
    for claim in claims:
        value = claim.get('name') or claim.get('relation')
        if not value or value not in narration or claim.get('reveal_order',end+1)>end:
            continue
        refs = [claim.get('source_panel_id')]+claim.get('attribution', {}).get('source_panel_ids', [])
        found.update(p for p in refs if p in panels)
    return [p for p in panels if p in found]


def premature_name_issues(bible, segments):
    """Catch definite early leaks; ambiguous mentions still need model review."""
    contexts = identity_context(bible, segments)
    events = {e['event_id']: e for e in bible['events']}
    panels = {p['source_panel_id']: p for p in bible['panels']}
    issues = []
    for segment, context in zip(segments, contexts):
        allowed_panels = {p for eid in segment.get('event_ids', [segment['event_id']])
                          + segment.get('context_event_ids', []) for p in events[eid]['source_panel_ids']}
        originals = [d['text'] for p in allowed_panels for d in panels[p].get('dialogue', [])]
        originals += [t for p in allowed_panels for t in panels[p].get('visible_text', [])]
        available_names = {c.get('name') for c in context['available_by_node_end']}
        for claim in context['not_yet_revealed']:
            name = claim.get('name')
            if name and name not in available_names and name in segment['text'] and not any(name in t for t in originals):
                issues.append({'event_id': segment['event_id'], 'aspect': 'references',
                               'detail': '姓名提前于人物对应揭晓：'+name,
                               'suggestion': '依据当前已知信息使用稳定称呼，保留后文姓名揭晓位置'})
    return issues
