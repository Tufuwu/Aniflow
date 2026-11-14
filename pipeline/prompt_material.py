"""Purpose-specific views. Full evidence and reports remain in immutable revisions."""
from copy import deepcopy
import re


def script_text(segments):
    keys = ('segment_id', 'event_id', 'event_ids', 'context_event_ids', 'source_panel_ids', 'section', 'text')
    return [{k:deepcopy(s[k]) for k in keys if k in s} for s in segments]


def character_index(characters):
    keys = ('character_id','name','canonical_name','name_status','gender','description','aliases','pending_names',
            'name_claims','label_claims','source_panel_ids')
    return [{k:deepcopy(c[k]) for k in keys if k in c} for c in characters]


def necessary_events(bible, beats):
    ids = {eid for b in beats for eid in b.get('event_ids',[b['event_id']])+b.get('context_event_ids',[])}
    return [deepcopy(e) for e in bible['events'] if e['event_id'] in ids]


def text_evidence(bible, beats, facts=None):
    """Text editors need facts and original lines once, not all visual forensics."""
    events = necessary_events(bible,beats)
    if facts is not None:
        # The independently prepared sheet is the text editor's basis. Event
        # IDs resolve to full local evidence; raw OCR and recognition doubts
        # belong to the later image check, not every candidate's text review.
        return {'fact_sheet':fact_sheet_view(facts),
                'events':[{'event_id':e['event_id']} for e in events]}
    pids = {p for e in events for p in e['source_panel_ids']}
    keys = ('event_id','order','summary','character_ids','actions','causes','source_panel_ids',
            'kind','time','facets','status','pending_causes','internal_causes','reveal_panel_id','reveal_order')
    return {'events':[{k:deepcopy(e[k]) for k in keys if k in e}
            for e in events],
            'sources':[{k:deepcopy(p[k]) for k in
             ('source_panel_id','dialogue','visible_text','uncertainties','identity_clues') if k in p}
             for p in bible['panels'] if p['source_panel_id'] in pids]}


def current_fact_sheet(bible, plan, script):
    from pipeline.story_bible import digest
    if script.get('fact_sheet_basis')==digest({'bible':bible,'plan':plan}):
        return script.get('fact_sheet')


def revision_feedback(feedback):
    if not feedback:
        return feedback
    if not isinstance(feedback,dict):
        return str(feedback)
    return {k:deepcopy(feedback[k]) for k in ('passed','issues','blocking_issues','error') if k in feedback}


def context_catalog(bible):
    """Address reference units without guessing equivalence from titles or words.

    This is a view of existing memory/notes, not a second memory store. Unknown
    provenance is explicit; only the digest decides relevance to its own plan.
    """
    from pipeline.chapter_memory import prompt_context
    prior = prompt_context(bible) if bible.get('event_integration') else bible.get('prior_context', {})
    events = {e['event_id']: e for e in bible['events']}
    panels = {p['source_panel_id']: p for p in bible['panels']}
    rows = []

    def add(value, path, status, owner=None, revision=None):
        if isinstance(value, str):
            value = {'text': value}
        if not isinstance(value, dict) or not value.get('text', '').strip():
            return
        refs = value.get('event_ids', [])
        source = {'path': path, 'revision': revision, 'event_ids': deepcopy(refs)}
        source['events'] = [{
            'event_id': eid, 'origin': deepcopy(events[eid].get('history_origin', {
                'chapter': prior.get('chapter'), 'revision': bible.get('analysis_revision')})),
            'panels': [{'source_panel_id': pid, 'sha256': panels.get(pid, {}).get('sha256'),
                        'origin': deepcopy(panels.get(pid, {}).get('history_origin'))}
                       for pid in events[eid].get('source_panel_ids', [])]}
            for eid in refs if eid in events]
        # Explicit source metadata must survive even when it conflicts with the
        # local event. Never deduplicate two versions merely sharing an ID.
        for key in ('source_revision', 'source_version', 'source_ref', 'history_origin', 'source_panel_ids',
                    'unit_ids', 'chapter', 'sha256', 'certainty'):
            if key in value:
                source[key] = deepcopy(value[key])
        if status == 'unverified_reference' and 'status' in value:
            source['reported_status'] = deepcopy(value['status'])
        attribute = value.get('attribute') or {}
        if attribute:
            source['attribute'] = deepcopy(attribute)
            source['certainty'] = attribute.get('certainty')
            for key in ('fact_id','reveal_panel_id','reveal_order','verification','origin'):
                if key in value:
                    source[key] = deepcopy(value[key])
            if status != 'unverified_reference':
                from pipeline.character_profile import fact_status
                status = fact_status(value, status == 'reviewed_history')
                if status == 'confirmed':
                    status = 'reviewed_history'
        elif status != 'unverified_reference' and value.get('certainty') in {'inferred','unknown'}:
            status = 'uncertain' if value['certainty'] == 'inferred' else 'unknown'
        rows.append({'context_id': f'X{len(rows)+1:04d}', 'text': value['text'],
                     'kind': path.split('/')[0],
                     'status': 'unverified_reference' if status == 'unverified_reference' else
                         status if attribute or value.get('certainty') in {'inferred','unknown'} else value.get('status', status),
                     'character_id': owner, 'source': source})

    pin = prior.get('previous_chapter', {})
    status = 'reviewed_history' if prior.get('status') == 'available' else 'needs_verification'
    for key in ('summary', 'background', 'open_threads'):
        for i, row in enumerate(prior.get(key, [])):
            add(row, f'{key}/{i}', status, revision=pin.get('revision'))
    for i, char in enumerate(prior.get('character_states', [])):
        for field in ('facts','pending_facts'):
            for j, row in enumerate(char.get(field, [])):
                add(row, f'character_states/{i}/{field}/{j}', status, char.get('character_id'), pin.get('revision'))
    notes = prior.get('series_notes') or {}
    for key in ('background', 'characters', 'unknowns'):
        value = notes.get(key, [])
        if isinstance(value, str):
            # Offsets resolve back to the immutable original, including short
            # notes which never passed through the optional reference brief.
            for match in re.finditer(r'[^\n]+', value):
                add({'text': match.group(), 'source_ref': notes.get('source_ref'),
                     'unit_ids': [f'{key}:{match.start()}:{match.end()}']},
                    f'series_notes/{key}/{match.start()}:{match.end()}', 'unverified_reference',
                    revision=notes.get('revision'))
        else:
            for i, row in enumerate(value):
                add({**row, 'source_ref': notes.get('source_ref')}, f'series_notes/{key}/{i}',
                    'unverified_reference', revision=notes.get('revision'))
    return rows


def source_covered(row, bible, nodes):
    """Prove coverage by event identity/version/hash, never semantic similarity."""
    source = row['source']
    refs = source['event_ids']
    covered = {eid for n in nodes for f in n['facts'] for eid in f['event_ids']}
    if not refs or not set(refs) <= covered or len(source['events']) != len(refs):
        return False
    if row['status'] not in {'reviewed_history', 'confirmed', 'confirmed_fact'}:
        return False
    if source.get('certainty') not in (None, 'confirmed', 'supported', 'explicit', 'observed'):
        return False
    origins = [e['origin'] for e in source['events']]
    if 'chapter' in source and any(o.get('chapter') != source['chapter'] for o in origins):
        return False
    if 'source_revision' in source and any(o.get('revision') != source['source_revision'] for o in origins):
        return False
    if 'source_version' in source and any(o.get('revision') != source['source_version'] for o in origins):
        return False
    if 'history_origin' in source and any(any(o.get(k) != v for k,v in source['history_origin'].items()) for o in origins):
        return False
    if 'source_panel_ids' in source and set(source['source_panel_ids']) != {
            p['source_panel_id'] for e in source['events'] for p in e['panels']}:
        return False
    if 'sha256' in source and any(p['sha256'] != source['sha256'] for e in source['events'] for p in e['panels']):
        return False
    return True


def bind_context_usage(bible, nodes):
    """Validate the digest's per-node decisions; absent/ambiguous stays visible.

    Kept on the existing nodes so candidates share the exact same plan-bound
    fact sheet. No extra model phase, context cache, or mutable source store.
    """
    from pipeline.story_model import require, text
    catalog = {r['context_id']: r for r in context_catalog(bible)}
    seen = set()
    for node in nodes:
        decisions = node.get('context_usage', [])
        require(isinstance(decisions, list), '前情用途格式无效')
        for decision in decisions:
            cid = decision.get('context_id') if isinstance(decision, dict) else None
            require(cid in catalog and cid not in seen, '前情来源未知或重复')
            seen.add(cid)
            require(decision.get('usage') in {'supplement', 'covered', 'irrelevant', 'uncertain'}, '前情用途无效')
            text(decision.get('reason'), '前情用途依据')
            proven = source_covered(catalog[cid], bible, nodes)
            if not proven and decision['usage'] == 'covered':
                decision.update(usage='uncertain', reason='来源不足以确认重复；原判断：'+decision['reason'])
    for cid, row in catalog.items():
        if cid not in seen:
            proven = (source_covered(row, bible, nodes) and
                      any(row['text'] == f['text'] for n in nodes for f in n['facts']))
            nodes[0].setdefault('context_usage', []).append({
                'context_id': cid, 'usage': 'covered' if proven else 'uncertain',
                'reason': '同一来源已在事实简表' if proven else '尚无本规划的资料取舍；保留待核对，不能当作事实'})
    return nodes


def fact_sheet_view(nodes, spoken=False):
    return [{k:deepcopy(v) for k,v in n.items() if k != 'context_usage' and k != 'facts'} |
            {'facts':[deepcopy(f) for f in n['facts'] if not spoken or f.get('usage') != 'verification_only']}
            for n in nodes]


def supplemental_context(bible, nodes=None):
    """Only needed nonduplicate context; unresolved material is clearly marked."""
    nodes = bind_context_usage(bible, deepcopy(nodes)) if nodes else []
    decisions = {d['context_id']: d for n in nodes for d in n.get('context_usage', [])}
    result = []
    for row in context_catalog(bible):
        decision = decisions.get(row['context_id'], {'usage':'uncertain', 'reason':'旧稿尚无当前规划的取舍记录'})
        if decision['usage'] in {'supplement', 'uncertain'}:
            result.append({**row, 'selection':decision['usage'], 'reason':decision['reason']})
    return result


def compact_identity(bible, beats):
    from pipeline.character_identity import identity_context
    # Preserve availability, unknown status and reveal positions. Full quotes,
    # attribution, recognition attempts and legacy dossiers resolve locally.
    keys = ('character_id','name','label','relation','character_ids','name_status','gender',
            'reveal_order','reveal_panel_id','source_panel_id','source_panel_ids','status')
    return [{k:([ {field:deepcopy(row[field]) for field in keys if field in row}
                    for row in v] if isinstance(v,list) else v) for k,v in node.items()}
            for node in identity_context(bible, beats)]


def editor_material(bible, plan, nodes=None, beats=None):
    """Shared text inputs, deliberately independent of image-check inputs."""
    beats = beats if beats is not None else plan['beats']
    events = necessary_events(bible, beats)
    ids = {cid for e in events for cid in e.get('character_ids', [])}
    supplements = supplemental_context(bible, nodes)
    ids.update(r['character_id'] for r in supplements if r['character_id'])
    keys = ('character_id','name','canonical_name','name_status','gender','aliases')
    characters = []
    for char in bible.get('characters', []):
        if char['character_id'] not in ids and not char.get('pending_names'):
            continue
        row = {k:deepcopy(char[k]) for k in keys if k in char}
        if char.get('pending_names'):
            row['pending_names'] = [{k:deepcopy(c[k]) for k in ('name','source_panel_id','status') if k in c}
                                    if isinstance(c,dict) else c for c in char['pending_names']]
        characters.append(row)
    beat_keys = ('event_id','event_ids','context_event_ids','section','title','viewer_question','payoff','reason')
    planned = {b['event_id']: b for b in plan['beats']}
    # User references are not a fabricated previous chapter. In particular,
    # Chapter 1 keeps its recap empty even with necessary external references.
    history = [r for r in supplements if bible.get('prior_context', {}).get('previous_chapter')
               and not r['source']['path'].startswith('series_notes/')]
    references = [r for r in supplements if r not in history]
    return {'plan':{**{k:deepcopy(plan[k]) for k in ('style','essential_event_ids','ending_rule') if k in plan},
                    'beats':[{k:deepcopy(v) for k,v in {**planned.get(b['event_id'],{}),**b}.items()
                              if k in beat_keys} for b in beats]},
            'characters':characters, 'identity_availability':compact_identity(bible, beats),
            'prior_context':history, 'reference_context':references}
