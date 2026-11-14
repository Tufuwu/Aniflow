"""Typed annotations over existing character facts, with read-only projections.

Names remain owned by character_identity. No free-text classification, model
call, migration, or second evidence store is performed here.
"""
from copy import deepcopy

from pipeline.story_bible import digest
from pipeline.story_model import require, text

FIELDS = ('identity', 'personality', 'ability', 'state', 'other')
CERTAINTIES = {'explicit', 'observed', 'inferred', 'unknown'}


def attribute_metadata(row):
    attribute = row.get('attribute')
    if attribute is None:
        return None  # Legacy text stays unclassified; never guess by keywords.
    require(isinstance(attribute, dict) and set(attribute) == {'field','key','certainty'},
            '人物属性字段格式无效')
    require(attribute['field'] in FIELDS, '人物属性类型无效；姓名须走原身份核对流程')
    key = text(attribute['key'], '人物属性键')
    require(len(key) <= 80, '人物属性键过长')
    require(attribute['certainty'] in CERTAINTIES, '人物属性确定性无效')
    return {**deepcopy(attribute), 'key':key}


def annotate_fact(row, character_id, origin):
    metadata = attribute_metadata(row)
    return {'character_id':character_id, 'origin':deepcopy(origin),
            'attribute':metadata, 'fact_id':'CF_'+digest({
                'character_id':character_id, 'origin':origin, 'text':row['text'],
                'event_ids':row['event_ids'], 'attribute':metadata})[:24]}


def validate_attributes(additions):
    seen = set()
    for fact in additions:
        attribute = fact.get('attribute') or {}
        if not attribute:
            continue
        key = (attribute['field'],attribute['key'])
        require(key not in seen,'同次人物更新的同一属性只能给出一个当前值')
        seen.add(key)


def merge_character_facts(previous, additions, pending, usable):
    """Replace current values by exact field/key; keep proposals separate."""
    facts, proposals = deepcopy(previous), deepcopy(pending)
    def same_key(left, right):
        a,b = left.get('attribute') or {}, right.get('attribute') or {}
        return bool(a and b and (a['field'],a['key']) == (b['field'],b['key']))
    for fact in additions:
        if not fact.get('attribute'):
            if not any(f['text']==fact['text'] and f.get('event_ids')==fact.get('event_ids') for f in facts):
                facts.append(deepcopy(fact))  # Unclassified legacy compatibility.
        elif fact_status(fact,usable) == 'confirmed':
            facts = [f for f in facts if not same_key(f,fact)]
            proposals = [f for f in proposals if not same_key(f,fact)]
            facts.append(deepcopy(fact))
        else:
            proposals = [f for f in proposals if not same_key(f,fact)]
            proposals.append(deepcopy(fact))
    return facts,proposals


def merge_world_rules(previous, additions, usable):
    rules = deepcopy(previous)
    for row in additions:
        key = row.get('rule_key')
        confirmed = usable and row.get('certainty') in {'explicit','observed'} and row.get('verification',{}).get('verdict') == 'supported'
        if key and confirmed:
            rules = [r for r in rules if r.get('rule_key') != key]
        rules.append(deepcopy(row))
    return rules


def fact_status(fact, usable):
    attr = fact.get('attribute') or {}
    certainty = attr.get('certainty',fact.get('certainty'))
    if certainty == 'unknown':
        return 'unknown'
    if certainty == 'inferred' or (attr.get('field') == 'personality' and certainty != 'explicit'):
        return 'uncertain'
    if not usable or fact.get('verification',{}).get('verdict') != 'supported':
        return 'needs_verification'
    return 'confirmed' if certainty in {'explicit','observed'} else 'unclassified'


def source_view(fact, panels):
    return {'origin':deepcopy(fact.get('origin')), 'event_ids':deepcopy(fact.get('event_ids',[])),
            'panels':[{'source_panel_id':pid, 'sha256':panels.get(pid,{}).get('sha256')}
                      for pid in fact.get('source_panel_ids',[])]}


def character_records(bible, states, usable=False, main_character_ids=None):
    """Four visible profile fields, plus temporary state and unclassified history."""
    panels = {p['source_panel_id']:p for p in bible.get('panels',[])}
    by_id = {s['character_id']:s for s in states}
    main = set(main_character_ids if main_character_ids is not None else by_id)
    records = []
    for char in bible.get('characters',[]):
        if char['character_id'] not in main:
            continue
        claims = [c for c in char.get('name_claims',[])
                  if c.get('name') == char.get('name') and c.get('name_kind','primary') == 'primary']
        known = char.get('name_status') == 'confirmed' and bool(claims)
        name = {'value':char.get('name') if char.get('name_status') != 'unknown' else None,
                'status':'confirmed' if known else 'unknown' if char.get('name_status') == 'unknown' else 'needs_verification',
                'source_panel_ids':list(dict.fromkeys(p for c in claims for p in
                    [c.get('source_panel_id')]+c.get('attribution',{}).get('source_panel_ids',[]) if p)),
                'reveal_panel_id':claims[-1].get('reveal_panel_id') if claims else None,
                'reveal_order':claims[-1].get('reveal_order') if claims else None}
        record = {'character_id':char['character_id'], 'name':name,
                  **{field:[] for field in FIELDS}}
        character_state = by_id.get(char['character_id'],{})
        facts = character_state.get('facts',[])
        for fact in facts+character_state.get('pending_facts',[]):
            attr = fact.get('attribute') or {}
            status = fact_status(fact,usable)
            field = attr.get('field','other')
            record[field].append({'fact_id':fact.get('fact_id'), 'key':attr.get('key'),
                'value':fact['text'], 'status':status, 'certainty':attr.get('certainty'),
                'current':fact in facts,
                'source':source_view(fact,panels), 'reveal_panel_id':fact.get('reveal_panel_id'),
                'reveal_order':fact.get('reveal_order')})
        records.append(record)
    return records


def world_background(bible, memory, usable=False):
    """World overview and main events reuse existing background and summary."""
    panels = {p['source_panel_id']:p for p in bible.get('panels',[])}
    def rows(items):
        return [{'text':f['text'], 'role':f.get('role'), 'certainty':f.get('certainty'),
                 'status':('unknown' if f.get('certainty') == 'unknown' else
                           'uncertain' if f.get('certainty') == 'inferred' else
                           'reviewed' if usable else 'needs_verification'),
                 'source':source_view(f,panels)} for f in items]
    return {'status':'reviewed' if usable else 'needs_verification',
            'overview':rows(memory.get('background',[])), 'main_events':rows(memory.get('summary',[]))}
