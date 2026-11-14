"""Text-only reconstruction from the complete current-chapter analysis.

Raw events and panels remain evidence. Only explicit fragment references connect
the reconstruction to them; titles, keywords and similarity never resolve IDs.
"""
from copy import deepcopy

from pipeline.story_bible import StoryError, digest
from pipeline.story_model import array, prompt_json, require, text, validated_result

VERSION = 1
KINDS = {'current_action', 'world_rule', 'recollection', 'historical_explanation', 'unresolved', 'non_story'}
STATUSES = {'supported', 'uncertain', 'unknown', 'conflicting', 'needs_history'}
ROLES = {'premise', 'goal', 'obstacle', 'choice', 'trigger', 'action', 'reaction', 'result', 'setting', 'unknown'}
OUTPUT_CONTRACT = {'version': VERSION, 'kinds': sorted(KINDS), 'statuses': sorted(STATUSES),
                   'roles': sorted(ROLES), 'coverage': 'every_fragment', 'links': 'explicit_ids_only',
                   'events': {'summary': 'nonempty_string', 'fragment_ids': 'ordered_nonempty_unique_ids',
                              'kind': 'enum', 'time': {'original_time': 'nonempty_string', 'status': 'enum'},
                              'facets': [{'role': 'enum', 'text': 'nonempty_string',
                                          'fragment_ids': 'ordered_nonempty_unique_ids', 'status': 'enum'}]}}
INSTRUCTIONS = '''本话信息整合与事件还原 v1。输入是本次完整图片分析，不是背景或人物摘要；不重新看图。
仅使用本话已有信息，禁止引入其他章节、未来或外部参考。需要历史才能理解的内容保留 needs_history。
extracted_text 是未核验OCR提示，不能据此新增剧情；使用分析后的对白、可见文字、事件和已有来源关联。
尽可能理解并串联零散片段，整理一个或多个清楚连贯的事件；同一连续过程可合并动作、对白、反应和结果。
时间、场景、行动目标或局势明显变化时可拆分；一个原片段含多个过程时可在多个事件引用同一 fragment_id。
允许语义归并和自然承接，不逐句照抄；不补造身份、心理、因果、行动结果，先后不等于因果。
保留人物处境、已有目标、阻碍、选择、转折触发、关键行动、配角行动、反应和结果。
本阶段不筛选口播重点或主要人物，不因重要性低而删信息；后续规划负责选重点和承接，事实整理绑定规划，整稿写作形成口播。
世界规则、危险限制、环境异常、回忆和历史说明保留为相应 kind，写明作用、原作时间及未知限制；不能改写成当前动作。
颜色、服装、姿势、构图保留在原证据，只有帮助辨认或理解事件时进入描述，不照搬取证清单。
未知信息不升级；人物发言只是发言，推测保留为推测。不合并、矛盾或不确定片段也须保留为 unresolved 或相应状态。
所有 fragments 必须在 events.fragment_ids 中出现，不可静默丢弃；无需另写覆盖理由或长篇来源报告。
source_panel_ids、event_id、causes、人物与对白关联由程序继承，不要生成或猜测这些字段。
按原作揭示顺序排列事件及片段；原作时间和揭示位置不同，回忆不提前到本话开头。
summary 是连贯的剧情梳理，不是正式口播；不加吸引观众的套话、评价或未来预告，不另写背景/人物摘要。
返回 {"events":[{"summary":"发生了什么，保留确定性", "fragment_ids":["F00001"],
"kind":"current_action|world_rule|recollection|historical_explanation|unresolved|non_story",
"time":{"original_time":"原作明确时间或未知", "status":"supported|unknown|uncertain|needs_history"},
"facets":[{"role":"premise|goal|obstacle|choice|trigger|action|reaction|result|setting|unknown",
"text":"已有前提、行动、反应、结果、必要设定或未知限制", "fragment_ids":["F00001"],
"status":"supported|uncertain|unknown|conflicting|needs_history"}]}]}。
每个事件至少一个 facet；只填已有环节，不为凑齐链条补造。facets 的引用必须属于该事件，不能把疑点标成 supported。
资料：'''


def chapter_input(bible, source_revision):
    """Full local analysis, with history and old reference summaries excluded."""
    panels = [deepcopy(p) for p in bible['panels'] if not p.get('historical')]
    pids = {p['source_panel_id'] for p in panels}
    require(len(pids) == len(panels), '本话分析来源ID重复')
    originals = bible.get('analysis_events', bible['events'])
    events = [deepcopy(e) for e in originals if not e.get('historical')]
    require(events, '本话没有可整合的分析片段')
    require(len({e['event_id'] for e in events}) == len(events), '原分析事件ID重复')
    pending = []
    local_ids = {e['event_id'] for e in events}
    for e in events:
        require(e.get('source_panel_ids') and set(e['source_panel_ids']) <= pids, '本话分析来源缺失或混入历史')
        # A historical premise stays pending here; no historical content is sent.
        for cause in e.get('causes', []):
            if cause.get('event_id') not in local_ids:
                pending.append({'event_id': e['event_id'], 'status': 'needs_history'})
        e['causes'] = [c for c in e.get('causes', []) if c.get('event_id') in local_ids]
    # Identity recognition is not profile maintenance. Include current evidence
    # for every participant, without importing an older dossier's assertions.
    characters = []
    for c in bible.get('characters', []):
        current = set(c.get('source_panel_ids', [])) & pids
        if not current:
            continue
        if set(c.get('source_panel_ids', [])) <= pids:
            characters.append(deepcopy(c))
            continue
        row = {'character_id': c['character_id'], 'source_panel_ids': [p['source_panel_id'] for p in panels
                if p['source_panel_id'] in current]}
        for key in ('identity_evidence', 'name_claims', 'label_claims', 'pending_names'):
            row[key] = [deepcopy(v) for v in c.get(key, []) if isinstance(v, dict)
                and v.get('source_panel_id') in pids
                and set(v.get('source_panel_ids', v.get('attribution', {}).get('source_panel_ids', []))) <= pids]
        characters.append(row)
    relationships = [deepcopy(r) for r in bible.get('relationships', [])
                     if r.get('source_panel_id') in pids
                     and set(r.get('attribution', {}).get('source_panel_ids', [])) <= pids]
    return {'source_revision': source_revision, 'schema_version': bible.get('schema_version'),
            'analysis_version': bible.get('analysis_version'),
            'panels': panels, 'events': events, 'characters': characters,
            'relationships': relationships, 'pending_history': pending,
            'boundary_review': deepcopy(bible.get('boundary_review', {})),
            'identity_review': {k: deepcopy(v) for k, v in bible.get('identity_review', {}).items()
                                if k in {'status', 'unresolved', 'error'}}}


def fragments_for(analysis):
    """Stable lightweight addresses into the full input, including non-events."""
    panels = analysis['panels']
    positions = {p['source_panel_id']: i for i, p in enumerate(panels)}
    rows = []

    def add(path, value, pids, event_id=None, status=None):
        if isinstance(value, dict):
            status = status or value.get('status') or value.get('certainty')
        status = status if status in STATUSES else 'supported' if status in {None, 'explicit', 'observed', 'confirmed'} else 'uncertain'
        rows.append({'path': path, 'value': deepcopy(value), 'source_panel_ids': pids,
                     'source_event_id': event_id, 'status': status,
                     'position': min(positions[p] for p in pids)})

    for i, e in enumerate(analysis['events']):
        add(f'events/{i}', e, e['source_panel_ids'], e['event_id'])
    for i, p in enumerate(panels):
        for key in ('dialogue', 'visible_text', 'uncertainties', 'scene', 'world_rules', 'environment_anomalies'):
            values = p.get(key, [])
            if not isinstance(values, list):
                if values:
                    add(f'panels/{i}/{key}', values, [p['source_panel_id']])
                continue
            for j, value in enumerate(values):
                add(f'panels/{i}/{key}/{j}', value, [p['source_panel_id']],
                    status='uncertain' if key == 'uncertainties' else None)
    rows.sort(key=lambda r: r['position'])
    return [{**r, 'fragment_id': f'F{i+1:05d}'} for i, r in enumerate(rows)]


def consume(raw, analysis, fragments):
    """Validate coverage, uncertainty and order; inherit all evidence in code."""
    require(isinstance(raw, dict), '事件整合必须是JSON对象')
    rows = array(raw.get('events'), '整合events')
    require(rows, '整合事件不能为空')
    by_id = {f['fragment_id']: f for f in fragments}
    positions = {f['fragment_id']: i for i, f in enumerate(fragments)}
    original = {e['event_id']: e for e in analysis['events']}
    excluded = {p['source_panel_id'] for p in analysis['panels'] if p.get('panel_type', 'story') != 'story'}
    excluded.update(p['source_panel_id'] for p in analysis.get('boundary_review', {}).get('panels', [])
                    if p.get('classification') == 'non_story')
    excluded.update(pid for e in analysis['events'] if e.get('narrative_eligible') is False for pid in e['source_panel_ids'])
    used, events, anchors = set(), [], set()

    def refs(value, allowed):
        require(isinstance(value, list) and value and all(isinstance(x, str) and x in allowed for x in value)
                and len(value) == len(set(value)), '事件整合片段引用无效')
        require(value == sorted(value, key=positions.get), '事件整合片段顺序无效')
        return value

    previous = None
    for row in rows:
        require(isinstance(row, dict), '整合事件格式无效')
        require(not ({'event_id', 'source_panel_ids', 'causes', 'character_ids', 'dialogue'} & row.keys()),
                '整合来源和因果必须由程序继承')
        ids = refs(row.get('fragment_ids'), by_id)
        indexes = {positions[f] for f in ids}
        if previous is not None:
            require(min(previous) <= min(indexes), '整合事件颠倒揭示顺序')
            left, right = previous - indexes, indexes - previous
            require(not left or not right or max(left) < min(right), '整合事件交错揭示顺序')
        previous = indexes
        used.update(ids)
        kind = row.get('kind')
        require(kind in KINDS, '整合事件类型无效')
        when = row.get('time')
        require(isinstance(when, dict) and when.get('status') in STATUSES, '原作时间状态无效')
        text(when.get('original_time'), '原作时间或未知')
        facets = array(row.get('facets'), '事件组成')
        require(facets, '事件组成不能为空')
        for facet in facets:
            require(isinstance(facet, dict) and facet.get('role') in ROLES
                    and facet.get('status') in STATUSES, '事件组成角色或状态无效')
            text(facet.get('text'), '事件组成文字')
            fids = refs(facet.get('fragment_ids'), set(ids))
            require(facet['status'] != 'supported' or all(by_id[f]['status'] == 'supported' for f in fids),
                    '不确定来源不能升级为确认事实')
        require(set(ids) <= {f for part in facets for f in part['fragment_ids']}, '事件组成漏掉片段')
        origins = list(dict.fromkeys(by_id[f]['source_event_id'] for f in ids if by_id[f]['source_event_id']))
        # Keep an original anchor where possible. Split siblings receive fresh IDs.
        anchor = next((eid for eid in origins if eid not in anchors), f'I{len(events)+1:05d}')
        require(anchor not in anchors and (anchor in origins or anchor not in original), '整合事件编号冲突')
        anchors.add(anchor)
        pids = {p for f in ids for p in by_id[f]['source_panel_ids']}
        sources = [p['source_panel_id'] for p in analysis['panels'] if p['source_panel_id'] in pids]
        contributors = [original[eid] for eid in origins]
        require(not (pids & excluded) or pids <= excluded, '不能跨越已核对的剧情与非剧情边界合并事件')
        non_story = pids <= excluded
        require(kind != 'non_story' or non_story, '事件整合不能自行排除本话剧情')
        uncertain = any(by_id[f]['status'] != 'supported' for f in ids) or any(p['status'] != 'supported' for p in facets)
        source_panels = [p for p in analysis['panels'] if p['source_panel_id'] in pids]
        events.append({'event_id': anchor, 'order': len(events), 'summary': text(row.get('summary'), '事件梳理'),
            'fragment_ids': ids, 'source_event_ids': origins, 'source_panel_ids': sources,
            'kind': kind, 'time': deepcopy(when), 'facets': deepcopy(facets),
            'status': 'uncertain' if uncertain or kind == 'unresolved' else 'supported',
            'reveal_panel_id': sources[-1],
            'reveal_order': max(p.get('order', i) for i, p in enumerate(analysis['panels']) if p['source_panel_id'] in pids),
            'character_ids': list(dict.fromkeys([c for e in contributors for c in e.get('character_ids', [])] +
                [d['speaker_id'] for p in source_panels for d in p.get('dialogue', []) if d.get('speaker_id')])),
            'actions': [p['text'] for p in facets if p['role'] == 'action'],
            'dialogue': [deepcopy(d) for p in analysis['panels'] if p['source_panel_id'] in pids for d in p.get('dialogue', [])],
            'evidence': [deepcopy(v) for e in contributors for v in e.get('evidence', [])],
            'importance': max((e.get('importance', 3) for e in contributors), default=3),
            'role': next((e.get('role', 'setup') for e in reversed(contributors)), 'setup'),
            'causes': [],
            'narrative_eligible': not non_story})
        if not events[-1]['narrative_eligible']:
            events[-1]['exclusion_reason'] = '；'.join(e.get('exclusion_reason', '原分析明确非剧情') for e in contributors) or '来源为非剧情内容'
    require(used == set(by_id), '事件整合遗漏本话信息片段')
    # Preserve explicit causes only. A split makes a raw-event target ambiguous;
    # keep the original reference pending instead of guessing the destination.
    for e in events:
        pending = []
        for eid in e['source_event_ids']:
            for cause in original[eid].get('causes', []):
                target = [v for v in events if cause['event_id'] in v['source_event_ids']]
                if len(target) == 1 and target[0]['event_id'] == e['event_id']:
                    e.setdefault('internal_causes', []).append({**deepcopy(cause), 'source_event_id': eid})
                    continue
                if len(target) == 1 and target[0]['order'] < e['order']:
                    inherited = {**deepcopy(cause), 'event_id': target[0]['event_id']}
                    if inherited not in e['causes']:
                        e['causes'].append(inherited)
                else:
                    pending.append({**deepcopy(cause), 'status': 'uncertain'})
        pending.extend({'status': 'needs_history', 'source_event_id': p['event_id']}
                       for p in analysis['pending_history'] if p['event_id'] in e['source_event_ids'])
        if pending:
            e['pending_causes'] = pending
            e['status'] = 'uncertain'
    return events


def resolve_fragment(bible, fragment):
    """Resolve against a source revision by exact IDs, even with history prefixed.

    path indexes address the filtered model input; source IDs address the full
    immutable Bible, whose history may shift those array indexes.
    """
    eid = fragment.get('source_event_id')
    if eid:
        matches = [e for e in bible.get('analysis_events', bible['events']) if e['event_id'] == eid]
        require(len(matches) == 1, '回查原分析事件不存在或重复')
        return deepcopy(matches[0])
    pids = fragment['source_panel_ids']
    matches = [p for p in bible['panels'] if p['source_panel_id'] in pids]
    require(len(pids) == len(matches) == 1, '回查原分析来源不存在或重复')
    value = matches[0]
    for part in fragment['path'].split('/')[2:]:
        value = value[int(part)] if isinstance(value, list) else value[part]
    return deepcopy(value)


def integrate_chapter(store, model, bible, source_revision):
    analysis = chapter_input(bible, source_revision)
    fragments = fragments_for(analysis)
    # Content occurs once, in the complete analysis. The additional catalog is
    # only a lightweight path/index, not another paraphrased reference document.
    prompt = INSTRUCTIONS + prompt_json({'analysis': analysis,
        'fragments': [{k: v for k, v in f.items() if k != 'value'} for f in fragments]})

    def compute():
        feedback = ''
        for attempt in range(3):
            try:
                raw = model.ask('event_integration', prompt + feedback, fresh=attempt > 0)
                return consume(raw, analysis, fragments)
            except StoryError as exc:
                if attempt == 2:
                    raise
                feedback = '\n上次返回未通过程序校验：' + str(exc) + '。保留所有片段，只修正整合JSON。'

    events = validated_result(store, model, 'event_integration',
        {'input': analysis, 'fragments': fragments, 'prompt': prompt, 'output_contract': OUTPUT_CONTRACT}, compute)
    result = deepcopy(bible)
    result.setdefault('analysis_events', deepcopy(bible['events']))
    result['events'] = [deepcopy(e) for e in result['analysis_events'] if e.get('historical')] + events
    result['event_integration'] = {'version': VERSION, 'source_revision': source_revision,
        'input_hash': digest(analysis), 'prompt_hash': digest(prompt), 'output_contract_hash': digest(OUTPUT_CONTRACT),
        'model': model.cache_identity('event_integration') if hasattr(model, 'cache_identity') else {'class': type(model).__name__},
        'identity_review': deepcopy(analysis['identity_review']),
        'fragment_index': [{k: deepcopy(f[k]) for k in ('fragment_id', 'path', 'source_panel_ids', 'source_event_id', 'status')}
                           for f in fragments]}
    result['bible_id'] = digest({k: v for k, v in result.items() if k != 'bible_id'})
    return result
