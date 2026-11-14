"""Versioned spoken claims, display evidence and measured audio alignment.

Verification context is deliberately separate from facts actually spoken. Missing
or stale metadata stays reviewable; elapsed duration never proves semantics.
"""
import math
import re

from pipeline.story_bible import digest, StoryError
from pipeline.story_context import narrated_ids, context_ids, source_ids
from pipeline.story_model import require


def evidence_hash(bible, segment=None):
    if segment is not None:
        from pipeline.character_identity import identity_source_ids
        ids = list(narrated_ids(segment))
        ids += [eid for eid in context_ids(bible,ids) if eid not in ids]
        sources = set(source_ids(bible,ids))
        sources.update(identity_source_ids(bible,segment['text'],ids))
        current = [p for p in bible['panels'] if not p.get('historical')]
        primary = set(source_ids(bible,narrated_ids(segment)))
        first = next((i for i,p in enumerate(current) if p['source_panel_id'] in primary),0)
        sources.update(p['source_panel_id'] for p in current[max(0,first-2):first])
        return digest({'events':[e for e in bible['events'] if e['event_id'] in ids],
            'characters':bible.get('characters',[]),'relationships':bible.get('relationships',[]),
            'panels':[{k:p.get(k) for k in ('source_panel_id','order','sha256','dialogue','visible_text','uncertainties')}
                      for p in bible['panels'] if p['source_panel_id'] in sources]})
    return digest({'events': bible['events'], 'characters':bible.get('characters', []),
                   'relationships':bible.get('relationships', []), 'panels': [
        [p['source_panel_id'], p['sha256']] for p in bible['panels']]})


def scope_hash(segment):
    return digest({k:segment.get(k) for k in
                   ('segment_id','event_id','event_ids','context_event_ids','source_panel_ids')})


def make_grounding(bible, segment, rows):
    allowed = set(narrated_ids(segment))
    allowed.update(context_ids(bible, list(allowed)))
    events = {e['event_id']: e for e in bible['events']}
    units, cursor = [], 0
    for row in rows:
        claims = row.get('units')
        if not claims:
            return {'status': 'needs_review', 'issues': ['缺少逐句讲述事实与展示来源'],
                    'text_hash': digest(segment['text']), 'units': []}
        require(isinstance(claims, list), '语义依据单元格式无效')
        require(''.join(c.get('text', '') for c in claims if isinstance(c, dict)) == row['text'],
                '语义依据必须逐字覆盖当前句子')
        for claim in claims:
            value = claim.get('text')
            require(isinstance(value, str) and value.strip(), '语义依据文字为空')
            # FactChecker sentences trim inter-sentence whitespace; retain it here.
            start = cursor
            while cursor < len(segment['text']) and segment['text'][cursor].isspace():
                cursor += 1
            value = segment['text'][start:cursor] + value
            require(segment['text'][start:start+len(value)] == value, '语义依据与原稿不符')
            ids = claim.get('event_ids')
            display = claim.get('display_panel_ids')
            background = claim.get('background_event_ids', [])
            require(isinstance(ids, list) and ids and all(isinstance(e, str) and e in allowed for e in ids)
                    and len(set(ids)) == len(ids), '口播事实事件引用越界')
            require(isinstance(background, list) and all(isinstance(e, str) and e in allowed for e in background),
                    '核验背景事件引用越界')
            require(isinstance(display, list) and display and all(isinstance(p, str) for p in display)
                    and len(set(display)) == len(display) and set(display) <= set(source_ids(bible, ids)),
                    '展示来源缺少当前口播事实依据')
            require(all(set(events[e]['source_panel_ids']) & set(display) for e in ids),
                    '正文引用的事件缺少展示依据')
            end = start + len(value)
            units.append({'unit_id': f"{segment['segment_id']}:U{len(units)+1}", 'text': value,
                          'start_char': start, 'end_char': end, 'event_ids': ids,
                          'display_panel_ids': display, 'background_event_ids': background})
            cursor = end
    if segment['text'][cursor:].isspace() and units:
        units[-1]['text'] += segment['text'][cursor:]
        units[-1]['end_char'] = len(segment['text'])
    require(''.join(u['text'] for u in units) == segment['text'], '语义依据未完整覆盖原稿')
    return {'version': 3, 'status': 'mapped', 'text_hash': digest(segment['text']), 'scope_hash':scope_hash(segment),
            'evidence_hash': evidence_hash(bible,segment), 'units': units, 'issues': []}


def grounded_units(bible, segment):
    check = segment.get('fact_check') or {}
    mapping = check.get('grounding') or {}
    if check.get('status') != 'passed' or mapping.get('status') != 'mapped':
        return []
    basis = evidence_hash(bible,segment if mapping.get('version',0)>=3 else None)
    if mapping.get('text_hash') != digest(segment['text']) or mapping.get('evidence_hash') != basis:
        return []
    if mapping.get('scope_hash') != scope_hash(segment):
        return []
    units = mapping.get('units', [])
    require(''.join(u['text'] for u in units) == segment['text'], '语义来源与当前文案不符')
    return units


def lexical(value):
    return ''.join(re.findall(r'[\w\u4e00-\u9fff]', value.lower()))


def attach_tts_boundaries(info, path, narration, result=None):
    """Read only a local sidecar from the very same stream/cache, never synthesize."""
    import json
    from pathlib import Path
    boundary_file = Path(path).with_suffix('.boundaries.json')
    metadata = result if isinstance(result, dict) else {}
    if boundary_file.is_file():
        metadata = json.loads(boundary_file.read_text('utf-8'))
    if metadata.get('text_hash') == digest(narration) and metadata.get('audio_sha256') == info['sha256']:
        info = {**info, 'boundaries': metadata.get('boundaries', []),
                'boundary_text_hash': metadata['text_hash'], 'boundary_audio_sha256': info['sha256']}
    return info


def measured_spans(units, audio):
    """Align on measured unit/word edges only. Never interpolate a semantic edge."""
    if not units:
        return [], '缺少当前正文的逐句展示依据'
    narration = ''.join(u['text'] for u in units)
    duration = audio['duration']
    edges = {0: 0.}
    if audio.get('unit_timings'):
        timings, rate = audio['unit_timings'], audio.get('sample_rate')
        require(type(rate) is int and rate > 0, '语义段采样率无效')
        require(''.join(t['text'] for t in timings) == narration, '配音语义段与正文不符')
        offset, sample = 0, 0
        for t in timings:
            require(t['start_sample'] == sample and sample < t['speech_end_sample'] <= t['end_sample'],
                    '实测语义段边界无效')
            edges[offset] = sample / rate
            offset += len(lexical(t['text']))
            sample = t['end_sample']
        require(abs(sample/rate-duration) < .002, '实测语义段时长不符')
        edges[offset] = duration
    elif audio.get('boundaries'):
        if (audio.get('boundary_text_hash') != digest(narration)
                or audio.get('boundary_audio_sha256') != audio.get('sha256')):
            return [], 'TTS边界与当前文本或音频缓存不符'
        boundaries = audio['boundaries']
        if ''.join(lexical(b['text']) for b in boundaries) != lexical(narration):
            return [], 'TTS边界未完整匹配原文，需检查读音或转写'
        offset, last_end = 0, 0.
        for b in boundaries:
            start, end = b['start'], b['end']
            if not (type(start) in (int, float) and type(end) in (int, float)
                    and math.isfinite(start) and math.isfinite(end) and last_end <= start < end <= duration+.05):
                return [], 'TTS实测边界无效'
            if lexical(b['text']):
                edges[offset] = 0. if offset == 0 else start
                offset += len(lexical(b['text']))
            last_end = end
        edges[offset] = duration
    else:
        return [], '缺少实测TTS边界；整段时长不能证明语义同步'
    spans, offset = [], 0
    for unit in units:
        end = offset + len(lexical(unit['text']))
        if offset not in edges or end not in edges or edges[end] <= edges[offset]:
            return [], '语义切点位于实测边界内部，需更细边界或实测语义段'
        spans.append({**unit, 'start': edges[offset], 'end': edges[end]})
        offset = end
    return spans, None
