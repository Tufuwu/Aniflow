"""Editorial image/text associations; offsets count Unicode code points, end exclusive."""
from copy import deepcopy
import re

from pipeline.story_bible import digest
from pipeline.story_model import require

MATCH_TYPES = {'direct', 'transition', 'commentary', 'callback'}


def text_units(narration):
    """Supply copyable boundaries instead of asking a model to count characters."""
    units, start = [], 0
    for match in re.finditer(r'[。！？!?；;，,：:\n]+[”’」』"\s]*', narration):
        end = match.end()
        if narration[start:end].strip():
            units.append({'start_char': start, 'end_char': end, 'text': narration[start:end]})
            start = end
    if start < len(narration):
        if narration[start:].strip():
            units.append({'start_char': start, 'end_char': len(narration), 'text': narration[start:]})
        elif units:
            units[-1].update(end_char=len(narration), text=narration[units[-1]['start_char']:])
    return units


def validate_ranges(narration, ranges, *, require_annotations=False):
    require(isinstance(ranges, list), '选图缺少结构化文字范围 text_ranges')
    result, previous = [], -1
    for item in ranges:
        require(isinstance(item, dict), '选图文字范围格式无效')
        start, end = item.get('start_char'), item.get('end_char')
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(narration),
                '选图文字起止位置越界或无效')
        require(start >= previous, '同一图片的文字范围重复、交叠或顺序错误')
        value = narration[start:end]
        require(value.strip() and item.get('text') == value, '选图文字范围与当前原文不一致')
        normalized = {'start_char': start, 'end_char': end, 'text': value}
        if require_annotations or 'match_type' in item or 'reason' in item:
            require(item.get('match_type') in MATCH_TYPES, '文字范围缺少有效的承接方式')
            require(isinstance(item.get('reason'),str) and item['reason'].strip(), '文字范围缺少承接理由')
            normalized.update(match_type=item['match_type'],reason=item['reason'])
        result.append(normalized)
        previous = end
    return result


def validate_context_ranges(narration, ranges, item):
    """Borrowed images provide editorial continuity, never new event evidence."""
    if not item.get('context_only'):
        return
    directions = {o['direction'] for o in item.get('context_origins',[])}
    require(directions, '相邻镜头缺少来源记录')
    for r in ranges:
        require(r.get('match_type') in {'transition','commentary','callback'} and r.get('reason'),
                '借用相邻镜头须注明过渡、吐槽或回扣及理由')
        if 'previous' not in directions:
            tail = text_units(narration)
            require(r['match_type']=='transition' and r['end_char']==len(narration)
                    and tail and r['start_char']>=tail[-1]['start_char'],
                    '后段镜头仅允许用于段尾过渡，不得提前展示回扣或反应')


def build_mapping(group, version=None):
    version = version if version is not None else (group.get('text_mapping') or {}).get('version',2)
    require(version in (1,2), '未知的图片文字关联版本')
    narration, shots = group['text'], group['selected']
    intervals, edges, issues = [], {0, len(narration)}, []
    for shot in shots:
        ranges = validate_ranges(narration, shot.get('text_ranges', []))
        validate_context_ranges(narration,ranges,shot)
        if not ranges:
            issues.append(f"图片 {shot['panel_id']} 缺少对应原文")
        for item in ranges:
            edges.update((item['start_char'], item['end_char']))
            intervals.append((item['start_char'], item['end_char'], shot['panel_id'], item))
    edges = sorted(edges)
    atoms = []
    for start, end in zip(edges, edges[1:]):
        matches = [(pid,item) for a,b,pid,item in intervals if a <= start and end <= b]
        ids = list(dict.fromkeys(pid for pid,_ in matches))
        atom = {'start_char': start, 'end_char': end, 'text': narration[start:end], 'display_panel_ids': ids}
        if version >= 2:
            atom['associations'] = [{'panel_id':pid,'match_type':item.get('match_type','direct'),
                                     'reason':item.get('reason'), 'start_char':item['start_char'],
                                     'end_char':item['end_char']} for pid,item in matches]
        atoms.append(atom)
    for i, atom in enumerate(atoms):
        if atom['display_panel_ids']:
            continue
        if atom['text'].strip():
            issues.append(f"原文 [{atom['start_char']}, {atom['end_char']}) 没有关联图片")
        else:
            # Whitespace belongs to adjacent speech, without assigning new spoken words.
            neighbor = next((a for a in reversed(atoms[:i]) if a['display_panel_ids']), None)
            neighbor = neighbor or next((a for a in atoms[i+1:] if a['display_panel_ids']), None)
            atom['display_panel_ids'] = list(neighbor['display_panel_ids']) if neighbor else []
            if version >= 2:
                atom['associations'] = deepcopy(neighbor['associations']) if neighbor else []
    units = []
    for atom in atoms:
        if units and units[-1]['display_panel_ids'] == atom['display_panel_ids']:
            units[-1].update(end_char=atom['end_char'], text=units[-1]['text']+atom['text'])
            if version >= 2:
                for association in atom['associations']:
                    if association not in units[-1]['associations']:
                        units[-1]['associations'].append(association)
        else:
            units.append({**atom, 'unit_id': f"{group['narration_id']}:R{len(units)+1}"})
    fields = ('panel_id','text_ranges','context_only','context_origins') if version >= 2 else ('panel_id','text_ranges')
    basis = {'text': narration, 'shots': [{k:s.get(k) for k in fields} for s in shots]}
    return {'version': version, 'offset_unit': 'unicode_codepoint', 'end_exclusive': True,
            'text_hash': digest(narration), 'basis_hash': digest(basis),
            'status': 'needs_review' if issues else 'mapped', 'issues': issues, 'units': units}


def attach_mapping(group):
    if any('text_ranges' in r for r in group['decisions']):
        by_id = {r['panel_id']: r for r in group['decisions']}
        for shot in group['selected']:
            row = by_id[shot['panel_id']]
            shot['text_ranges'] = deepcopy(row.get('text_ranges', []))
            for key in ('context_only','context_origins'):
                if key in row:
                    shot[key] = deepcopy(row[key])
        group['text_mapping'] = build_mapping(group)
    return group


def mapped_text_units(group):
    if 'text_mapping' not in group:
        return []
    current = build_mapping(group)
    require(group['text_mapping'] == current, '图片与文字范围关联已过期，请重新审核')
    require(current['status'] == 'mapped', '图片与文字范围尚未完整关联：'+'；'.join(current['issues']))
    return current['units']
