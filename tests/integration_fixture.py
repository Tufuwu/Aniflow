"""Synthetic one-to-one response for existing workflow tests, never a model."""
import json


def integration_response(prompt):
    data = json.JSONDecoder().raw_decode(prompt.split('资料：', 1)[1])[0]
    if 'fragment_columns' in data:
        fragments = [dict(zip(data['fragment_columns'], row)) for row in data['fragments']]
        events, used = [], set()
        for source in [f for f in fragments if f['kind'] == 'event']:
            related = [f for f in fragments if (f['kind'] == 'event' and f['source'] == source['source']) or
                       (f['kind'] != 'event' and f['source'] in source['content']['source_panel_ids'])]
            used.update(f['id'] for f in related)
            events.append({'summary': source['content']['summary'], 'fragment_ids': [f['id'] for f in related],
                'kind': 'current_action', 'time': {'original_time': '本话当前', 'status': 'supported'},
                'facets': [{'role': 'action', 'text': source['content']['summary'], 'fragment_ids': [f['id']],
                            'status': f['status']} for f in related]})
        for f in fragments:
            if f['id'] not in used:
                events.append({'summary': '保留未归并信息', 'fragment_ids': [f['id']], 'kind': 'unresolved',
                    'time': {'original_time': '未知', 'status': 'unknown'},
                    'facets': [{'role': 'unknown', 'text': str(f['content']), 'fragment_ids': [f['id']], 'status': 'unknown'}]})
        order = {f['id']: i for i, f in enumerate(fragments)}
        events.sort(key=lambda e: min(order[f] for f in e['fragment_ids']))
        return {'events': events}
    events = []
    used = set()
    for source in data['analysis']['events']:
        fragments = [f for f in data['fragments'] if f['source_event_id'] == source['event_id'] or
                     (not f['source_event_id'] and set(f['source_panel_ids']) <= set(source['source_panel_ids']))]
        used.update(f['fragment_id'] for f in fragments)
        events.append({'summary': source['summary'], 'fragment_ids': [f['fragment_id'] for f in fragments],
            'kind': 'current_action', 'time': {'original_time': '本话当前', 'status': 'supported'},
            'facets': [{'role': 'action', 'text': source['summary'], 'fragment_ids': [f['fragment_id']],
                        'status': f['status']} for f in fragments]})
    for f in data['fragments']:
        if f['fragment_id'] not in used:
            value = data['analysis']
            for part in f['path'].split('/'):
                value = value[int(part)] if isinstance(value, list) else value[part]
            events.append({'summary': '保留未归并信息', 'fragment_ids': [f['fragment_id']], 'kind': 'unresolved',
                'time': {'original_time': '未知', 'status': 'unknown'},
                'facets': [{'role': 'unknown', 'text': str(value), 'fragment_ids': [f['fragment_id']], 'status': 'unknown'}]})
    order = {f['fragment_id']: i for i, f in enumerate(data['fragments'])}
    events.sort(key=lambda e: min(order[f] for f in e['fragment_ids']))
    return {'events': events}
