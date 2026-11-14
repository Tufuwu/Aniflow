"""Offline usage audit. Response limits are never counted as consumption."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def audit(root):
    stages = defaultdict(lambda:{'requests':0,'usage_known':0,'usage_unknown':0,
        'input_tokens':0,'output_tokens':0,'reasoning_tokens':0,'reasoning_unknown':0,
        'finish_reasons':Counter(),'retry_reasons':Counter(),'reuse':Counter()})
    records = []
    for path in sorted((Path(root)/'calls').glob('*.json')):
        raw = json.loads(path.read_text(encoding='utf-8'))
        stage = raw.get('request',{}).get('stage','unknown');row = stages[stage]
        meta = raw.get('response_metadata') or {};usage = meta.get('usage') or {}
        inp = usage.get('prompt_tokens',usage.get('input_tokens'))
        out = usage.get('completion_tokens',usage.get('output_tokens'))
        reasoning = (usage.get('completion_tokens_details') or usage.get('output_tokens_details') or {}).get('reasoning_tokens')
        row['requests']+=1
        known = isinstance(inp,int) and isinstance(out,int)
        row['usage_known' if known else 'usage_unknown']+=1
        if known:row['input_tokens']+=inp;row['output_tokens']+=out
        if isinstance(reasoning,int):row['reasoning_tokens']+=reasoning
        else:row['reasoning_unknown']+=1
        reason = raw.get('retry_reason') or 'not_recorded'
        row['retry_reasons'][reason]+=1
        row['finish_reasons'][meta.get('finish_reason') or 'unknown']+=1
        records.append({'file':path.name,'stage':stage,'usage_status':'reported' if known else 'unknown',
                        'input_tokens':inp,'output_tokens':out,'reasoning_tokens':reasoning,
                        'finish_reason':meta.get('finish_reason'),'retry_reason':raw.get('retry_reason')})
    for path in (Path(root)/'usage').glob('*.json'):
        event = json.loads(path.read_text(encoding='utf-8'))
        if event['event'] in {'cache_hit','checkpoint_reuse','checkpoint_saved'}:
            stages[event['stage']]['reuse'][event['event']]+=1
    return {'stages':dict(stages),'requests':records,'note':'Totals cover reported usage only; unknown usage is not zero. No cost estimate.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('chapter',type=Path)
    args=parser.parse_args();print(json.dumps(audit(args.chapter),ensure_ascii=False,indent=2))
