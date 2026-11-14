"""Cross-paragraph information checks; keywords alone never block a callback."""
from difflib import SequenceMatcher
import re

from pipeline.story_bible import digest
from pipeline.story_model import require


# This is a review contract, not a scoring rubric or a story-specific rule.
CONTENT_ASPECTS = ('opening', 'turns', 'exposition', 'references', 'action', 'emotion', 'ending')
CONTENT_DETAILS = {
    'opening': ('entry_context', 'concrete_focus', 'follow_through'),
    'exposition': ('purpose', 'placement'),
    'references': ('listener_tracking', 'identity_timing', 'grounded_labels'),
    'action': ('oral_delivery',),
}


def review_basis(bible, plan, segments):
    return digest({'bible': bible, 'plan': plan, 'segments':[
        {k:s.get(k) for k in ('segment_id','event_id','event_ids','context_event_ids','source_panel_ids','text')}
        for s in segments]})


def review_ready(review, bible, plan, segments):
    return bool(review and review.get('passed') and not review.get('issues')
                and (review.get('information_review') or {}).get('status') == 'reviewed'
                and (review.get('content_review') or {}).get('status') == 'reviewed'
                and (review.get('content_review') or {}).get('version', 0) >= 2
                and review.get('basis_hash') == review_basis(bible, plan, segments))


def content_issues(segments, bible, raw):
    checks = raw.get('content_checks')
    if checks is None:
        return [], {'status':'needs_review', 'reason':'缺少开场、转折、背景、称呼、行动、情绪及结尾复核'}
    require(isinstance(checks,list) and len(checks)==len(CONTENT_ASPECTS), '叙事内容复核漏项')
    require(all(isinstance(c,dict) for c in checks)
            and [c.get('aspect') for c in checks]==list(CONTENT_ASPECTS), '叙事内容复核项目或顺序无效')
    incomplete = [c['aspect'] for c in checks if any(
        not isinstance(c.get('assessments'), dict) or
        not isinstance(c['assessments'].get(k), str) or not c['assessments'][k].strip()
        for k in CONTENT_DETAILS.get(c['aspect'], ()))]
    if incomplete:
        return [], {'status':'needs_review', 'reason':'缺少背景关注点、身份揭晓或口头讲述检查：'+','.join(incomplete)}
    by_id = {s['event_id']:s for s in segments}
    evidence_ids = {e['event_id'] for e in bible['events']}
    issues = []
    for check in checks:
        require(check.get('verdict') in {'passed','issues','not_applicable'}
                and isinstance(check.get('detail'),str) and check['detail'].strip(), '叙事内容判断缺少理由')
        citations, evidence, findings = check.get('citations'), check.get('evidence_event_ids'), check.get('issues')
        require(isinstance(citations,list) and isinstance(evidence,list) and isinstance(findings,list), '叙事内容复核依据格式无效')
        for cite in citations:
            require(isinstance(cite,dict) and cite.get('event_id') in by_id
                    and isinstance(cite.get('text'),str) and cite['text'].strip()
                    and cite['text'] in by_id[cite['event_id']]['text'], '叙事复核必须引用当前段落原文')
        require(all(isinstance(e,str) and e in evidence_ids for e in evidence), '叙事复核引用未知证据')
        require(check['verdict']=='not_applicable' or citations and evidence, '叙事复核缺少正文及事件依据')
        require(check['aspect'] not in {'opening','references','ending'} or check['verdict']!='not_applicable',
                '开场、称呼和结尾不能跳过复核')
        require(bool(findings)==(check['verdict']=='issues'), '叙事复核结论与问题不一致')
        for finding in findings:
            require(isinstance(finding,dict) and finding.get('event_id') in by_id
                    and finding['event_id'] in {c['event_id'] for c in citations}
                    and all(isinstance(finding.get(k),str) and finding[k].strip() for k in ('detail','suggestion')),
                    '叙事问题必须指向可定向修订的节点')
            issues.append({**finding,'aspect':check['aspect']})
    return issues, {'version':2, 'status':'reviewed', 'checks':checks}


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


def information_issues(segments, raw, candidates):
    issues = []
    by_id = {s['event_id']:s for s in segments}
    assessments = raw.get('repeat_assessments', [])
    require(isinstance(assessments, list), '重复复核格式无效')
    require(all(isinstance(r,dict) for r in assessments), '重复复核格式无效')
    require(len({r.get('repeat_id') for r in assessments}) == len(assessments)
            and all(r.get('repeat_id') in {c['repeat_id'] for c in candidates} for r in assessments),
            '重复复核引用未知或重复项')
    for candidate in candidates:
        decision = next((r for r in assessments if r['repeat_id']==candidate['repeat_id']), {})
        purpose = decision.get('purpose')
        addition = decision.get('new_information', '')
        useful = (purpose in {'callback','emphasis','payoff'} and isinstance(decision.get('reason'),str)
                  and bool(decision['reason'].strip()) and isinstance(addition,str) and addition.strip()
                  and addition in by_id[candidate['event_id']]['text']
                  and addition not in by_id[candidate['earlier_event_id']]['text'])
        if not useful:
            issues.append({**candidate, 'detail':'跨段逐字重复，尚无可核实的新推进或回扣作用',
                           'suggestion':'定向删除重复交代，保留本段新增行动、变化或兑现'})
    flow = raw.get('information_flow')
    # Legacy responses remain readable but cannot count as a semantic review.
    if flow is None:
        return issues, {'status':'needs_review','reason':'缺少逐段新增信息与语义重复复核'}
    require(isinstance(flow,list) and len(flow)==len(segments), '新增信息复核漏段')
    for row,segment in zip(flow,segments):
        require(isinstance(row,dict) and row.get('event_id')==segment['event_id'], '新增信息复核锚点无效')
        additions = row.get('new_information')
        require(isinstance(additions,list) and all(isinstance(t,str) and t.strip() and t in segment['text'] for t in additions),
                '新增信息必须引用当前正文')
        repeats = row.get('repeats', [])
        require(isinstance(repeats,list), '语义重复复核格式无效')
        earlier = {s['event_id'] for s in segments[:segments.index(segment)]}
        for repeat in repeats:
            require(isinstance(repeat,dict) and repeat.get('earlier_event_id') in earlier
                    and repeat.get('purpose') in {'redundant','callback','emphasis','payoff'}
                    and isinstance(repeat.get('detail'),str) and repeat['detail'].strip(), '语义重复关联无效')
            purpose = repeat.get('new_information')
            useful = (isinstance(purpose,str) and purpose.strip() and purpose in additions
                      and purpose not in by_id[repeat['earlier_event_id']]['text'])
            if repeat['purpose']=='redundant' or not useful:
                issues.append({'event_id':segment['event_id'], 'detail':'跨段语义重复：'+repeat['detail'],
                               'earlier_event_id':repeat['earlier_event_id']})
        if not additions:
            issues.append({'event_id':segment['event_id'], 'detail':'本段未说明新增信息、行动、情绪变化或有作用的停顿'})
    return issues, {'status':'reviewed','segments':flow,'repeat_assessments':assessments}
