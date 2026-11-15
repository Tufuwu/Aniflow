"""Candidate identities and text-only comparisons; selection grants no approval."""
from copy import deepcopy
from pipeline.story_bible import digest
from pipeline.story_model import require, text
from pipeline.input_compaction import compact_json
from pipeline.prompt_material import (script_text, revision_feedback, editor_material,
                                     current_fact_sheet, text_evidence)
from pipeline.narration_review import review_ready

VARIANTS = ('从当前处境和具体问题组织讲述，保留必要背景。',
            '调整有据背景的组织与口语承接，让行动与反应更容易跟随。',
            '突出原作已有的代价、反差与情绪兑现，保护安静空间。')


def candidate_basis(candidate):
    basis = {'analysis_id':candidate['analysis_id'],'plan':candidate['plan'],
                   'segments':script_text(candidate['script']['segments']),
                   'review':candidate['script'].get('narrative_review')}
    if candidate['script'].get('source_material_basis'):
        basis['source_material_basis'] = candidate['script']['source_material_basis']
    return digest(basis)


def comparison_basis(candidates):
    return digest([(c['candidate_id'],candidate_basis(c)) for c in candidates])


def comparison_pool(state):
    ids = state.get('candidate_set')
    return [c for c in state.get('candidates',[]) if ids is None or c['candidate_id'] in ids]


def sync_selected(state):
    cid = state.get('selected_candidate_id')
    for c in state.get('candidates',[]):
        if c['candidate_id']==cid:
            c['plan'],c['script'] = deepcopy(state['plan']),deepcopy(state['script'])
            c['status'] = state.get('stage','review')
    comparison = state.get('candidate_comparison')
    if comparison and comparison.get('basis_hash')!=comparison_basis(comparison_pool(state)):
        comparison.update(status='needs_review',recommended_candidate_id=None,reason='候选正文或复核依据已变化，需重新比较')


def compare_candidates(model, bible, candidates):
    basis = comparison_basis(candidates)
    eligible = [c for c in candidates if c['analysis_id']==bible['bible_id'] and
                review_ready(c['script'].get('narrative_review'),bible,c['plan'],c['script']['segments'])]
    if len(eligible)<2:
        return {'status':'needs_review','basis_hash':basis,'recommended_candidate_id':None,
                'reason':'至少需要两份当前文字复核通过的候选；保留现稿，等待有目标的处理'}
    shared, material_ids = {}, {}
    for candidate in eligible:
        plan = candidate['plan']
        facts = current_fact_sheet(bible,plan,candidate['script'])
        key = digest({'plan':plan,'source_basis':candidate['script'].get('source_material_basis'), 'facts':facts})
        material_ids[candidate['candidate_id']] = key
        if key not in shared:
            shared[key] = {**editor_material(bible,plan,facts), **text_evidence(bible,plan['beats'],facts)}
    prompt = '''Candidate Comparison v2。比较当前候选的必要背景、具体关注点、人物辨认、转折触发、口语、情绪空间、重复与兑现。
所有分析与fact_sheet都是待核材料，模型整理或确认不等于正确，不能把来源覆盖当作正文已讲清。
引用每份候选的原句，分别说明优点、问题和事实风险，比较后给出推荐理由。没有合适稿时推荐 null，保持待审。
不能以篇幅短、句子短或感叹号多代替吸引力判断。候选只能调整表达与背景组织，不能改变原作时间、因果、事实及身份揭晓顺序。
这是文字推荐，不是原图事实核验或人工审核。没有原图时不得臆测画面。不要重写任何正文。
返回 {"candidates":[{"candidate_id":"原编号","citations":[{"event_id":"节点锚点","text":"逐字原句"}],
"strengths":"具体优点","problems":"具体问题或无","fact_risks":"事实风险或仍待原图核验"}],
"recommended_candidate_id":null,"reason":"比较这些引用与证据后的推荐/待审理由"}。
各候选按 material_id 共用依据，不能因参考资料存在就要求全部写入正文，未知与揭晓限制仍然有效。
资料：'''+compact_json({'candidates':[{'candidate_id':c['candidate_id'],'material_id':material_ids[c['candidate_id']],
             'script':script_text(c['script']['segments']),
             'review':revision_feedback(c['script'].get('narrative_review'))} for c in eligible],
             'shared_material':shared})
    try:
        raw = model.ask('candidate_compare',prompt)
        rows = raw.get('candidates');by_id = {c['candidate_id']:c for c in eligible}
        require(isinstance(rows,list) and len(rows)==len(eligible),'比较结果漏候选')
        require({r.get('candidate_id') for r in rows}==set(by_id),'比较候选编号无效')
        for r in rows:
            for key in ('strengths','problems','fact_risks'):text(r.get(key),'比较'+key)
            quotes = r.get('citations')
            require(isinstance(quotes,list) and quotes,'比较缺少原句')
            segments = {s['event_id']:s['text'] for s in by_id[r['candidate_id']]['script']['segments']}
            require(all(isinstance(q,dict) and q.get('event_id') in segments and
                        isinstance(q.get('text'),str) and q['text'].strip() and
                        q['text'] in segments[q['event_id']] for q in quotes),'比较引用不是当前候选原句')
        require(raw.get('recommended_candidate_id') is None or raw['recommended_candidate_id'] in by_id,'推荐编号无效')
        text(raw.get('reason'),'推荐理由')
        return {**raw,'status':'compared' if raw.get('recommended_candidate_id') else 'needs_review','basis_hash':basis}
    except Exception as exc:
        return {'status':'needs_review','basis_hash':basis,'recommended_candidate_id':None,'reason':str(exc)}
