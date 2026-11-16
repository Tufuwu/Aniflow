"""Bounded model correction of fact-sheet references; never normalize bad IDs."""
import copy
import socket

import pytest

from pipeline.script_writer import ScriptWriter
from pipeline.story_bible import StoryError
from pipeline.narrative_director import NarrativeDirector
from test_engagement_flow import story
from test_engagement_flow import polish_payload


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**k):raise AssertionError('Offline retry test')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket,'create_connection',deny)


def test_bad_reference_type_is_corrected_by_model_once_and_validated_result_reused(story):
    store,state,model,_=story;original=model.ask
    plan=NarrativeDirector(model).plan(state['bible']);attempts=[]
    def ask(stage,prompt,images=None,fresh=False):
        raw=copy.deepcopy(original(stage,prompt,images,fresh))
        if stage=='story_digest':
            attempts.append((prompt,fresh))
            if len(attempts)==1:raw['nodes'][0]['facts'][0]['event_ids']='E1'
        return raw
    model.ask=ask;writer=ScriptWriter(model,store)
    facts=writer.prepare_facts(state['bible'],plan)
    assert len(attempts)==2 and [f for _,f in attempts]==[False,True]
    assert 'event_ids 必须是非空数组' in attempts[1][0]
    assert facts[0]['facts'][0]['event_ids']==['E1','E2']
    assert writer.prepare_facts(state['bible'],plan)==facts and len(attempts)==2


def test_out_of_scope_reference_remains_rejected_after_three_attempts(story):
    store,state,model,_=story;original=model.ask
    plan=NarrativeDirector(model).plan(state['bible']);attempts=[]
    def ask(stage,prompt,images=None,fresh=False):
        raw=copy.deepcopy(original(stage,prompt,images,fresh))
        if stage=='story_digest':
            attempts.append(prompt);raw['nodes'][0]['facts'][0]['event_ids']=['E5']
        return raw
    model.ask=ask
    with pytest.raises(StoryError,match='引用越界'):
        ScriptWriter(model,store).prepare_facts(state['bible'],plan)
    assert len(attempts)==3 and '允许：' in attempts[-1]
    assert store.read()==state
