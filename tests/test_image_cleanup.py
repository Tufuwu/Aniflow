"""Persistent browser ownership, interrupted jobs, and reviewed display derivatives."""
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from flask import Flask
from PIL import Image
import pytest

from pipeline.chatgpt_browser import ChatGPTBrowser, CLEAN_PROMPT, cleanup_options
from pipeline.image_cleanup import clean_image_path, display_assets, selected_panels
from pipeline.shot_api import create_shot_blueprint
from pipeline.story_bible import StoryError, atomic_json, digest
from test_shot_editing import chapter, selected, reviewed


class FakeWeb:
    def __init__(self, fail_at=None):
        self.calls=[]
        self.fail_at=fail_at

    def edit_image(self, original, target, options, checkpoint, **kwargs):
        self.calls.append(str(original))
        if len(self.calls)==self.fail_at:
            raise StoryError('请在专用 Chrome 登录')
        target.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(original) as source:
            # Visibly distinct pixels let the preview test detect silent original fallback.
            Image.new('RGB',source.size,'#c94854').save(target)
        return {'status':'complete','original_sha256':hashlib.sha256(Path(original).read_bytes()).hexdigest(),
                'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'provider':'chatgpt_web','options':options}


def test_dedicated_profile_reconnects_after_python_restart_without_new_window(tmp_path,monkeypatch):
    root=tmp_path/'managed'
    first=ChatGPTBrowser(root,port=19225)
    monkeypatch.setattr(ChatGPTBrowser,'_show_chat',lambda self:None)
    version={'Browser':'Chrome/150','webSocketDebuggerUrl':'ws://127.0.0.1:19225/devtools/browser/ours'}
    count=[0]
    def version_response():
        count[0]+=1
        return None if count[0]==1 else version
    monkeypatch.setattr(first,'_version',version_response)
    launched=[]
    monkeypatch.setattr('pipeline.chatgpt_browser.chrome_binary',lambda:'installed-chrome.exe')
    monkeypatch.setattr('pipeline.chatgpt_browser.subprocess.Popen',lambda args,**kw:(launched.append(args) or SimpleNamespace(pid=1234)))
    state=first.open()
    assert state['running'] and state['managed'] and state['login']=='unknown'
    assert f'--user-data-dir={root.resolve()/"profile"}' in launched[0]
    assert '--remote-debugging-address=127.0.0.1' in launched[0]
    assert '--headless' not in launched[0]
    second=ChatGPTBrowser(root,port=19225)
    monkeypatch.setattr(second,'_version',lambda:version)
    assert second.open()['managed'] and len(launched)==1
    # Another Chrome on the old port must not be treated as the reusable login.
    monkeypatch.setattr(second,'_version',lambda:{**version,'webSocketDebuggerUrl':'ws://another-browser'})
    with pytest.raises(StoryError,match='占用'):
        second.open()


def test_completed_image_cache_is_bound_to_source_options_and_image_bytes(tmp_path,monkeypatch):
    original=tmp_path/'original.png';Image.new('RGB',(100,160),'blue').save(original)
    target=tmp_path/'clean.png';Image.new('RGB',(100,160),'red').save(target)
    options=cleanup_options({'watermark':True})
    sha=hashlib.sha256(original.read_bytes()).hexdigest()
    key=digest({'source':sha,'options':options,'prompt':CLEAN_PROMPT,'provider':'chatgpt_web_v1'})
    checkpoint=tmp_path/'task.json'
    atomic_json(checkpoint,{'key':key,'status':'complete','sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    browser=ChatGPTBrowser(tmp_path/'browser')
    monkeypatch.setattr(browser,'connect',lambda:pytest.fail('Cache must not submit again'))
    assert browser.edit_image(original,target,options,checkpoint)['status']=='complete'
    with pytest.raises(StoryError,match='不匹配'):
        browser.edit_image(original,target,{'dialogue':True},checkpoint)
    target.write_bytes(b'damaged')
    with pytest.raises(StoryError,match='内容已变化'):
        browser.edit_image(original,target,options,checkpoint)


def test_cleanup_requires_explicit_options_and_preserves_existing_destination(tmp_path):
    browser=ChatGPTBrowser(tmp_path/'browser')
    original=tmp_path/'original.png';Image.new('RGB',(60,80)).save(original)
    target=tmp_path/'existing.png';target.write_bytes(b'keep me')
    for value in ({},{'watermark':1},{'unknown':True},None):
        with pytest.raises(StoryError):cleanup_options(value)
    with pytest.raises(StoryError,match='不能覆盖'):
        browser.edit_image(original,target,{'watermark':True},tmp_path/'task.json')
    assert target.read_bytes()==b'keep me'


def test_login_expiry_keeps_submitted_job_and_does_not_resend_after_login(tmp_path,monkeypatch):
    original=tmp_path/'original.png';Image.new('RGB',(100,160),'blue').save(original)
    options=cleanup_options({'watermark':True})
    key=digest({'source':hashlib.sha256(original.read_bytes()).hexdigest(),'options':options,
                'prompt':CLEAN_PROMPT,'provider':'chatgpt_web_v1'})
    checkpoint=tmp_path/'task.json'
    conversation='https://chatgpt.com/c/previous-job'
    atomic_json(checkpoint,{'key':key,'status':'submitted','conversation_url':conversation,'assistant_count':0})
    class Page:
        url='https://auth.openai.com/login'
        def goto(self,*args,**kwargs):pass
        def bring_to_front(self):pass
        def screenshot(self,**kwargs):pass
        def get_by_role(self,*args,**kwargs):return SimpleNamespace(count=lambda:0)
        def locator(self,*args):pytest.fail('A submitted job must not send another prompt')
    page=Page()
    @contextmanager
    def connection():
        yield SimpleNamespace(pages=[],new_page=lambda:page)
    browser=ChatGPTBrowser(tmp_path/'browser')
    monkeypatch.setattr(browser,'status',lambda:{'running':True})
    monkeypatch.setattr(browser,'connect',connection)
    with pytest.raises(StoryError,match='登录'):
        browser.edit_image(original,tmp_path/'clean.png',options,checkpoint)
    job=json.loads(checkpoint.read_text('utf-8'))
    assert job['status']=='submitted' and job['conversation_url']==conversation
    monkeypatch.setattr(browser,'_login_required',lambda page:False)
    with pytest.raises(StoryError,match='超时'):
        browser.edit_image(original,tmp_path/'clean.png',options,checkpoint,timeout=0)


def test_new_web_gallery_downloads_blob_image_through_its_viewer(tmp_path):
    source=tmp_path/'downloaded-original.png';Image.new('RGB',(80,100),'#bd4c77').save(source)
    target=tmp_path/'clean.png'
    actions=[]
    class Control:
        def __init__(self,name,count=1):self.name=name;self.total=count;self.first=self
        def count(self):return self.total
        def click(self):actions.append(self.name)
        def wait_for(self,**kwargs):pass
    class Answer:
        def get_by_role(self,*args,**kwargs):return Control('inline-download',0)
        def get_by_test_id(self,name):
            assert name=='generated-image-preview'
            return Control('open-image-viewer')
    class Page:
        def get_by_role(self,role,name):
            return Control('download-original' if name.search('下载') else 'close-image-viewer')
        def expect_download(self,**kwargs):
            @contextmanager
            def received():
                yield SimpleNamespace(value=SimpleNamespace(save_as=lambda path:Path(path).write_bytes(source.read_bytes())))
            return received()
    ChatGPTBrowser._save_result(Page(),Answer(),{'src':'blob:https://chatgpt.com/result'},target)
    assert actions==['open-image-viewer','download-original','close-image-viewer']
    with Image.open(target) as actual:
        assert actual.size==(80,100) and actual.getpixel((0,0))==(189,76,119)
    assert not target.with_suffix('.download').exists()


def test_cleanup_retains_originals_and_needs_manual_adoption(chapter):
    story,source,pipeline=chapter
    state=selected(chapter)
    before={p['source_panel_id']:story.evidence_path(p).read_bytes() for p in source['bible']['panels']}
    fake=FakeWeb()
    state=pipeline.clean_images(state['revision'],{'watermark':True,'dialogue':True},provider=fake)
    assert len(fake.calls)==len(selected_panels(state))
    assert all(i['review_status']=='pending' for i in state['clean_images'].values())
    with pytest.raises(StoryError,match='尚未审核'):
        display_assets(story,source,state,pipeline.store.root)
    pid=next(iter(selected_panels(state)))
    state=pipeline.review_clean_image(state['revision'],pid,'approve')
    path=clean_image_path(story,source,state,pipeline.store.root,pid)
    assert path.read_bytes()!=before[pid]
    assert before=={p['source_panel_id']:story.evidence_path(p).read_bytes() for p in source['bible']['panels']}
    for other in selected_panels(state)-{pid}:
        state=pipeline.review_clean_image(state['revision'],other,'keep_original')
    assert set(display_assets(story,source,state,pipeline.store.root))=={pid}
    path.write_bytes(b'tampered')
    with pytest.raises(StoryError,match='内容已变化'):
        display_assets(story,source,state,pipeline.store.root)


def test_interrupted_batch_saves_completed_images_and_blocks_export(chapter):
    story,source,pipeline=chapter
    state=selected(chapter)
    with pytest.raises(StoryError,match='登录'):
        pipeline.clean_images(state['revision'],{'watermark':True},provider=FakeWeb(fail_at=2))
    current=pipeline.read()
    assert len(current['clean_images'])==1
    assert current['image_cleanup']['status']=='needs_attention'
    assert current['revision']!=state['revision']
    assert current['export'] is None
    with pytest.raises(StoryError):
        display_assets(story,source,current,pipeline.store.root)


def test_approval_checks_source_and_rejects_paths_outside_media(chapter,tmp_path):
    story,source,pipeline=chapter
    state=selected(chapter)
    state=pipeline.clean_images(state['revision'],{'watermark':True},provider=FakeWeb())
    pid=next(iter(state['clean_images']))
    injected=copy.deepcopy(state)
    injected['clean_images'][pid]['relative_path']=str(tmp_path/'outside.png')
    with pytest.raises(StoryError,match='资源不存在'):
        clean_image_path(story,source,injected,pipeline.store.root,pid,pending=True)
    injected=copy.deepcopy(state)
    injected['clean_images'][pid]['original_sha256']='changed'
    with pytest.raises(StoryError,match='过期'):
        clean_image_path(story,source,injected,pipeline.store.root,pid,pending=True)
    with pytest.raises(StoryError,match='当前已选'):
        pipeline.clean_images(state['revision'],{'watermark':True},panel_ids=['unknown'],provider=FakeWeb())


def test_preview_adopts_only_reviewed_clean_asset_and_disable_restores_original(chapter,monkeypatch):
    story,source,pipeline=chapter
    state=selected(chapter)
    state=pipeline.clean_images(state['revision'],{'watermark':True},provider=FakeWeb())
    pid=next(iter(state['clean_images']))
    group=next(g for g in state['groups'] if any(s['panel_id']==pid for s in g['selected']))
    app=Flask(__name__);app.config['TESTING']=True
    monkeypatch.setattr('pipeline.shot_api.story_store',lambda project:story)
    app.register_blueprint(create_shot_blueprint())
    client=app.test_client()
    endpoint=f'/api/editing/test/preview/{group["narration_id"]}/{pid}'
    before=client.get(endpoint+'?revision='+state['revision']).data
    state=pipeline.review_clean_image(state['revision'],pid,'approve')
    after=client.get(endpoint+'?revision='+state['revision']).data
    assert before!=after
    state=pipeline.review_clean_image(state['revision'],'all','disable')
    assert client.get(endpoint+'?revision='+state['revision']).data==before
    assert display_assets(story,source,state,pipeline.store.root)=={}


def test_real_export_uses_approved_derivative_without_changing_original_evidence(chapter):
    pytest.importorskip('moviepy')
    story,source,pipeline=chapter
    state=reviewed(chapter)
    state=pipeline.clean_images(state['revision'],{'watermark':True},provider=FakeWeb())
    for pid in list(state['clean_images']):
        state=pipeline.review_clean_image(state['revision'],pid,'approve')
    state=pipeline.direct(state['revision'])
    state=pipeline.export(state['revision'])
    root=pipeline.store.root/'exports'/state['export']['export_id']
    manifest=json.loads((root/'manifest.json').read_text('utf-8'))
    assert set(manifest['display_assets'])==selected_panels(state)
    with Image.open(root/'work/base_0000.png') as frame:
        assert frame.getpixel((540,875))==(201,72,84)
    assert state['export']['report']['complete']
