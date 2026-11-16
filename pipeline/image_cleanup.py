"""Reviewable display derivatives; source evidence and shot mappings stay immutable."""
from copy import deepcopy
import hashlib
from pathlib import Path
from uuid import uuid4

from PIL import Image

from pipeline.chatgpt_browser import ChatGPTBrowser, CLEAN_PROMPT, BLOCK_REPAIR_PROMPT, cleanup_options
from pipeline.story_bible import StoryError, digest
from pipeline.image_masks import mask_regions


def selected_panels(state):
    return {s['panel_id'] for g in state.get('groups', []) for s in g['selected']}


def clean_image_path(story, source, state, root, pid, pending=False):
    panel = next((p for p in source['bible']['panels'] if p['source_panel_id']==pid), None)
    if panel is None:
        raise StoryError('清图来源不存在')
    story.evidence_path(panel)
    item = state.get('clean_images', {}).get(pid)
    if not item or item.get('original_sha256') != panel['sha256']:
        raise StoryError('清图已过期，请按当前原图重新处理')
    if not pending and item.get('review_status') != 'approved':
        raise StoryError('清图尚未审核通过')
    path = (root/item['relative_path']).resolve()
    if not path.is_relative_to((root/'media/image_cleanup').resolve()) or not path.is_file():
        raise StoryError('清图资源不存在')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise StoryError('清图内容已变化，请重新处理和审核')
    return path


def display_assets(story, source, state, root):
    settings=state.get('image_cleanup', {})
    if not settings.get('enabled'):
        return {}
    selected=selected_panels(state)
    targets=selected if settings['scope']=='selected' else selected & set(settings['panel_ids'])
    assets={}
    for pid in targets:
        item=state.get('clean_images',{}).get(pid)
        if item and item.get('review_status')=='keep_original':
            # Explicit manual choice, never a silent fallback on a failed request.
            panel=next(p for p in source['bible']['panels'] if p['source_panel_id']==pid)
            if item.get('original_sha256')!=panel['sha256']:
                raise StoryError('保留原图的选择已过期')
            continue
        path=clean_image_path(story,source,state,root,pid)
        assets[pid]={**deepcopy(item),'path':str(path)}
    return assets


class ImageCleanupMixin:
    def clean_images(self, revision, options, panel_ids=None, fresh=False, progress=None, provider=None, feedback=None, watermark_regions=None, delete_chat_after=False):
        options=cleanup_options(options)
        if type(fresh) is not bool:
            raise StoryError('重新生成开关无效')
        if type(delete_chat_after) is not bool:
            raise StoryError('聊天清理开关无效')
        if feedback is not None and (not isinstance(feedback,str) or len(feedback)>2000):
            raise StoryError('清图修改建议需为2000字以内的文字')
        feedback=(feedback or '').strip()
        provider=provider or ChatGPTBrowser()
        with self.store.lock:
            state=self._current(revision)
            source=self.source(state)
            selected=selected_panels(state)
            if panel_ids is None:
                targets=sorted(selected)
            else:
                if (not isinstance(panel_ids,list) or not panel_ids
                        or any(not isinstance(p,str) or p not in selected for p in panel_ids)
                        or len(panel_ids)!=len(set(panel_ids))):
                    raise StoryError('只能清理当前已选镜头，分镜ID不能重复')
                targets=list(panel_ids)
            if not targets:
                raise StoryError('请先选择需要用于成片的镜头')
            if watermark_regions is not None and (not isinstance(watermark_regions,dict) or any(k not in selected for k in watermark_regions)):
                raise StoryError('预处理区域只能对应当前已选图片')
            regions={pid:mask_regions(boxes) for pid,boxes in (watermark_regions or {}).items()}
            previous=state.get('image_cleanup',{})
            all_selected=panel_ids is None or previous.get('enabled') and previous.get('scope')=='selected'
            saved_targets=sorted(set(targets)|set(previous.get('panel_ids',[]) if previous.get('enabled') else []))
            state.update(image_cleanup={'enabled':True,'scope':'selected' if all_selected else 'specific',
                'panel_ids':saved_targets,'options':options,'provider':'chatgpt_web','status':'running',
                'watermark_regions':{**previous.get('watermark_regions',{}),**regions},'delete_chat_after':delete_chat_after},export=None)
            state.setdefault('clean_images',{})
            state=self.store.save(state,'start_image_cleanup',revision)
            try:
                for index,pid in enumerate(targets):
                    panel=next(p for p in source['bible']['panels'] if p['source_panel_id']==pid)
                    original=self.story.evidence_path(panel)
                    boxes=state['image_cleanup']['watermark_regions'].get(pid,[])
                    key=digest({'source':panel['sha256'],'options':options,'prompt':BLOCK_REPAIR_PROMPT if boxes else CLEAN_PROMPT,'feedback':feedback,'regions':boxes,'repair_version':1})
                    if fresh:
                        key += '-'+uuid4().hex
                    directory=self.store.root/'media/image_cleanup'
                    target=directory/(key+'.png')
                    if progress:
                        progress(index,len(targets),f'GPT 网页清图：来源 {pid}')
                    item=provider.edit_image(original,target,options,directory/(key+'.json'),
                        progress=lambda m: progress(index,len(targets),f'来源 {pid}：{m}') if progress else None,feedback=feedback,watermark_regions=boxes)
                    if item.get('status')!='complete' or item.get('original_sha256')!=panel['sha256']:
                        raise StoryError('清图尚未完整生成或来源不匹配')
                    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:
                        raise StoryError('清图文件不完整或校验失败')
                    with Image.open(target) as picture:
                        picture.verify()
                    if delete_chat_after:
                        try:
                            provider.delete_job_chat(directory/(key+'.json'),confirmed=True)
                            item['conversation_deleted']=True
                        except Exception as exc:
                            item['conversation_deletion_error']=str(exc)
                    old=state['clean_images'].get(pid,{})
                    reviewed=old.get('review_status','pending') if not fresh and old.get('options')==options and old.get('sha256')==item['sha256'] and old.get('original_sha256')==panel['sha256'] else 'pending'
                    state['clean_images'][pid]={k:v for k,v in item.items() if k in
                        {'sha256','original_sha256','provider','options','conversation_url','prompt','watermark_regions',
                         'conversation_deleted','conversation_deletion_error'}}
                    state['clean_images'][pid].update(relative_path=target.relative_to(self.store.root).as_posix(),review_status=reviewed)
                    state=self.store.save(state,f'clean_image:{pid}',state['revision'])
                state['image_cleanup'].update(status='needs_review',error=None)
                if progress:
                    progress(len(targets),len(targets),'清图已保存，请对照原图审核后采用')
                return self.store.save(state,'complete_image_cleanup',state['revision'])
            except Exception as exc:
                state['image_cleanup'].update(status='needs_attention',error=str(exc))
                self.store.save(state,'image_cleanup_interrupted',state['revision'])
                raise

    def review_clean_image(self, revision, pid, action):
        with self.store.lock:
            state=self._current(revision)
            source=self.source(state)
            if action=='disable':
                state.setdefault('image_cleanup',{})['enabled']=False
            else:
                if pid not in selected_panels(state):
                    raise StoryError('清图来源不在当前已选镜头中')
                item=state.get('clean_images',{}).get(pid)
                if action=='approve':
                    clean_image_path(self.story,source,state,self.store.root,pid,pending=True)
                    item['review_status']='approved'
                elif action=='keep_original':
                    panel=next(p for p in source['bible']['panels'] if p['source_panel_id']==pid)
                    self.story.evidence_path(panel)
                    item=state.setdefault('clean_images',{}).setdefault(pid,{})
                    item.update(original_sha256=panel['sha256'],review_status='keep_original')
                else:
                    raise StoryError('不支持的清图审核操作')
            state['export']=None
            return self.store.save(state,f'cleanup_{action}:{pid}',revision)
