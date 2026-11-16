"""AniFlow-owned, persistent Chrome session and resumable ChatGPT image jobs.

Only browser UI is used to submit images; authentication is left to the user.
The Chrome profile lives on disk and survives Python/browser restarts.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

from filelock import FileLock, Timeout
from PIL import Image

import config
from pipeline.story_bible import StoryError, atomic_json, digest
from pipeline.image_masks import mask_regions, solid_mask

CHATGPT_URL = 'https://chatgpt.com/'
COMPOSER = ':is(#prompt-textarea, #mobile-composer-prompt, [contenteditable="true"][role="textbox"], [contenteditable="true"][data-placeholder]):visible'
CLEAN_PROMPT = '''请编辑附件中的这一张原始漫画图片，用于解说视频。
清除要求：{instructions}。
完整去掉要清除的气泡、文本框的文字、边框和尖角，并自然修补遮挡区域。
严格保留原图的长宽比例、全部分格位置、人物身份、脸、表情、姿势、衣服、手和道具，保留原有动作、环境、光照和画风。
不得重新构图、裁切、拼图、增添人物、物品或情节。保留画面中实际物体上的剧情标识，例如窗户上的 SOS、产品包装。
只输出一张清理后的图片，不要解释，不要输出多张变体。任务编号：{marker}。'''
BLOCK_REPAIR_PROMPT = '''请修复附件图片中的纯洋红色矩形区域。
根据矩形四周的颜色、线条、纹理和透视，只在矩形内部自然补全背景，完整去掉洋红色色块及色边。
严格保留原画布尺寸比例、分格边框、人物的脸、表情、姿势、衣服、手、道具、动作和环境。不要扩大修补范围，不要改变人物身份，不要裁切、重新构图或增加内容。
{additional}
矩形位置采用原图归一化 [x,y,宽,高]：{regions}。
只输出一张修复后的完整图片，不要解释。任务编号：{marker}。'''
MODES = {'watermark': '出版水印、网站标志及半透明平台 logo',
         'dialogue': '对白和心理活动气泡、聊天浮层及其文字',
         'text': '旁白说明框、装饰性文字和拟声字'}


def cleanup_options(value):
    if not isinstance(value, dict) or set(value)-set(MODES):
        raise StoryError('清图选项无效')
    if any(type(v) is not bool for v in value.values()):
        raise StoryError('清图选项必须是布尔值')
    result = {k: value.get(k, False) for k in MODES}
    if not any(result.values()):
        raise StoryError('请选择需要清除的水印、对白或文字')
    return result


def chrome_binary():
    override = os.getenv('CHATGPT_CHROME_PATH', '').strip()
    if override:
        if Path(override).is_file():
            return str(Path(override).resolve())
        raise StoryError('CHATGPT_CHROME_PATH 指定的 Chrome 不存在')
    paths = [Path(os.getenv('PROGRAMFILES', 'C:/Program Files'))/'Google/Chrome/Application/chrome.exe',
             Path(os.getenv('PROGRAMFILES(X86)', 'C:/Program Files (x86)'))/'Google/Chrome/Application/chrome.exe',
             Path(os.getenv('LOCALAPPDATA', ''))/'Google/Chrome/Application/chrome.exe',
             Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')]
    return next((str(p) for p in paths if p.is_file()), None) or shutil.which('google-chrome') or shutil.which('chrome') or _missing_chrome()


def _missing_chrome():
    raise StoryError('未找到 Google Chrome，请安装 Chrome 或设置 CHATGPT_CHROME_PATH')


class ChatGPTBrowser:
    def __init__(self, root=None, port=None):
        self.root = Path(root or config.CHATGPT_BROWSER_DIR).resolve()
        self.profile = self.root/'profile'
        self.port = int(port or config.CHATGPT_BROWSER_PORT)
        if not 1024 <= self.port <= 65535:
            raise StoryError('Chrome 调试端口必须为1024到65535')
        self.endpoint = f'http://127.0.0.1:{self.port}'
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.root/'browser.lock'))

    def _version(self):
        try:
            # Local connection must not go through the user's HTTP proxy.
            with build_opener(ProxyHandler({})).open(self.endpoint+'/json/version', timeout=1) as response:
                result = json.load(response)
            if not re.search(r'(?:Chrome|Chromium)/', result.get('Browser', '')):
                raise StoryError('指定端口不是 Chrome 调试服务')
            return result
        except (OSError, ValueError):
            return None

    def status(self):
        marker = self.root/'session.json'
        live = self._version()
        owned = False
        if marker.is_file():
            data = json.loads(marker.read_text('utf-8'))
            owned = (data.get('profile') == str(self.profile) and data.get('port') == self.port
                     and bool(live) and data.get('browser_id') == live.get('webSocketDebuggerUrl'))
        return {'running': bool(live), 'managed': owned, 'browser': 'chrome',
                'endpoint': self.endpoint, 'profile': str(self.profile), 'login': 'unknown',
                'message': '请在程序专用 Chrome 中登录；登录状态由 Chrome 配置目录保存'}

    def open(self):
        try:
            with self.lock.acquire(timeout=0):
                state = self.status()
                if state['running']:
                    if not state['managed']:
                        raise StoryError('调试端口已被其他浏览器占用，请设置另一个 CHATGPT_BROWSER_PORT')
                    self._show_chat()
                    return state
                self.profile.mkdir(parents=True, exist_ok=True)
                args = [chrome_binary(), f'--user-data-dir={self.profile}',
                        f'--remote-debugging-port={self.port}', '--remote-debugging-address=127.0.0.1',
                        '--no-first-run', '--no-default-browser-check', '--new-window', CHATGPT_URL]
                process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, close_fds=True)
                atomic_json(self.root/'session.json', {'profile': str(self.profile), 'port': self.port,
                                                     'pid': process.pid, 'url': CHATGPT_URL})
                deadline = time.monotonic()+20
                while time.monotonic() < deadline:
                    version = self._version()
                    if version:
                        atomic_json(self.root/'session.json', {'profile':str(self.profile),'port':self.port,
                            'pid':process.pid,'url':CHATGPT_URL,'browser_id':version['webSocketDebuggerUrl']})
                        self._show_chat()
                        return self.status()
                    time.sleep(.25)
                raise StoryError('Chrome 未能启动调试连接，请检查专用配置目录是否被其他进程占用')
        except Timeout as exc:
            raise StoryError('Chrome 正在处理其他清图任务，请稍后重试') from exc

    def _show_chat(self):
        with self.connect() as context:
            page=next((p for p in context.pages if urlsplit(p.url).hostname in {'chatgpt.com','auth.openai.com'}),None)
            if page is None:
                page=context.new_page()
                page.goto(CHATGPT_URL,wait_until='domcontentloaded',timeout=60000)
            page.bring_to_front()

    @contextmanager
    def connect(self):
        state = self.status()
        if not state['running'] or not state['managed']:
            raise StoryError('请先点击“打开专用 Chrome”并在其中登录 ChatGPT')
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise StoryError('缺少 playwright，请运行 python -m pip install playwright；无需下载另一套浏览器') from exc
        with sync_playwright() as runtime:
            browser = runtime.chromium.connect_over_cdp(self.endpoint, timeout=15000)
            if not browser.contexts:
                raise StoryError('Chrome 没有可用的持久会话')
            # Do not close browser/context: they belong to the persistent user session.
            yield browser.contexts[0]

    @staticmethod
    def _login_required(page):
        if urlsplit(page.url).hostname != 'chatgpt.com':
            return True
        return page.get_by_role('button', name=re.compile(r'^(Log in|登录)$', re.I)).count() > 0

    def check_login(self):
        result = self.status()
        if not result['running'] or not result['managed']:
            return result
        with self.connect() as context:
            page = next((p for p in context.pages if urlsplit(p.url).hostname == 'chatgpt.com'), None)
            if page is None:
                signing_in=any(urlsplit(p.url).hostname=='auth.openai.com' for p in context.pages)
                result.update(login='required' if signing_in else 'unknown',
                              message='请继续在专用 Chrome 完成登录' if signing_in else '请点击“打开专用 Chrome”进入 ChatGPT')
            else:
                logged_out = self._login_required(page)
                composer = page.locator(COMPOSER).count()
                result.update(login='required' if logged_out else 'ready' if composer else 'unknown',
                              message='请手动完成登录或页面验证' if logged_out or not composer else '已找到可用的 GPT 聊天页面')
        return result

    @staticmethod
    def deletion_candidate(checkpoint):
        """Only completed jobs with intact local results may enter chat cleanup."""
        checkpoint=Path(checkpoint)
        job=json.loads(checkpoint.read_text('utf-8'))
        if job.get('status')!='complete':
            raise StoryError('图片尚未完整下载，不能删除用于续接的会话')
        path=Path(job.get('path',''))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=job.get('sha256'):
            raise StoryError('本地清图文件缺失或校验失败，不能删除会话')
        url=job.get('conversation_url','')
        address=urlsplit(url)
        if address.scheme!='https' or address.hostname!='chatgpt.com' or not re.fullmatch(r'/c/[a-zA-Z0-9-]+',address.path):
            raise StoryError('任务会话地址无效，不能清理')
        return {'checkpoint':str(checkpoint.resolve()),'conversation_url':url,'path':str(path.resolve()),
                'sha256':job['sha256'],'key':job['key'],'deleted':bool(job.get('conversation_deleted'))}

    def delete_job_chat(self,checkpoint,confirmed=False):
        candidate=self.deletion_candidate(checkpoint)
        if candidate['deleted']:
            return candidate
        if confirmed is not True:
            raise StoryError('图片已保存，删除网页聊天前需要明确确认')
        with self.lock.acquire(timeout=0),self.connect() as context:
            page=context.new_page()
            page.goto(candidate['conversation_url'],wait_until='domcontentloaded',timeout=60000)
            if self._login_required(page):
                raise StoryError('请先在专用 Chrome 登录，再清理已完成任务的会话')
            marker='_'+candidate['key'][:16]
            users=page.locator('[data-message-author-role="user"]:visible, [data-chatgpt-search-unit-key$=":user"]:visible').filter(has_text=marker)
            users.first.wait_for(state='visible',timeout=30000)
            # The task's own conversation is opened explicitly; never delete by
            # a generic sidebar title, which could belong to another user chat.
            menu=page.locator('[data-testid="conversation-options-button"], [data-testid="app-shell-header-context-menu-surface"]').last
            menu.wait_for(state='visible',timeout=30000)
            menu.click()
            delete=page.get_by_role('menuitem',name=re.compile(r'^(删除|删除聊天|Delete|Delete chat)$',re.I))
            delete.first.click(timeout=15000)
            dialog=page.get_by_role('dialog').last
            dialog.wait_for(state='visible',timeout=15000)
            dialog.get_by_role('button',name=re.compile(r'^(删除|Delete)$',re.I)).click(timeout=15000)
            dialog.wait_for(state='hidden',timeout=15000)
            job=json.loads(Path(checkpoint).read_text('utf-8'))
            job['conversation_deleted']=True
            atomic_json(Path(checkpoint),job)
            candidate['deleted']=True
            page.close()
            return candidate

    def edit_image(self, original, destination, options, checkpoint, timeout=600, progress=None, feedback=None, watermark_regions=None):
        """One durable request per image; reconnects submitted jobs without resending."""
        original, destination, checkpoint = Path(original), Path(destination), Path(checkpoint)
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        options = cleanup_options(options)
        if feedback is not None and (not isinstance(feedback,str) or len(feedback)>2000):
            raise StoryError('清图修改建议需为2000字以内的文字')
        feedback=(feedback or '').strip()
        regions=mask_regions(watermark_regions)
        if regions and not options['watermark']:
            raise StoryError('标记了水印区域时请开启去水印选项')
        sha = hashlib.sha256(original.read_bytes()).hexdigest()
        template=BLOCK_REPAIR_PROMPT if regions else CLEAN_PROMPT
        request_key={'source': sha, 'options': options, 'prompt': template, 'provider':'chatgpt_web_v1'}
        if feedback:
            request_key['feedback']=feedback
        if regions:
            request_key['watermark_regions']=regions
        key = digest(request_key)
        job = json.loads(checkpoint.read_text('utf-8')) if checkpoint.exists() else {'key':key,'status':'queued'}
        if job['key'] != key:
            raise StoryError('清图任务与原图或清除选项不匹配')
        if job['status']=='failed_before_generation':
            job.setdefault('failed_attempts',[]).append({k:job.get(k) for k in ('conversation_url','detail','account_refresh_retries')})
            job.update(status='queued',conversation_url=None,account_refresh_retries=0)
            atomic_json(checkpoint,job)
        if destination.exists() and not checkpoint.exists():
            raise StoryError('输出图片已存在，请选择新的路径，不能覆盖已有图片')
        if job['status'] in {'needs_review','provider_refused'}:
            raise StoryError(job.get('detail','请人工核对上次返回结果，或选择重新生成'))
        if job['status'] == 'complete':
            if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != job['sha256']:
                raise StoryError('缓存清图内容已变化，请重新创建任务')
            return job
        if regions:
            additional='另外清除：'+'；'.join(MODES[k] for k in ('dialogue','text') if options[k])+'。' if options['dialogue'] or options['text'] else ''
            prompt=BLOCK_REPAIR_PROMPT.format(additional=additional,regions=json.dumps(regions,ensure_ascii=False),marker='AF_REPAIR_'+key[:16])
        else:
            prompt = CLEAN_PROMPT.format(instructions='；'.join(MODES[k] for k in ('dialogue','text') if options[k]) or '保留完整画面，仅修整明显的平面遮挡区域', marker='AF_CLEAN_'+key[:16])
        if feedback:
            prompt += '\n补充修正要求：'+feedback
        input_path=original
        if regions:
            input_path=solid_mask(original,checkpoint.with_suffix('.masked.png'),regions)
        def save(**values):
            job.update(values)
            atomic_json(checkpoint, job)
        if not self.status()['running']:
            self.open()
        try:
            with self.lock.acquire(timeout=0), self.connect() as context:
                url = job.get('conversation_url')
                if url and (urlsplit(url).scheme != 'https' or urlsplit(url).hostname != 'chatgpt.com'):
                    raise StoryError('清图会话地址无效')
                page = next((p for p in context.pages if url and p.url == url), None)
                worker_file=self.root/'worker.json'
                if page is None and url:
                    marker=('AF_REPAIR_' if regions else 'AF_CLEAN_')+key[:16]
                    matches=[p for p in context.pages if urlsplit(p.url).hostname=='chatgpt.com' and
                             p.locator('[data-message-author-role="user"]:visible, [data-chatgpt-search-unit-key$=":user"]:visible').filter(has_text=marker).count()]
                    if len(matches)==1:
                        page=matches[0]
                        save(conversation_url=page.url)
                        url=page.url
                if page is None:
                    page = context.new_page()
                    page.goto(url or CHATGPT_URL, wait_until='domcontentloaded', timeout=60000)
                page.bring_to_front()
                viewer=page.get_by_role('button',name=re.compile(r'^(关闭查看器|Close viewer)$',re.I))
                if viewer.count():
                    viewer.first.click()
                if self._login_required(page):
                    if job['status'] in {'queued','needs_login'}:
                        save(status='needs_login')
                    else:
                        # Expired login must never turn an already submitted job
                        # into a fresh request after authentication resumes.
                        save(login_required=True)
                    raise StoryError('请在专用 Chrome 中手动登录 ChatGPT，完成后重试清图')
                page.screenshot(path=str(checkpoint.with_suffix('.png')))
                if job['status'] in {'queued','needs_login'}:
                    composer = page.locator(COMPOSER).last
                    composer.wait_for(state='visible', timeout=60000)
                    mains=page.get_by_role('main').filter(has=page.locator(COMPOSER))
                    surface=mains.last if mains.count() else page
                    model_picker=page.get_by_role('button',name='选择 ChatGPT 模型',exact=True)
                    if model_picker.count():
                        page.wait_for_function('''() => Array.from(document.querySelectorAll('button[aria-label="选择 ChatGPT 模型"]')).some(b => b.textContent.trim() && !b.textContent.includes('选择'))''',timeout=30000)
                    inputs = surface.locator('input[type="file"][aria-label="附加文件"], input[type="file"][aria-label="Attach files"]')
                    if not inputs.count():
                        inputs = surface.locator('input[type="file"]:not([capture])')
                    if not inputs.count():
                        add = page.get_by_role('button', name=re.compile(r'添加文件|Add files|Attach', re.I)).first
                        add.click(timeout=10000)
                        upload = page.get_by_role('menuitem', name=re.compile(r'上传|Upload|Add photos', re.I))
                        if upload.count():
                            with page.expect_file_chooser(timeout=10000) as chooser:
                                upload.first.click()
                            chooser.value.set_files(str(input_path))
                        else:
                            inputs.first.set_input_files(str(input_path), timeout=10000)
                    else:
                        inputs.first.set_input_files(str(input_path), timeout=10000)
                    composer.fill(prompt)
                    send = surface.locator(':is([data-testid="send-button"], button[aria-label="发送"], button[aria-label="发送提示"], button[aria-label="Send"], button[aria-label="Send prompt"]):visible').last
                    send.wait_for(state='visible', timeout=30000)
                    deadline=time.monotonic()+60
                    while not send.is_enabled() and time.monotonic()<deadline:
                        time.sleep(.25)
                    if not send.is_enabled():
                        raise StoryError('图片仍未上传完成，发送按钮不可用')
                    before = page.locator('[data-message-author-role="assistant"]:visible, [data-chatgpt-search-unit-key$=":assistant"]:visible').count()
                    galleries_before = page.locator('[data-testid="generated-image-gallery"]:visible').count()
                    # Persist before clicking: an uncertain click must never auto-submit twice.
                    save(status='submitting', assistant_count=before, gallery_count=galleries_before, prompt=prompt, original_sha256=sha,
                         watermark_regions=regions,masked_input_path=str(input_path) if regions else None)
                    send.click(timeout=15000)
                    save(status='submitted')
                    page.wait_for_url(re.compile(r'^https://chatgpt\.com/c/[^/?]+'),timeout=60000,wait_until='domcontentloaded')
                    save(conversation_url=page.url)
                elif job['status'] == 'submitting' or not url:
                    raise StoryError('上次提交状态不确定，请在专用 Chrome 核对该任务；程序不会重复发送')
                deadline = time.monotonic()+timeout
                while time.monotonic() < deadline:
                    if '/c/' in page.url and job.get('conversation_url') != page.url:
                        save(conversation_url=page.url)
                    if self._login_required(page):
                        raise StoryError('登录已失效，请手动登录后继续；已有任务不会自动重发')
                    alerts='\n'.join(page.get_by_role('alert').all_text_contents())
                    if '账户已切换' in alerts or 'account has changed' in alerts.lower():
                        if job.get('account_refresh_retries',0)>=1:
                            save(detail=alerts[:1000])
                            raise StoryError('网页账户状态仍未就绪，请在专用 Chrome 确认当前登录后继续')
                        retry=page.get_by_role('button',name=re.compile(r'^(重试|Retry)$',re.I))
                        if not retry.count():
                            raise StoryError('网页账户已切换，请刷新登录连接后继续')
                        save(account_refresh_retries=1)
                        retry.first.click()
                        try:
                            page.wait_for_url(re.compile(r'^https://chatgpt\.com/c/(?!local-chatgpt)[^/?]+'),timeout=60000,wait_until='domcontentloaded')
                        except Exception:
                            if 'local-chatgpt' in page.url:
                                save(status='failed_before_generation',detail='网页提示账户切换，重试后仍未创建服务器会话')
                                raise StoryError('网页账户切换导致请求未开始生成；请刷新登录连接后继续此任务')
                            raise
                        save(conversation_url=page.url)
                        continue
                    turns=page.locator('[data-message-author-role="assistant"]:visible, [data-chatgpt-search-unit-key$=":assistant"]:visible')
                    galleries=page.locator('[data-testid="generated-image-gallery"]:visible')
                    gallery_before=job.get('gallery_count',0 if job['assistant_count']==0 else None)
                    answer=None
                    if gallery_before is not None and galleries.count()>gallery_before:
                        answer=galleries.nth(gallery_before)
                    elif turns.count() > job['assistant_count']:
                        answer=turns.nth(job['assistant_count'])
                    if answer is not None:
                        images=answer.locator('img').evaluate_all('(imgs) => imgs.filter(i => i.complete && i.naturalWidth >= 256 && i.naturalHeight >= 256).map(i=>({src:i.currentSrc||i.src,width:i.naturalWidth,height:i.naturalHeight}))')
                        stopping=page.locator('[data-testid="stop-button"], button[aria-label="停止"], button[aria-label="Stop"], button[aria-label="Stop generating"]').count()
                        if len(images)==1 and not stopping:
                            destination.parent.mkdir(parents=True,exist_ok=True)
                            self._save_result(page, answer, images[0], destination)
                            with Image.open(destination) as cleaned, Image.open(original) as source:
                                cleaned.verify()
                            with Image.open(destination) as cleaned, Image.open(original) as source:
                                ratio=(cleaned.width/cleaned.height)/(source.width/source.height)
                                if abs(ratio-1)>.035:
                                    save(status='needs_review', detail='返回图片比例与原图不符', path=str(destination))
                                    raise StoryError('GPT 返回图片比例与原图不符，需人工检查')
                            save(status='complete', path=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                                 original_sha256=sha,options=options,provider='chatgpt_web',review_status='pending')
                            page.screenshot(path=str(checkpoint.with_suffix('.png')))
                            atomic_json(worker_file,{'conversation_url':page.url,'completed_key':key})
                            return job
                        if len(images)>1 and not stopping:
                            save(status='needs_review', detail='GPT 返回多张图片，需人工选择')
                            raise StoryError('GPT 返回多张图片，需人工选择后再采用')
                    responses=page.locator('[data-message-author-role="assistant"]:visible, [data-chatgpt-search-unit-key$=":assistant"]:visible')
                    if responses.count() and not page.get_by_role('button',name=re.compile(r'^(停止|Stop|Stop generating)$',re.I)).count():
                        response=responses.last.inner_text()
                        if re.search(r'不能.{0,30}(?:去除|移除|帮助)|无法.{0,30}(?:水印|生成|编辑)|达到.{0,15}(?:上限|限制)|cannot.{0,50}(?:remove|watermark)',response,re.I|re.S):
                            save(status='provider_refused',detail=response[:1000])
                            raise StoryError('GPT 网页未接受本次清图要求：'+response[:300])
                    if progress:
                        progress('正在等待 GPT 生成清图；可在专用 Chrome 中查看进度')
                    time.sleep(2)
                raise StoryError('等待 GPT 清图超时；任务已保存，重试会继续等当前会话')
        except Timeout as exc:
            raise StoryError('专用 Chrome 正在处理另一张图片，请稍后重试') from exc

    @staticmethod
    def _save_result(page, answer, image, destination):
        downloads=answer.get_by_role('button', name=re.compile(r'下载|Download', re.I))
        temporary=destination.with_suffix('.download')
        opened_viewer=False
        if not downloads.count() and urlsplit(image['src']).scheme!='https':
            preview=answer.get_by_test_id('generated-image-preview')
            if not preview.count():
                raise StoryError('当前图片尚无可下载地址，请在 Chrome 中完成生成后重试')
            preview.first.click()
            downloads=page.get_by_role('button',name=re.compile(r'^(下载|Download)$',re.I))
            downloads.first.wait_for(state='visible',timeout=15000)
            opened_viewer=True
        if downloads.count():
            with page.expect_download(timeout=60000) as received:
                downloads.first.click()
            received.value.save_as(str(temporary))
        else:
            # Only the actual new assistant image's DOM URL is fetched.
            address=urlsplit(image['src'])
            if address.scheme != 'https':
                raise StoryError('当前图片尚无可下载地址，请在 Chrome 中完成生成后重试')
            response=page.request.get(image['src'],timeout=60000)
            if not response.ok:
                raise StoryError('清图下载失败，请保留会话后重试')
            temporary.write_bytes(response.body())
        try:
            with Image.open(temporary) as picture:
                picture.convert('RGB').save(destination,format='PNG')
        finally:
            temporary.unlink(missing_ok=True)
            if opened_viewer:
                close=page.get_by_role('button',name=re.compile(r'^(关闭查看器|Close viewer)$',re.I))
                if close.count():
                    close.first.click()
