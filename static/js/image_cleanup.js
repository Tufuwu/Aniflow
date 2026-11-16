/* Optional ChatGPT web cleanup, using AniFlow's reusable Chrome login. */
window.ImageCleanup = (() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const enc = encodeURIComponent;
  let browserMessage = '打开程序专用 Chrome，登录一次；后续清图任务复用该窗口和登录状态。';
  const labels = {approved:'已审核采用',pending:'等待对照审核',keep_original:'人工选择保留原图'};
  function markRegions(url,input,label) {
    const dialog=document.createElement('dialog');
    dialog.innerHTML='<h2>标记预处理区域</h2><p>在原图上拖出矩形。只盖住需要修补的区域，避开脸、手和关键动作。</p><canvas style="display:block;max-width:100%;touch-action:none"></canvas><div class="row"><button data-save>保存标记</button><button data-clear class="secondary">清空</button><button data-close class="secondary">取消</button></div>';
    document.body.append(dialog);
    const canvas=dialog.querySelector('canvas'),ctx=canvas.getContext('2d'),picture=new Image();
    let regions=JSON.parse(input.value||'[]'),anchor=null,draft=null;
    function draw(){if(!picture.complete||!picture.naturalWidth)return;ctx.drawImage(picture,0,0,canvas.width,canvas.height);ctx.fillStyle='#ff00ff';[...regions,...(draft?[draft]:[])].forEach(([x,y,w,h])=>ctx.fillRect(x*canvas.width,y*canvas.height,w*canvas.width,h*canvas.height));}
    const point=event=>{const b=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-b.left)/b.width)),Math.max(0,Math.min(1,(event.clientY-b.top)/b.height))];};
    picture.onload=()=>{const scale=Math.min(760/picture.naturalWidth,Math.max(350,innerHeight-260)/picture.naturalHeight,1);canvas.width=Math.round(picture.naturalWidth*scale);canvas.height=Math.round(picture.naturalHeight*scale);draw();};picture.src=url;
    canvas.onpointerdown=event=>{anchor=point(event);canvas.setPointerCapture(event.pointerId);};
    canvas.onpointermove=event=>{if(!anchor)return;const [x,y]=point(event);draft=[Math.min(x,anchor[0]),Math.min(y,anchor[1]),Math.abs(x-anchor[0]),Math.abs(y-anchor[1])];draw();};
    canvas.onpointerup=()=>{if(draft&&draft[2]*canvas.width>=2&&draft[3]*canvas.height>=2&&regions.length<20)regions.push(draft);draft=null;anchor=null;draw();};
    dialog.querySelector('[data-clear]').onclick=()=>{regions=[];draw();};
    dialog.querySelector('[data-save]').onclick=()=>{input.value=JSON.stringify(regions);label.textContent=`标记预处理区域（${regions.length} 处）`;dialog.close();dialog.remove();};
    dialog.querySelector('[data-close]').onclick=()=>{dialog.close();dialog.remove();};dialog.oncancel=()=>dialog.remove();dialog.showModal();
  }
  function ready(state) {
    const setup=state.image_cleanup;
    if (!setup?.enabled) return true;
    const selected=[...new Set((state.groups||[]).flatMap(g=>g.selected.map(s=>s.panel_id)))];
    return selected.filter(pid=>setup.scope==='selected'||setup.panel_ids.includes(pid))
      .every(pid=>['approved','keep_original'].includes(state.clean_images?.[pid]?.review_status));
  }
  function mount(target, {state,api,disabled,action,message}) {
    const selected=[...new Set(state.groups.flatMap(g=>g.selected.map(s=>s.panel_id)))];
    const options=state.image_cleanup?.options||{watermark:true,dialogue:true,text:false};
    const image=pid=>`${api()}/evidence/${enc(pid)}?revision=${enc(state.revision)}`;
    target.innerHTML=`<section class="group"><h2>网页版 GPT 清图（可选）</h2>
      <p>专用 Chrome 由 AniFlow 启动并保存登录。清图将上传到该窗口中的 ChatGPT，返回后保留原图与副本供对照。</p>
      <div class="toolbar"><button data-browser="open" ${disabled()?'disabled':''}>打开专用 Chrome / 登录</button>
      <button data-browser="check" class="secondary" ${disabled()?'disabled':''}>检查登录连接</button>
      <button data-clean-disable class="secondary" ${disabled()||!state.image_cleanup?.enabled?'disabled':''}>本次使用原图</button></div>
      <p id="cleanup-browser" role="status">${esc(browserMessage)}</p>
      <div class="toolbar"><label><input type="checkbox" id="clean-watermark" ${options.watermark?'checked':''}>纯色遮盖与区域修补</label>
      <label><input type="checkbox" id="clean-dialogue" ${options.dialogue?'checked':''}>去对白气泡</label>
      <label><input type="checkbox" id="clean-text" ${options.text?'checked':''}>去旁白框 / 拟声文字</label>
      <label><input type="checkbox" id="clean-delete-chat" ${state.image_cleanup?.delete_chat_after!==false?'checked':''}>下载校验后删除本次网页聊天</label>
      <button data-clean-start ${disabled()?'disabled':''}>处理勾选图片 / 继续任务</button></div>
      <p>先在图片上标记待修补区域，程序生成纯色遮盖副本，再让模型修补色块。修补提示仅描述色块、背景纹理和构图保留要求。生成后对照审核，再用于成片。</p>
      ${state.image_cleanup?.error?`<p class="status error">${esc(state.image_cleanup.error)}</p>`:''}</section>
      <div class="library">${selected.map(pid=>{const item=state.clean_images?.[pid];return `<article class="panel-card">
      <label><input type="checkbox" data-clean-pid="${esc(pid)}" checked ${disabled()?'disabled':''}>来源 ${esc(pid)}</label>
      <p>${esc(labels[item?.review_status]||'尚未处理')}</p><p>原图</p><a href="${image(pid)}" target="_blank" rel="noopener"><img loading="lazy" src="${image(pid)}" alt="原始分镜 ${esc(pid)}"></a>
      ${item?.conversation_deleted?'<p>本次网页聊天已清理</p>':item?.conversation_deletion_error?`<p class="status error">图片已保存，聊天清理待处理：${esc(item.conversation_deletion_error)}</p>`:''}
      ${item?.relative_path?`<p>清图结果</p><a href="${api()}/images/${enc(pid)}?revision=${enc(state.revision)}" target="_blank" rel="noopener"><img loading="lazy" src="${api()}/images/${enc(pid)}?revision=${enc(state.revision)}" alt="清图结果 ${esc(pid)}"></a>`:''}
      <label>修改建议（可选）<textarea data-clean-feedback="${esc(pid)}" maxlength="2000" placeholder="例如：右下角色块边缘需与沙发纹理衔接。" ${disabled()?'disabled':''}></textarea></label>
      <input type="hidden" data-clean-regions="${esc(pid)}" value="${esc(JSON.stringify(state.image_cleanup?.watermark_regions?.[pid]||item?.watermark_regions||[]))}">
      <button class="secondary" data-clean-mark="${esc(pid)}" ${disabled()?'disabled':''}>标记预处理区域（${(state.image_cleanup?.watermark_regions?.[pid]||item?.watermark_regions||[]).length} 处）</button>
      <div class="row"><button data-clean-approve="${esc(pid)}" ${disabled()||!item?.relative_path||item.review_status==='approved'?'disabled':''}>审核采用</button>
      <button data-clean-original="${esc(pid)}" class="secondary" ${disabled()||item?.review_status==='keep_original'?'disabled':''}>保留原图</button>
      <button data-clean-fresh="${esc(pid)}" class="secondary" ${disabled()?'disabled':''}>重新生成</button></div></article>`;}).join('')}</div>`;
    const readOptions=()=>Object.fromEntries(['watermark','dialogue','text'].map(k=>[k,target.querySelector(`#clean-${k}`).checked]));
    const readRegions=ids=>Object.fromEntries(ids.map(pid=>[pid,JSON.parse(target.querySelector(`[data-clean-regions="${CSS.escape(pid)}"]`).value||'[]')]));
    target.onclick=async event=>{
      const button=event.target.closest('button');if(!button||button.disabled)return;
      if(button.dataset.cleanMark){const pid=button.dataset.cleanMark;return markRegions(image(pid),target.querySelector(`[data-clean-regions="${CSS.escape(pid)}"]`),button);}
      if(button.dataset.browser){
        button.disabled=true;
        try{
          const opening=button.dataset.browser==='open';
          const response=await fetch('/api/editing/image-browser'+(opening?'/open':'?check=1'),opening?{method:'POST'}:{});
          const result=await response.json();if(!response.ok)throw new Error(result.error||'浏览器连接失败');
          browserMessage=result.login==='ready'?'已连接程序专用 Chrome，可以开始清图。':result.message;
          target.querySelector('#cleanup-browser').textContent=browserMessage;
        }catch(e){message(e.message,true);}finally{button.disabled=false;}
        return;
      }
      if(button.hasAttribute('data-clean-start')){
        const ids=[...target.querySelectorAll('[data-clean-pid]:checked')].map(n=>n.dataset.cleanPid);
        if(!ids.length)return message('请勾选至少一张图片。',true);
        return action('images/clean',{options:readOptions(),panel_ids:ids.length===selected.length?null:ids,watermark_regions:readRegions(ids),delete_chat_after:target.querySelector('#clean-delete-chat').checked});
      }
      if(button.dataset.cleanFresh)return action('images/clean',{options:readOptions(),panel_ids:[button.dataset.cleanFresh],fresh:true,feedback:target.querySelector(`[data-clean-feedback="${CSS.escape(button.dataset.cleanFresh)}"]`).value,watermark_regions:readRegions([button.dataset.cleanFresh]),delete_chat_after:target.querySelector('#clean-delete-chat').checked});
      if(button.dataset.cleanApprove)return action(`images/${enc(button.dataset.cleanApprove)}/approve`);
      if(button.dataset.cleanOriginal)return action(`images/${enc(button.dataset.cleanOriginal)}/keep_original`);
      if(button.hasAttribute('data-clean-disable'))return action('images/all/disable');
    };
  }
  return {mount,ready};
})();
