const form = document.querySelector('#quality-form');
const statusNode = document.querySelector('#quality-status');
const results = document.querySelector('#quality-results');

function addText(parent, tag, text) {
  const element = document.createElement(tag);
  element.textContent = text;
  parent.append(element);
  return element;
}

function addComparison(card, item) {
  const {quality, plan} = item;
  const problems = [
    quality.blurry && '模糊', quality.noisy && '噪点', quality.blocky && '压缩块'
  ].filter(Boolean);
  const methods = [
    plan.scale > 1 && `Real-ESRGAN ${plan.scale}×`,
    ...plan.filters.map(filter => ({denoise: '降噪', deblock: '去压缩块', sharpen: '锐化'})[filter]),
    item.faces_restored && `CodeFormer 修复 ${item.faces_restored} 张脸`
  ].filter(Boolean);
  addText(card, 'p', `${quality.width}×${quality.height} → 素材需求 ${plan.required.join('×')} · ${problems.join('、') || '未检测到明显画质问题'}`);
  addText(card, 'p', `${methods.join('、') || '保留原图'}${plan.meets_resolution ? '' : ' · 素材尺寸仍低于需求'}${item.cache_hit ? ' · 已复用缓存' : ''}`);
  const comparison = addText(card, 'div', '');
  comparison.className = 'comparison';
  for (const [caption, url] of [['原图', item.before_url], ['处理结果', item.after_url]]) {
    const figure = addText(comparison, 'figure', '');
    const image = addText(figure, 'img', '');
    image.src = url;
    image.alt = caption;
    addText(figure, 'figcaption', caption);
  }
  if (item.enhanced) {
    const link = addText(card, 'a', '下载增强图片');
    link.href = item.after_url;
    link.download = 'enhanced.png';
  }
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = form.querySelector('button');
  button.disabled = true;
  results.replaceChildren();
  const fields = new FormData(form);
  const files = fields.getAll('image');
  const options = JSON.stringify({enabled: true, face: fields.has('face'),
    fidelity: Number(fields.get('fidelity')), max_zoom: Number(fields.get('zoom'))});
  try {
    for (const [index, file] of files.entries()) {
      statusNode.textContent = `正在处理 ${index + 1} / ${files.length}`;
      const card = addText(results, 'article', '');
      addText(card, 'h2', file.name);
      const body = new FormData();
      body.set('image', file);
      body.set('width', fields.get('width'));
      body.set('height', fields.get('height'));
      body.set('options', options);
      try {
        const response = await fetch('/api/quality-enhancement/preview', {method: 'POST', body});
        const item = await response.json();
        if (!response.ok) throw new Error(item.error || '增强失败');
        addComparison(card, item);
      } catch (error) {
        addText(card, 'p', error.message);
      }
    }
    statusNode.textContent = `已处理 ${files.length} 张图片`;
  } finally {
    button.disabled = false;
  }
});
