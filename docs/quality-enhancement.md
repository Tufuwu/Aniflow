# 本地智能画质增强

在 AniFlow 顶部打开「画质增强」，可批量选择图片、填写目标尺寸和最大缩放，并逐张查看画质诊断、处理方案和前后对比。单图可下载增强 PNG。打开「音视频生成」中的智能画质增强开关可在分镜片段和自动流水线中使用同一策略。「剧情分镜剪辑」的导出页也有独立开关。CapCut 导出沿用项目设置。

## 安装模型

1. 在 AniFlow 的 Python 环境运行 `python -m pip install -r requirements-enhancement.txt`。有 NVIDIA GPU 时，主环境应使用相互匹配且启用了 CUDA 的 PyTorch 和 torchvision。
2. 将 [RealESRGAN_x2plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) 和 [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) 放在 `models/enhancement/`。也支持同名的 [Kornia SafeTensors 转换版](https://huggingface.co/kornia/real_esrgan)，两个格式同时存在时优先读取 SafeTensors。
3. 需要人脸修复时，将 [CodeFormer](https://github.com/sczhou/CodeFormer) 克隆到 `models/enhancement/CodeFormer`，按照上游说明在单独的 Python 环境中安装依赖。可用 Python 3.10、PyTorch 2.1.2 和 torchvision 0.16.2 搭建兼容旧版 torchvision 接口的环境。通过上游 [预训练权重](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0) 准备这些文件：

   - `weights/CodeFormer/codeformer.pth`
   - `weights/facelib/detection_Resnet50_Final.pth`
   - `weights/facelib/parsing_parsenet.pth`

在 AniFlow 的 `.env` 中设置 `ANIFLOW_CODEFORMER_PYTHON` 为独立环境的 Python 可执行文件绝对路径。也可用 `ANIFLOW_ENHANCE_MODELS` 和 `ANIFLOW_CODEFORMER_DIR` 指定其他模型目录。权重下载完成后，无需在推理时联网。人脸保真度默认 0.9，配合对齐、面部遮罩和与原脸混合来减少身份漂移。插画人脸和极小人脸的检测与修复效果受模型限制。

## 配置和处理策略

项目设置和剧情剪辑导出请求接受 `quality_enhancement`：

```json
{
  "quality_enhancement": {
    "enabled": true,
    "face": false,
    "fidelity": 0.9,
    "device": "auto",
    "tile": 0,
    "max_zoom": 1.15
  }
}
```

默认关闭。没有提交该配置时，可用 `ANIFLOW_ENHANCE=1` 设置服务端默认值。页面中的显式开关会覆盖默认值。`device` 支持 `auto`、`cuda` 和 `cpu`。分块为 0 时按当前可用显存自动选取，非零值使用 32–1024 范围内的 16 的倍数。CUDA 推理显存不足时会缩小分块重试。可运行 `python -m tools.enhance_media hardware` 查看设备与可用显存。

素材需求为 `ceil(实际显示宽度 × 最大缩放)` 和 `ceil(实际显示高度 × 最大缩放)`。独立预览以填写的输出宽高为准。剪辑入口用实际裁切及素材在画框内的尺寸，缩放取镜头动画与配置上限的较大值。达到尺寸且没有明显劣化时，直接复用输入。尺寸不足时按需选择 2× 或 4×，超出 4× 能力时仍会标记尺寸不足。尺寸足够时按检测结果选择轻度去噪、去块或锐化。OpenCV 指标是启发式判断，漫画网点和规则线条可能影响评分。

图片结果以源文件哈希、裁切范围、处理方案、配置及模型文件版本作为缓存依据，独立写入 `output/enhancement/images/`。前后对比的原图预览副本写入 `output/enhancement/previews/`。CapCut 工程会引用缓存里的增强图片；如需把工程迁移到其他机器，请一并迁移这些素材。启用增强时，旧视频片段会重新渲染并复用图片缓存。

## 命令行与 Python 视频接口

```powershell
python -m tools.enhance_media images panel1.png panel2.jpg --target 1920 1080 --zoom 1.15
python -m tools.enhance_media video source.mp4 output/enhanced.mp4 --scale 2 --temporal 0.45
```

Python 接口为 `pipeline.quality_video.enhance_video(source, destination, scale=2, options=None, strength=.45, progress=None)`。输出独立的 H.264 MP4 并保留音轨（AAC）。相同输入和设置可复用 `output/enhancement/videos/` 中的结果。按固定帧率素材设计，变帧率输入会按读取到的帧率输出。时序处理只对超分辨率产生的细节层做光流对齐与融合，并在变动较大的区域抑制历史帧。快速运动、遮挡或复杂转场仍可能出现残余闪烁。图片转视频会在生成镜头运动前只增强一次静态素材。
