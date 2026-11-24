# AniFlow 独立视频后期剪辑

## 接入位置和实现阶段

AniFlow 的 Flask / Socket.IO 主应用通过 `pipeline/orchestrator.py` 调用 `pipeline/compositor.py`，从分格图片、配音和配置产出分格视频及整片。故事审核流程由 `pipeline/shot_pipeline.py` 读取已审核选镜与实测配音，使用 `timeline_director.py` 生成故事时间轴，再交由 `timeline_compositor.py` 导出。两条路径都可以把图片、分格 MP4 或整片 MP4 的路径交给后期剪辑。

1. `timeline.py` 定义版本化 Timeline、结构验证、镜头边界和工程读写。相对素材路径以工程所在目录为基准，自动规划写入绝对路径。
2. `media.py` 和 `render.py` 使用 FFmpeg 解码、编码和封装，使用 OpenCV 变换画面、合成各轨和检测边缘。预览与成片共用时间轴、音轨与合成逻辑。
3. `director.py` 从有序分镜、画面采样、运动量和主体框生成草稿，`tools.py` 提供 CLI 和 Agent 共用的 Python 入口。现有分镜、配音、视频任务状态与产物由原入口管理，后期剪辑通过单独命令启动。

## 使用方法

将分镜存为 JSON 数组。相对素材路径以这个分镜文件所在目录为基准。`duration` 和 `source_in` 在分镜输入中以秒计，`pace` 可选 `fast`、`normal` 或 `slow`，`scene_change` 表示此处是场景变化。可用 `subject` 补充人工确认的主体框。

```json
[
  {"path": "panel.png", "text": "她凝视远处", "subject": [0.22, 0.12, 0.46, 0.57]},
  {"path": "generated.mp4", "text": "转入回忆", "scene_change": true, "source_in": 0.5, "duration": 2}
]
```

```powershell
python -m pipeline.postproduction plan storyboard.json output/edit.json
python -m pipeline.postproduction validate output/edit.json
python -m pipeline.postproduction render output/edit.json output/preview.mp4 --preview
python -m pipeline.postproduction render output/edit.json output/final.mp4
```

预览长边最多 640 像素。正式输出默认 1920×1080，`plan --width 1080 --height 1920 --fps 30` 可建立竖屏工程。再次编辑时直接修改并保存 `edit.json`，用 `validate` 检查后渲染到新的 MP4 路径。素材保留在原有位置，移动工程及素材后可调整引用路径。`plan --replace` 会用新草稿替换已有剪辑工程。

默认在本地结合脚本关键词、跨帧变化、边缘分布与可用的动漫人脸检测规划。`plan --vision-model MODEL` 可选用现有 Ollama 或兼容 OpenAI 的 endpoint，并将分镜文本和素材缩略帧交给所选模型。人工给定的主体框和时长会优先使用，视频时长受素材末尾限制。模型建议经过枚举与几何检查，调用失败时保留本地决策并在 `decisions` 记录错误类型。

## Timeline JSON v1

| 字段 | 含义 |
| --- | --- |
| `format`、`version` | 固定为 `aniflow.postproduction` 和 `1` |
| `canvas` | `width`、`height` 为 2–4096 的偶数像素，`fps` 为 1–120 的整数 |
| `assets` | 素材 ID 到 `{kind, path}` 的映射，类型为 `image`、`video`、`audio` |
| `tracks` | 依次绘制的 `video` 轨，以及参与混音的 `audio` 轨，每条轨包含按时间排序的 `clips` |
| `decisions` | 自动规划的决策和画面分析，可选，供审核使用 |

每个镜头包含唯一 `id`、素材 ID `asset`、起始帧 `start` 和输出帧数 `duration`。`source_in` 默认 0，单位是按画布帧率换算的源媒体帧。`speed` 是由 `{frame, rate, ease}` 组成的数组，`rate` 默认 1，允许 0.125–8，视频与原声使用同一变速曲线。变速音频会随速度改变音高。`volume` 默认为 1，设为 0 可静音。独立音轨可引用音频文件或带声音的视频。

视频镜头的可选参数如下。

- `rect` 是镜头占据的画布区域，默认 `[0, 0, 1, 1]`。两条视频轨使用 `[0, 0, 0.5, 1]` 和 `[0.5, 0, 0.5, 1]` 可做左右分屏。把上层轨的区域设为 `[0.68, 0.05, 0.27, 0.27]` 可做画中画。`opacity` 默认为 1。
- `crop` 是素材的取景区，`subject` 是原始素材上的主体框。矩形格式统一为归一化的 `[x, y, width, height]`。`fit` 可选 `cover` 或 `contain`，后者使用模糊背景填充剩余画面。
- `camera` 列出 `{frame, x, y, zoom, rotation, ease}`。`x` 和 `y` 是取景区内归一化的运镜中心，`zoom` 相对于基础适配倍率，`rotation` 为度数。`shake` 为相对于取景区尺寸的抖动量，范围 0–0.05。`camera_preset()` 可生成快推、慢推、拉远、水平或垂直平移、旋转、特写、居中、抖动与静止的参数。
- `camera` 和 `speed` 的帧号均从镜头内第 0 帧开始递增。目标关键帧的 `ease` 控制前一段插值，可选 `linear`、`ease_in`、`ease_out`、`ease_in_out`，最后一个关键帧之后保持其值。
- `transition` 为入场转场，格式是 `{type, frames}`。`cut` 对应 0 帧，其余可选 `dissolve`、`black`、`slide`、`mask` 和 `zoom`，其中 `mask` 是水平遮罩擦除。转场两端使用相同 `rect`，时间轴重叠帧数等于 `frames`，黑场至少占 3 帧，其余转场至少占 2 帧。未放置镜头的画布区域为黑色。

渲染报告按镜头返回 `camera_clamped_frames` 和 `subject_cropped_frames`，分别统计为保证画面覆盖而限位的帧，以及已知主体框被裁切的帧。主体框来自人工、采样脸框或视觉模型，`subject_known` 为 `false` 时，诊断缺少主体依据。主体检查基于镜头的运镜和裁切，转场内的图形遮蔽属于转场效果。黑边按约半秒间隔抽样，分别报告源画面与镜头画面四边的最大黑边比例，顺序为左、上、右、下。黑边与主体框属于图像估计，素材暗场和快速主体运动会影响结果。

## MCP、Remotion 和 OpenCut

`plan_project`、`validate_project` 和 `render_project` 具有文件输入和可序列化返回值，可由未来 MCP 服务直接注册。当前入口用于本地可信调用。MCP 部署时可在服务层配置工作目录和素材访问范围，并将长渲染挂到任务队列。

Remotion 官方的 `renderMedia()` 可以接收 composition 和 `inputProps`，适合在浏览器预览或动态图形需要时，将同一 Timeline 转为 Remotion composition。当前 FFmpeg / OpenCV 渲染器已能复用项目的 Python 环境，Remotion 适配属于后续独立接入。 [Remotion renderer](https://www.remotion.dev/docs/renderer/render-media)

OpenCut 官方主分支处于重写阶段，其 README 将 Editor API、MCP、无界面运行列为计划能力。Classic 版可独立运行，项目级 JSON 导入导出也有上游议题。现阶段保持 AniFlow Timeline 自包含，待公开接口可用后，可先验证素材引用、裁切和硬切镜头的往返，再评估运镜及转场映射。 [OpenCut](https://github.com/OpenCut-app/OpenCut) [OpenCut Classic](https://github.com/OpenCut-app/opencut-classic) [项目导入导出议题](https://github.com/OpenCut-app/OpenCut/issues/719)
