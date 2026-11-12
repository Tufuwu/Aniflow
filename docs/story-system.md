# 第一阶段：剧情理解与中文解说

实现范围是本地第一话的分镜分析、剧情证据库、解说稿、事实核验和人工审核。不调用 TTS、音频、视频、下载器或后续话。真正的内容质量需要真人按原作逐段验收，自动测试只能证明数据和流程约束。

## 仓库检查与复用决策

| 现有代码 | 已有能力 | 本阶段使用方式 |
| --- | --- | --- |
| `pipeline/llm_transport.py` | Ollama / OpenAI 兼容接口、图片消息、模型列表 | 直接复用；Story 支持独立云端配置，不改旧旁白配置 |
| `pipeline/narrator.py` | 图片编码、视觉与写稿分离、上下文旁白、单段重写 | 复用图片编码；旧的自由文本输出不作为可信事件 |
| `pipeline/scene_narrator.py` | 场景分组、视觉事件描述、故事上下文、音频 | 保留旧功能；固定长度分组不适合按剧情重要性规划，不调用音频 |
| `pipeline/script_rewriter.py` | 多风格自由改写 | 保留旧界面；缺少事件约束，新的中文三风格使用证据约束的 Script Writer |
| `pipeline/text_extractor.py` | 已裁切分镜的 OCR、对白缓存、多个 OCR 引擎 | 读取已有 `text` 作为未验证提示；修复了仅保留英文的提示及中文过滤 |
| `pipeline/stitcher.py` / `panel_extractor_yolo.py` | 拼接、检测、重叠框抑制 | 新本地输入适配器直接调用，重叠阅读窗口避免整条长图被过度缩小 |
| `PipelineState` / Review Text | 章节、原始带对白图、已有编辑界面 | 注册独立 Blueprint，原 Review Text 加载共享审核组件 |

## 分阶段交付

1. Story Analyzer + JSON Story Bible：分镜顺序、人物档案、动作、原文对白、关键事件、显式因果、来源哈希。事件及来源 ID 由程序分配；整话读完后复核封面/片尾边界，依据原文和原图补充确切姓名、亲属关系，未知身份继续保留未知。
2. Narrative Director + Script Writer：按重要性选取主事件，用结构化里程碑保证冲突、高潮、规则、目标/代价和剧情结尾不被漏掉。因果前提并入相关主事件上下文，不强制每个特写单独写一段。按阅读顺序组织开场、发展、冲突、高潮和结尾。支持忠实叙事、爽点解说、高悬念。
3. Fact Checker + Review Text：原图和事件对照的逐句核验，四类问题、来源查看、手工修改、驳回、重新核验、单段生成、通过锁定、历史查看、JSON 导出。
4. 离线约束测试、旧传输回归测试、原创人工验收样例、真实第一话试运行。

## 启动

完整 AniFlow 安装完 `requirements.txt` 后，进入原来的 Review Text 页面，新增的中文剧情审核区域和旧旁白功能同时可用。也可以使用没有音视频依赖的入口：

```powershell
python -m pip install flask requests Pillow python-dotenv filelock
python story_app.py import --project first-chapter --source "本地第一话目录"
python story_app.py serve
```

打开 `http://127.0.0.1:8081/story-review?project=first-chapter`。

原始切片先点击“拼接并检测分镜”（需要已安装仓库的 ultralytics / torch / OpenCV）。YOLO 框只是检测建议，输入适配器额外利用已有留白分割能力保留整条漫画，防止遗漏文字及安静场景；长画面使用有重叠的来源区域，并以 `boundary_uncertain` 标记不确定边界。区域可能包含多个相邻分镜，分析器需要分别提取其中事件。

来源区域尚未确认时可以预分析和生成草稿，但不能锁定审核。查看输入图片并核对完整性和阅读顺序后，再点击确认分镜。检测不理想时，使用原 AniFlow 的可编辑检测框调整，或将人工整理好的分镜目录导入新项目：

```powershell
python story_app.py import --project first-chapter-reviewed --source "完整分镜目录" --panels
```

`--panels` 表示输入已是完整分镜，不能用它将下载切片冒充完整分镜。检测窗口和自动框可能漏检或截断，分镜完整性需要人工核对。

随后点击“分析剧情”和“规划并生成解说”。核对每段原图、人物、动作和事实问题，必要时编辑后重新核验。模型核验通过才开放“审核通过并锁定”；锁定后不能编辑、驳回、单段重写或整稿覆盖。模型不可用、漏句、结果无效、存在不确定内容都不会视为通过。

也可在已经完成输入准备后运行 `python story_app.py run --project first-chapter --style faithful --max-events 12` 生成草稿。写作阶段若核验发现明确事实问题，会结合具体反馈最多改写两次；仍未通过的内容保留问题状态。单段重写同样接收上一轮核验反馈。核验通过只表示模型判断有依据，不等于人工审核完成。

## 模型配置

读取仓库根目录的本地 `.env`，已有环境变量优先。`.env` 在 `.gitignore` 中，密钥不能提交。

```dotenv
STORY_API_BASE=https://api.deepseek.com
STORY_API_KEY=自行在本地配置
STORY_VISION_MODEL=deepseek-flash
STORY_WRITER_MODEL=deepseek-flash
STORY_CHECKER_MODEL=deepseek-flash
```

云端地址不带 `/v1`，复用的兼容传输层会追加 `/v1/chat/completions`。事实核验也读取原图，因此 checker 必须支持图片。不同环节可以指定不同模型名。未设置 `STORY_API_BASE` 时，沿用现有 Ollama / 9router 选择。云端视觉模型会接收到相关分镜图片；使用同一个模型的分析和核验可能产生相同偏差，不能替代人工审核。

DeepSeek 官方端点使用 JSON 输出格式；抽取和写稿关闭额外思考，逐句核验启用低档思考。该选项仅发送给官方 DeepSeek 域名，不影响旧 Ollama/9router 行为。请求记录和缓存键包含这些选项与图片编码参数。

## 数据结构与追踪

每个项目固定存放于 `output/story/<project>/Chapter_00001/`：

```text
inputs.json                    # 只读导入的切片/分镜清单
evidence/<sha256>.<ext>         # 原始证据快照；每次查看验证哈希
preparation/<input_hash>/      # 拼接图、检测窗口、裁切图、框和原始清单
calls/<stage>-<uuid>.json      # 模型、提示词、图片哈希、原始响应或失败原因
cache/<request_hash>.json      # 内容、模型、后端、完整提示词共同决定的缓存
revisions/<uuid>.json          # 不变版本：bible、plan、script、审核结果
current.json                   # 原子更新的当前版本指针
.chapter.lock                  # 跨进程互斥锁
```

Story Bible 包含：`panels`（顺序、图路径、哈希、对白、识别疑点）、`characters`（稳定 ID、姓名、外貌、别名、身份证据）、`events`（事件 ID、动作、人物、重要性、叙事职能、原文/视觉依据、明确因果）。仅有时间顺序不建立因果。

Script 每段都有 `segment_id`、`event_id`、`source_panel_ids`、`context_event_ids`、`section`、`text`、`review_status`、`fact_check`。上下文事件只能来自明确因果前提，其原图也一并进入来源索引和事实审核。事实报告逐句记录文本、判定、问题类型和依据。模型无法改变段落的事件和来源关联。旧旁白不会写入或覆盖新 Story 数据，新 Story 也不会自动覆盖旧的 narration 字段。

每段写作和核验后立即保存版本；中断可查看已保存草稿。分析中的成功模型调用保留缓存，重试分析可复用。整章生成失败后重新生成会使用已缓存的相同调用。显式单段重写和重新核验会绕过响应缓存。所有编辑带当前版本号；旧页面提交被拒绝，防止覆盖别人刚审核的内容。模型调用记录不包含 API 密钥或请求认证头。

已审核段落受保护，整稿重做需要另建项目，旧文案保存在原项目和历史版本中。本阶段没有自动解锁已审核稿的入口。

## API

| 接口 | 用途 |
| --- | --- |
| `GET /api/story/<project>` | 当前状态、历史、输入类型、后台任务 |
| `GET /api/story/<project>?revision=<id>` | 查看历史版本 |
| `POST .../prepare` / `.../confirm_panels` | 拼接分镜 / 确认输入 |
| `POST .../analyze` | 分析剧情 |
| `POST .../generate` | 规划和写稿；可传 style 和 max_events |
| `POST .../segments/<id>/<action>` | edit / regenerate / check / approve / reject |
| `GET .../jobs/<id>` | 后台任务进度和错误 |
| `GET .../evidence/<panel_id>?revision=<id>` | 此版本原图证据 |
| `GET .../export` | 含来源、审核状态的最新 JSON；草稿状态不会被隐藏 |

所有写入请求需传 `revision`（初始为 null）。编辑另传 `text`。长任务返回 202 与 job_id。并行同章操作或版本冲突返回/报告冲突。重启后后台任务不会自行续跑，但已提交版本、模型调用和缓存都保留。

## 自动测试及人工验收

```powershell
python -m pytest tests/test_story_pipeline.py -q
python tests/test_llm_transport.py
node --check static/js/story_review.js
node --check static/js/app.js
```

`tests/fixtures/story_acceptance.json` 是原创三分镜样例，列出正确叙述及人物颠倒、事件反转、无依据亲属关系、错误因果、未来剧情等反例。离线测试使用假模型验证处理这些判定的机制，并不证明真实模型一定识别得出这些错误。

真实章节验收：

- 对照原图检查分镜完整性和阅读顺序，不能把下载切片边界当成分镜边界。
- 全部人物姓名/身份有画面或对白依据；同一人物身份稳定；身份未知时保持未知。
- 无对白动作不能漏；角色说法不直接升级为事实；只保留明确因果。
- 三种风格分别检查：关键事件忠于原作，开场进入冲突，高潮不夸大，结尾不编造后续。
- 每个句子的事实能回到对应事件和原图；每段点开证据、通过后锁定，再试重写应被拒绝。
- 对比原图哈希，检查单段重写不改变其他段，检查历史版本可恢复阅读。

自动检测、视觉分析和事实核验都可能出错。第一话的最终“已审核”状态必须由用户在界面中完成，不能用自动模型输出冒充人工验收。
