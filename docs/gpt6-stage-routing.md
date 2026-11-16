# 第一至第四阶段使用 GPT-6 Luna

2026-11-16更新：第四阶段的 `script_polish` 与可选 `script_polish_analysis` 已独立配置为 `openai/gpt-6-luna`，通过OpenRouter调用。`STORY_POLISH_MODEL`、`STORY_POLISH_API_BASE`、`STORY_POLISH_API_KEY` 的本地值来自已配置的Luna连接；密钥只保留在Git忽略的 `.env`。运行中的服务需重启载入配置。

第四阶段留空时兼容回退原共享文字模型/连接；显式旧transport/writer覆盖仍有效，也可用 `polish_model/polish_transport` 单独注入。请求沿用现有Luna兼容策略和JSON输出，16000输出容量，实际用量与截断检测读取第四阶段自己的transport。前三阶段、原图核验、身份识别及摘要路由不受此次切换影响。缓存绑定实际第四阶段模型、连接与提示，旧缓存/历史稿保留；完成回执也记录实际模型指纹，避免把旧模型改写冒充新模型完成。

实测稿件和对照单独保存，未写回正式章节。下文为此前第一至第三阶段的配置说明，其中“第四阶段沿用原模型”的旧状态已由本段更新。


本地 `.env` 已为本话信息整合（含现有字段纠错）、叙事规划及讲稿写作配置 `openai/gpt-6-luna`，通过 OpenRouter 调用。配置仅在创建默认 `StoryModel(store)` 时生效。第三阶段涵盖整稿、单段写作和定向返修；图片分析、身份识别、文字审稿、摘要及严格原图核验继续使用原来的模型和连接。

```dotenv
# 第一阶段：本话信息整合及字段纠错
STORY_INTEGRATION_MODEL=openai/gpt-6-luna
STORY_INTEGRATION_API_BASE=https://openrouter.ai/api
STORY_INTEGRATION_API_KEY=在本地填写OpenRouter密钥

# 第二阶段：叙事规划
STORY_PLANNING_MODEL=openai/gpt-6-luna
STORY_PLANNING_API_BASE=https://openrouter.ai/api
STORY_PLANNING_API_KEY=在本地填写OpenRouter密钥

# 第三阶段：整稿、单段写作及定向返修
STORY_SCRIPT_MODEL=openai/gpt-6-luna
STORY_SCRIPT_API_BASE=https://openrouter.ai/api
STORY_SCRIPT_API_KEY=与前一阶段相同的本地OpenRouter密钥
```

地址不含 `/v1`，传输层追加 `/v1/chat/completions`。密钥只保存在 Git 忽略的本地 `.env`，不会进入缓存身份、模型请求记录或本说明。已运行的服务需要重启后读取新配置。

各阶段的 MODEL 留空时沿用原文字模型，API_BASE 留空时沿用原连接。代码显式传入旧 `transport` 或 `writer_model` 时保留原来的整体覆盖行为；可以用 `integration_model/integration_transport`、`planning_model/planning_transport`、`script_model/script_transport` 单独覆盖对应阶段。测试中注入模拟 transport 不会意外启用云端配置。

GPT-6 Luna 的请求沿用 JSON 输出与原阶段流程，省略不支持的 temperature/top_p，不发送 DeepSeek 的 thinking 参数，使用服务商默认 reasoning。叙事规划、整稿和定向返修沿用原 16,000 输出容量，单段写作沿用 6,000。事件整合及字段纠错使用已核实的 128,000 最大输出容量；既有 `STORY_INTEGRATION_MAX_OUTPUT_TOKENS` 显式配置仍优先。这是请求容量，不是事件数、字数或故事长度配额。

依据：[GPT-6 Luna 官方参数与输出容量](https://developers.openai.com/api/docs/models/gpt-6-luna)、[OpenRouter 省略采样参数的行为](https://openrouter.ai/docs/api-reference/parameters)。本次不修改角色提示、整合/规划 JSON、引用校验、重试逻辑或正式项目内容。

缓存身份及原始请求均记录对应阶段的实际模型、后端和地址；第一阶段还记录输出容量，GPT 请求记录 JSON/采样参数策略。切换后不会误用旧模型对应的缓存；其他阶段缓存身份保持一致。旧规划、整合结果、候选稿、历史版本、缓存及人工审核状态保留，配置切换不会自动重跑任何阶段。

离线验证覆盖默认正式入口的真实 HTTP 载荷、两个阶段独立认证头、其他阶段路由、兼容旧显式覆盖、字段纠错也走整合连接、模型切换后的原始/有效缓存隔离、旧文件保留，以及实际阶段 transport 的用量及截断检测。测试使用模拟服务端，不产生付费模型请求。


第三阶段采用单稿生成，双/三候选循环、比较模型调用、人工选稿接口及页面控件均已删除。默认生成只完成一次写稿及文字复核；严格核验使用 `/verify` 或 CLI `verify` 操作当前稿。单稿生成依赖包含实际写稿模型、连接、系统角色和请求策略；切换模型后重新写一份当前稿，旧版本和缓存继续保留，整合及规划结果可复用。本地 `.env` 中三个第三阶段配置值已复制自第二阶段；密钥不会进入说明或报告。

本轮仅模拟 HTTP 载荷并离线验证，没有发起真实模型或配音调用；写作质量须另行实测。模型兼容性依据 [OpenAI 官方 GPT-6 Luna 文档](https://developers.openai.com/api/docs/models/gpt-6-luna)，沿用已有 JSON 输出与默认 reasoning，不发送不适用的采样或 DeepSeek thinking 参数。启动中的服务需重启以载入新配置。
