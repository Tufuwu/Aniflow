# 使用可复用的 Chrome 进行 GPT 网页清图

在“剧情分镜剪辑 → 图片清理”中点击“打开专用 Chrome / 登录”。这是 AniFlow 自己启动的可见 Chrome，与 Codex 的浏览器连接无关。请在打开的这个窗口中手动登录 ChatGPT，再点击“检查登录连接”。后续程序继续使用同一个配置目录，关闭网页、退出 AniFlow 或重启程序后也可重新连接。

默认配置目录是 `output/browser/chatgpt/profile`，调试连接仅使用本机 `127.0.0.1:9225`。不读取或复制平时 Chrome 配置中的 Cookie。Chrome 136 起要求远程调试使用独立的用户数据目录，因此已有普通 Chrome 窗口中的登录状态不会自动继承。

对需要局部修补的图片，先点击“标记预处理区域”，在原图上拖出矩形。程序保存原图，另外生成纯洋红色矩形覆盖的副本，再上传给模型。区域修补采用独立提示词，只要求去掉色块、还原周围纹理并保留人物与构图，不带水印、平台或版权相关表述。可同时选择清除对白气泡、旁白框及拟声字。

点击“处理勾选图片 / 继续任务”后，程序逐张上传到网页版 GPT 的聊天页面，下载对应新回复中的图片。每张图有独立任务记录和会话地址；超时后可继续等待同一任务，提交状态不确定时会停止并提示核对，不自动重复发送。网页拒绝或限流会明确保留记录并停止，不会无限等待。

清图默认关闭，不改变 AI 选镜开关的默认值。只处理已选来源，原图证据始终保留。生成结果存入当前章节的 `editing/media/image_cleanup`，不会自动认定审核通过。对照原图选择“审核采用”后，镜头预览和视频导出使用对应清图；也可以明确选择“保留原图”或“本次使用原图”。清图未审核、来源变化或文件校验失败时，导出会提示处理，不能静默回退。

登录、验证码、限流或页面结构改变可能需要手动处理。浏览器已连接不等于已登录；状态接口分别返回连接与页面就绪情况。清理后只剩空白的纯文字素材需要在镜头审核中换用相关人物、动作或环境画面，程序不会凭空绘制剧情。

任务收尾可清理生成图片的网页聊天。程序先验证任务完成、本地图片存在且 SHA256 一致，然后只按该任务保存的会话地址删除；未完成任务继续保留以便续接。本地图片、纯色遮盖坐标和处理记录保留。删除命令需要显式的 `--confirm-delete`，防止误删会话。

可选 `.env` 配置：

```dotenv
CHATGPT_CHROME_PATH=C:/Program Files/Google/Chrome/Application/chrome.exe
CHATGPT_BROWSER_DIR=D:/AniFlow-browser/chatgpt
CHATGPT_BROWSER_PORT=9225
```

命令行也使用相同配置，可与界面互相复用：

```powershell
python tools/chatgpt_images.py open
python tools/chatgpt_images.py status
python tools/chatgpt_images.py edit --input original.png --output cleaned.png --watermark --dialogue
```

依赖 `playwright` Python 包，连接本机已安装的 Chrome；不需要安装 Playwright 自带浏览器。

参考：[Chrome 远程调试的配置目录要求](https://developer.chrome.com/blog/remote-debugging-port)、[Playwright 连接已有 Chrome](https://playwright.dev/python/docs/api/class-browsertype#browser-type-connect-over-cdp)。
