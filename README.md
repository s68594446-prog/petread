# petread

宠物行为判读。上传一段视频,按行为学编码表拆解可观察信号,给出带依据的状态候选。

信号不足时会明确说"看不出来",不会编一个结论。

## 它做什么

1. **视频预处理** — 逐帧算运动曲线,按运动峰值智能抽帧(不是均匀抽),提取 16kHz 音轨
2. **知识库注入** — 按物种加载行为学编码表(犬/猫),含每个信号的置信度分级
3. **模型判读** — 全模态模型一次完成:视觉信号编码 + 声学事件匹配 + 时序对齐 + 规则裁决
4. **约束与护栏** — 低置信度候选自动丢弃;建议不得包含"惩罚警告信号"这类会加剧问题的内容

## 快速开始

需要 Python 3.11+ 和 ffmpeg。

**1. 克隆并装依赖**

```bash
git clone git@github.com:<你的账号>/petread.git
cd petread
```

```bash
brew install ffmpeg                        # macOS;Linux 用 apt install ffmpeg
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> opencv-python 目前没有 Python 3.14 的 arm64 wheel,建议用 3.11–3.13 建 venv。

**2. 配置密钥**

密钥只从环境变量读,`config.yaml` 里只写变量名。复制 [.env.example](.env.example) 为 `.env` 并填入你的 key:

```bash
cp .env.example .env
# 编辑 .env,把 MIMO_API_KEY 换成真实的 key
```

`.env` 已在 `.gitignore` 中,不会被提交。启动时 app.py 会自动加载它(通过 python-dotenv),之后不用每次在命令行传 key。

> 也可以不建 `.env`,直接 `export MIMO_API_KEY='...'`;已存在的环境变量优先,不会被 `.env` 覆盖。

**3. 启动**

```bash
.venv/bin/python app.py
```

默认监听 `0.0.0.0:7860`,本机浏览器打开 <http://localhost:7860>。

端口优先级:`PORT` > `PETREAD_PORT` > 默认 7860。改端口用 `PORT=9000 .venv/bin/python app.py`,开发时热重载加 `PETREAD_RELOAD=1`。

## 手机在局域网访问

服务监听 `0.0.0.0`,所以同一 Wi-Fi 下的手机可以直接访问你电脑的内网地址。

**1. 查电脑的内网 IP**

```bash
ipconfig getifaddr en0            # macOS,Wi-Fi 通常是 en0;有线试 en1
```

```bash
hostname -I | awk '{print $1}'    # Linux
```

拿到形如 `192.168.1.23` 的地址。

**2. 手机浏览器打开** `http://192.168.1.23:7860`(换成你的实际 IP)。

录视频入口需要 HTTPS 或 localhost 才能调起摄像头;局域网 HTTP 下 `<input capture>` 仍可从相册选择已录制的视频。手机和电脑必须连同一个网段;连不上先检查电脑防火墙是否放行 7860 端口。

## 用法

浏览器打开后:选物种 → 选场景标签(可多选)+ 补充描述 → 录视频或拍照 → 开始判读。

处理需要 20–30 秒,页面有进度条和已用秒数,不会干等。

结果分三层,默认只展开第一层:

| 层 | 内容 |
|---|---|
| 一 | 翻译一句话 + 置信度环 + 跨时段稳定性 + 可以做什么 |
| 二 | 时间轴(视觉信号与声学事件对齐)、冲突裁决、判断依据、**反方论证** |
| 三 | 模型返回的原始 JSON |

第一层里置信度用一个进度环可视化,颜色随高低变化(绿/蓝/琥珀)。信息不足时显示暖色警示卡,
不伪装成有结论。

底部有"实际情况是什么"反馈框,填的内容会存进 `feedback` 表,用于后续校准。

**按设备隔离历史(不做账户):** 首次访问时前端生成一个随机 `client_id`(UUID)存进
localStorage,之后 `/analyze`、`/history` 都带上它。`/history` 只返回该设备自己的记录;
换个浏览器/设备就是另一份历史。没有 `client_id` 的老数据不会显示给任何人。

**导出校准数据(全量,跨设备):** 历史记录默认折叠(它是校准数据来源,不会删)。
展开后点"导出 CSV",下载 **所有设备** 的 records + feedback 宽表(每条一行,含设备ID、时间、
模式、模型判断、置信度、你的实际反馈、是否信息不足等),直接丢进表格软件回顾偏差。

导出**不按 client_id 过滤**(要看到全部数据),但用口令保护:`GET /export?key=<EXPORT_KEY>`,
与环境变量 `EXPORT_KEY` 比对。**`EXPORT_KEY` 未设置时 `/export` 返回 503(禁用),不留后门。**
前端导出按钮会提示输入口令并记在本地,失败自动清除重问。

**两处判读增强(native / quick 都生效):**
- **跨时段一致性**:视频会被切成 2–4 段分别判读,各段一致则上调置信度、矛盾则下调,
  让置信度反映跨时间的稳定性而非单个瞬间。单图/纯音频只有一个时间切面,标为"无法评估",
  不因"看着一致"而加分。
- **强制反方论证**:给出主判断后必须论证"什么情况下它是错的"——列出反方假设、若成立本应
  出现什么信号、以及这些反方信号在素材里是否真的出现。反方信号一旦出现就必须下调置信度。
  目的是压制"看着自信、实则证据不足"的判断。

## 五种模式

| 模式 | 触发 | 说明 |
|---|---|---|
| `native` | 默认(选视频) | 视频直传模型,音轨随视频一起处理。最快最省 |
| `frames` | 勾选"用抽帧链路" | 智能抽帧 + 分三次调用,用于对比采样策略 |
| `quick` | 选图片 | 单张图片。主轴通道(体态/耳位)之一可读即出候选,置信度上限 0.75 |
| `audio` | 选音频 | 纯音频,只做声学分析,`visual_signals` 为空,置信度上限 0.7 |
| `text` | 切到"文字描述" | 来不及拍时用文字描述观察。不走视觉/音频,直接文字推理,置信度上限 0.65 |

素材入口不强制调用摄像头:视频/图片/音频三个入口都是普通文件选择(可从相册或文件选),
另外单独给了"直接拍摄"的录像/拍照按钮,想现拍时才用。

**文字模式**(应对来不及拍摄):切到"✍️ 文字描述",写下你看到的动作和姿态(耳朵朝向、尾巴位置、
身体重心、有没有发声)。模型会把描述里的**客观观察**映射到编码表,并**严格区分**客观观察与
你自己下的结论——只用观察做判断,你的结论仅作参考。描述里缺了哪些关键信号会主动列出
(`missing_signals`),引导你补充。描述空泛到读不出主轴信号(比如只说"它好像不开心")时判信息不足,
不硬编结论。结果顶部会标注"基于文字描述,可靠度有限,拍视频更准"。

**各模态的证据门槛是分开校准的。** 单图的判据是:体态 / 耳位这两个主轴通道**至少一个**可读,
且可观测通道总数 **≥ 2**,即出候选(低置信度);位移信号、亲和信号、尾部/耳部 motion、
瞳孔、胡须这些单帧里天然读不到或极易被遮挡的通道**永不计入分母**。纯音频则整组视觉通道都不计入,
只看声学证据。insufficient 只在真正读不出主轴信号时触发(没有动物、严重模糊、主体全遮挡)——
否则只要不是视频就会被必然判成"信息不足",那是规则缺陷而不是事实。

实测同一段素材:native 约 12.6k tokens / 24s,frames 约 32.2k tokens / 72s。

## 上传限制

应用自身的上限:视频 20MB,图片 10MB。前端会先拦一次并提示裁剪,后端边写边计字节数再校验一次——不能只靠前端,直接打接口就绕过去了。

**部署到线上平台时,平台的反向代理可能还有一层请求体上限。** 大文件 POST 有可能收到 `413 request entity too large`。应用自身 20MB 通常没问题,但在线版建议把视频控制得更短(5–10 秒、10MB 以内)最稳。真遇到 413,就是撞到平台上限了,只能进一步压缩素材。

## 部署到 Railway(Docker)

本项目带 `Dockerfile`,Railway 可直接用它构建。仓库里的 `railway.json` 已声明用 Dockerfile 构建(否则 Railway 默认走 nixpacks,识别不了 ffmpeg / opencv 的系统依赖)。

**要点:**

- **端口不写死。** Railway 通过 `PORT` 环境变量注入端口,`app.py` 读 `os.environ["PORT"]`(缺省 7860),host 固定 `0.0.0.0`。Dockerfile 的 `EXPOSE` 只是文档提示,不绑定实际端口。
- `MIMO_API_KEY` 在 Railway 项目的 **Variables** 里配,代码照旧从 `os.environ` 读,不用改。
- 数据目录由 `DATA_DIR` 控制,Docker 里默认 `/app/data`。

**⚠️ 数据不持久:** Railway 服务的容器文件系统是临时的,**重新部署 / 重启后 SQLite 历史记录和上传的文件都会丢失**。这不是 bug。要保留数据,在 Railway 里加一个 **Volume** 挂到某个路径(如 `/data`),再把服务变量 `DATA_DIR` 设成那个路径。

网页端具体怎么连 GitHub、配变量、拿访问地址,见本次交付时单独给出的步骤说明。

## 命令行

不开服务也能跑:

```bash
.venv/bin/python -m pipeline.providers --video pet.mp4 --mode native --species dog --context "客厅,快递员按门铃"
```

会打印完整原始输出、token 用量、耗时和实际发送的 payload 结构(base64 已脱敏)。

首次接入新端点时先探测 base64 支持:

```bash
.venv/bin/python -m pipeline.providers --probe --video pet.mp4
```

视频预处理单独自测(抽帧结果会存到 `data/frames/` 方便肉眼检查):

```bash
.venv/bin/python -m pipeline.video pet.mp4
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 前端页面 |
| POST | `/analyze` | multipart 上传,返回 SSE 流(进度 + 最终结果) |
| GET | `/history?client_id=&limit=30` | 该设备的历史记录(无 client_id 返回空) |
| GET | `/export?key=` | 导出**全部设备**的 records + feedback 为 CSV(需 EXPORT_KEY 口令) |
| GET | `/record/{id}` | 单条详情(含反馈) |
| POST | `/feedback/{record_id}` | 用户自评 |
| GET/POST | `/profiles` | 宠物档案 |
| GET | `/healthz` | 健康检查,含 key 是否就位 |

`/analyze` 直接返回 `text/event-stream`,前端用 fetch + ReadableStream 读。EventSource 只能发 GET、带不了 multipart,所以没用它——代价是没有自动重连,一次判读 20–30 秒,断了重传即可。

## 项目结构

```
config.yaml            provider 配置,只存环境变量名
app.py                 FastAPI 服务端
index.html             单文件前端,内联 CSS/JS,无框架
knowledge/
  dog.yaml             犬类行为学编码表
  cat.yaml             猫科行为学编码表
pipeline/
  video.py             运动曲线、智能抽帧、运动学特征、音轨提取
  prompts.py           prompt 模板与知识库注入
  providers.py         模型调用层,provider 可切换
  analyze.py           判读编排,供服务端调用
  store.py             SQLite 持久层
data/                  上传素材、抽帧、数据库(已 gitignore)
```

## 换模型

`config.yaml` 的 `providers.omni` 段。`type` 支持 `openai`(OpenAI 兼容端点,MiMo 走这个)、`deepseek`、`gemini`、`mock`。模型名不在代码里硬编码。

把 `mock` 改成 `true` 可以在不发真实请求的情况下跑通整条链路,假数据由知识库现场生成。

## 已知限制

- 判读质量依赖知识库,而知识库是行为学通识的编码,不是诊断工具。涉及疼痛、疾病或持续性行为问题,请找执业兽医或行为专业人员。
- 单张照片模式缺少时间维度和声音,信号不足的概率明显更高,这是方法本身的限制,不是 bug。
- 运动强度的档位阈值(`video.py` 的 `_classify_intensity`)是按 320×180 灰度帧差定的,不同拍摄条件下基线会整体偏移,积累一批真实素材后建议重新标定。
