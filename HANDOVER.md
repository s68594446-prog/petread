# HANDOVER — petread 交接文档

> 用途:在新的对话 / 新的 Claude Code 会话里快速恢复上下文。
> 基于当前代码真实状态写(最后核对:2026-07-27)。读完这份就能接着干,不用再翻一遍代码。

---

## 一、项目概述

**是什么。** 宠物(猫/狗)行为判读工具。用户拍视频 / 拍照 / 录音 / 打字描述,系统基于内置的行为学知识库,把画面和声音里**可观察的信号**拆解编码,再推理出情绪与状态候选,并给出主人可以做什么的建议。

**核心设计原则**(整套 prompt 和护栏都围绕这四条,改动时不要破坏):

1. **先报告观察到的信号,再推理。** 视觉/声学两步只做形态编码,严禁出现情绪词;情绪结论只允许出现在候选的 label、translation、suggestion 里。
2. **证据不足时拒判,不硬猜。** `insufficient_evidence=true` 是合法且有价值的结论。看不清就写"不可观测",不是失败。
3. **概率化输出,而非确定结论。** 每个候选带 0–1 的 confidence 和逐条 evidence(通道/观察值/时间戳/知识库依据/信号置信度),低于阈值的候选直接丢弃。
4. **反方论证降低过度自信。** 给出主判断后强制列出"它在什么情况下是错的",反方信号一旦真的出现就必须下调置信度。配合"跨时段一致性自检",压制"看着自信、实则证据不足"的判断。

---

## 二、技术栈与架构

**模型:小米 MiMo-V2.5**(全模态,OpenAI 协议兼容,一次调用完成视觉+声学+融合)。

- `base_url`: `https://api.xiaomimimo.com/v1`(在 `config.yaml`)
- 模型名: `mimo-v2.5`
- 鉴权: API key 只从环境变量 `MIMO_API_KEY` 读,`config.yaml` 里只写变量名。`auth_header: bearer`,遇 401 会自动换成 `api-key: <key>` 形式重试一次并打印生效的那种。
- 其他参数(经 `extra_body` 并入请求体顶层):`thinking: true`、`fps: 2`(native 模式视频采样帧率)、`media_resolution: default`、`timeout: 300s`、`max_retries: 3`(只对网络错误和 5xx 重试,4xx 不重试)、`confidence_floor: 0.5`。

**后端 FastAPI + 前端单文件 HTML + SQLite 存储。** 前端无框架、内联 CSS/JS。

**判读走"后台任务 + 可重连 SSE"两步(重要,这是最新架构):**

1. `POST /analyze`(multipart)只做校验、存上传、建任务,用 `asyncio.create_task` 在**后台**独立跑判读,**立刻返回 `{"job_id": ...}`**(JSON,不是流)。判读跑完会写库,全程不依赖任何 HTTP 连接是否还开着。
2. `GET /analyze/stream/{job_id}`(SSE)订阅进度。事件按下标缓存在任务里,连接从下标 0 重放,所以**断线重连、页面重载都能拿同一个 job_id 把进度和最终结果重新取回**。前端用 `EventSource`(它自带断线自动重连)。

> 早期版本是"`POST /analyze` 直接返回 SSE 流,前端 fetch + ReadableStream 读",图省事没拆两步。手机切屏会把页面挂起、掐断连接,结果发不回来、连库都写不成——这个坑现在已修,见第三节。任务存进程内存 `_JOBS`,完成后保留 30 分钟供重连;当前单进程 uvicorn 够用,多副本部署要换外部存储(Redis 等)。

**各文件职责(一句话):**

| 文件 | 职责 |
|---|---|
| `pipeline/video.py` | 视频预处理:帧差运动曲线、按运动峰值智能抽帧、运动学特征(强度/节律/周期性)、ffmpeg 抽 16kHz 音轨。对损坏/超短/无音轨视频优雅降级,不抛异常。 |
| `pipeline/providers.py` | 模型调用层。provider 可切换(openai / deepseek / gemini / mock),模型名全来自 config。含 JSON 解析容错、低置信度候选过滤 `filter_low_confidence`、suggestion 安全护栏 `check_suggestion_safety`、base64 支持探测 `probe_base64_support`。所有公开方法只返回 dict,不抛异常。 |
| `pipeline/prompts.py` | prompt 模板 + 知识库注入。核心是 `omni_prompt`(全模态单次调用的七步 prompt)。**各模态的充分性判据和置信度上限都在这里的 `_sufficiency_rules`。** |
| `pipeline/analyze.py` | 判读编排层,把 video/prompts/providers 串起来供服务端调用,只通过 progress 回调汇报进度。各模式的分支入口(`_run_native`/`_run_quick`/`_run_audio`/`_run_text`/`_run_frames`)。 |
| `app.py` | FastAPI 服务端。后台任务注册表 `_JOBS` + `_run_analysis_job`(判读跑完并写库,独立于连接)、`POST /analyze`(建任务返回 job_id)、`GET /analyze/stream/{job_id}`(可重连 SSE)、上传校验、历史/反馈/导出/档案接口。 |
| `pipeline/store.py` | SQLite 持久层(标准库 sqlite3,无 ORM)。records / feedback / profiles 三张表 + 迁移 + 导出。 |
| `knowledge/dog.yaml`、`knowledge/cat.yaml` | 犬/猫行为学编码表(见第七节)。 |
| `index.html` | 单文件前端。三层结果展示、置信度环、反馈框、历史、导出、client_id 生成。 |

**模式(代码里是 5 种,`analyze.MODES`):**

| 模式 | 触发 | 说明 |
|---|---|---|
| `native` | 默认(选视频) | 整段视频直传模型,由模型按 `fps` 自采样;**音轨包含在视频内,不单独拆**。最快最省。 |
| `quick` | 选图片 | 单张静态照片,无时间维度无音轨。 |
| `audio` | 选音频 | 纯音频,只做声学分析,`visual_signals` 强制置空。 |
| `text` | 切到"文字描述" | 来不及拍时用文字。不走视觉/音频链路,直接文字推理。 |
| `frames` | 勾选"用抽帧链路" | 智能抽帧 + 分三次调用(视觉→声学→融合)。**调试/对比采样策略用**,不是主路径。 |

> 你的规划里说的是"四种模式"(native/quick/audio/text)——那是给用户的四条主路径。`frames` 是额外保留的对照实现,平时用不到。

---

## 三、关键设计决策与踩过的坑(最重要,避免重蹈覆辙)

**为什么用 MiMo。** 全模态(视觉+音频一次调用搞定,不用拼三段)且便宜。原本的架构是三段式:vision 走 DeepSeek、audio 走 Gemini、text 走 DeepSeek —— 这套配置还在 `config.yaml` 里注释着,MiMo 不可用时可切回(需同时把 pipeline 的调用改回 `call_vision`/`call_audio`/`call_text`)。排除理由:三段式要维护三个 key、三次网络往返、还要自己做跨模态时序对齐,成本和复杂度都更高;MiMo 一次调用把视觉编码+声学编码+时序对齐+规则裁决全包了。

**智能抽帧按运动峰值,而非均匀抽。** `video.py` 先算逐帧帧差运动曲线,在曲线上找峰值,每个峰值前后各 ±0.2s 三连拍(pre/peak/post)+ 首尾基线帧,去重后按运动强度截断。目的是抓住瞬时信号(扑咬、僵直、尾巴动作的那一下),均匀抽会正好漏掉。只有曲线方差极低(几乎静止)时才退化为均匀抽帧。视频超过 15s 会先截取运动最密集的 10s 窗口。

**native 模式音轨自动处理,不需要单独拆音轨。** 整段视频直接交给模型,音轨随视频一起进去。只有 `frames` 模式才用 ffmpeg 单独抽 16kHz 单声道 wav。

**判读不能挂在一条 HTTP 连接上——切屏就丢结果的坑。** 早期 `POST /analyze` 直接返回 SSE 流,判读全程跑在这一条连接里。手机切到别的 App / 锁屏时,浏览器会挂起页面、掐断连接:结果发不回来,而且存记录的代码在流的末尾、连接断了根本不执行——**模型照跑照花 token,结果和校准数据全丢**。修正:拆成"后台任务 + 可重连 SSE"(见第二节)。判读用 `asyncio.create_task` 在后台独立跑、跑完就写库,与任何连接的存活无关;前端用 `EventSource` + localStorage 存 `job_id`,断线/重载后 `recoverPendingJob()` 拿 job_id 重连,服务端从事件下标 0 重放,幂等。**原则:任何耗时任务的"完成"和"写库"都不能依赖客户端连接还开着。**

**护栏写过头会系统性压制正常输出——这是踩过最深的坑,修正原则务必记住:**

- **suggestion 子串匹配的坑。** 早期护栏用简单子串匹配禁用词("制止""阻止"等),结果把"阻止孩子继续接近""不要制止它低吼"这类**正确**建议也误杀了。修正:`check_suggestion_safety` 现在做**否定感知**(动词前 8 字窗口内有"不要/别/避免"等否定词就不算违规)+ **作用对象邻近判定**(禁用动词只有作用在"警告信号"14 字半径内才算违规)。改护栏时必须保住这两层,否则会重新误杀。区分"永远禁止的处置动词"(惩罚/呵斥/打它…)和"只有作用于警告信号才禁止的动词"(制止/压制/阻止…)。
- **quick 模式 insufficient 门槛过严的坑。** 早期所有模态共用视频的门槛("可观测通道 < 3 就判证据不足"),导致**只要不是视频就几乎必然被判信息不足**——单图天然缺时间维度和声音,永远够不到 3 通道。那是规则缺陷,不是事实。修正见下条。

**充分性判据:按"主轴信号有没有",而非"通道数量"。** 判断证据是否充分,看的是判读主轴(体态 body_posture / 耳位 ear_position)读不读得到,而不是数够几个通道。各模态门槛是**分开校准**的(全在 `prompts.py` 的 `_sufficiency_rules`):

- **单图:** 体态/耳位这两个主轴通道**至少一个可读**,且可观测通道总数 **≥ 2**,即出候选。位移信号、亲和信号、尾/耳的 motion 子维度、瞳孔、胡须这些单帧天然读不到或极易遮挡的通道**永不计入分母**(读不到不扣分,读到了算加分)。insufficient 只在真读不出主轴时触发(没有动物、严重模糊/过曝、主体全遮挡)。
- **纯音频:** 整组视觉通道都不计入分母,只看声学证据。知识库里高置信度的发声(犬低吼、猫 hiss)可单独支撑候选。
- 反向也要防:证据确实充分时**不得**为了显得谨慎而判证据不足,那同样是错误输出。

**各模式置信度上限及原因**(在 `_sufficiency_rules`,改模型或改 prompt 时别丢):

| 模式 | 上限 | 原因 |
|---|---|---|
| `quick`(单图) | **0.75** | 静态素材无法区分"瞬间姿态"和"持续状态",也看不到变化方向。仅靠两个主轴通道出的候选建议落 0.5–0.65。结论只描述该瞬间,不推断前后。 |
| `audio`(纯音频) | **0.70** | 缺视觉通道无法交叉验证,也用不上大部分 context_modifiers。声学上无法区分的模式(玩耍性低吼 vs 威胁性低吼)必须并列多候选。 |
| `text`(文字) | **0.65** | 文字描述必然有信息损失,且依赖用户观察准确度。仅凭单一主轴观察时落 0.5–0.6。严格区分"客观观察"(可映射编码表、作依据)和"用户结论"("它好像不开心"——只记录、绝不作依据、不因用户已下结论就给高置信度)。 |

> native(视频)不设人为上限,因为它有完整的时间+声音维度,还能做跨时段一致性自检。

---

## 四、当前状态

**已完成的功能:**

- 五种模式(native / quick / audio / text / frames)全部打通,mock 模式可不发真实请求跑通全链路(假数据由知识库现场生成)。
- 视频预处理:运动曲线、智能抽帧、运动学特征(强度/节律/周期性 FFT)、音轨提取,全部对异常素材优雅降级。
- 全模态七步 prompt:视觉编码 → 声学编码 → 时序对齐 → 规则裁决(context_modifiers + conflict_rules)→ 候选与充分性判定 → **跨时段一致性自检** → **强制反方论证**。
- 后处理护栏:低置信度候选过滤、suggestion 安全检查(否定感知 + 邻近判定)。
- 前端三层结果展示(翻译+置信度环+稳定性+建议 / 时间轴+冲突+依据+反方 / 原始 JSON)、进度条+已用秒数、反馈框、历史列表、CSV 导出(口令保护)。
- 接口:`POST /analyze`(建后台任务,返回 job_id)、`GET /analyze/stream/{job_id}`(可重连 SSE 进度)、`/history`、`/export`、`/record/{id}`、`/feedback/{id}`、`/profiles`、`/healthz`。
- **判读后台化 + 可重连:** 切屏、锁屏、断网、整页重载都不再丢结果;记录一定落库(与连接存活无关)。无头脚本验证过"中途断连→同 job_id 重连→结果照常返回+已写库"。
- 上传双重校验(前端拦一次 + 后端边写边计字节数):视频 20MB / 图片 10MB / 音频 15MB。

**部署:Railway,Dockerfile 构建。**

- 仓库带 `Dockerfile` + `railway.json`(声明用 DOCKERFILE 构建,否则 Railway 默认走 nixpacks 识别不了 ffmpeg/opencv 的系统依赖)。系统依赖:ffmpeg、libgl1、libglib2.0-0。
- **端口不写死:** Railway 通过 `PORT` 环境变量注入,`app.py` 读 `os.environ["PORT"]`(缺省 7860),host 固定 `0.0.0.0`。端口优先级 `PORT` > `PETREAD_PORT` > 7860。
- `MIMO_API_KEY`(和建议设的 `EXPORT_KEY`)在 Railway 项目 Variables 里配。
- **访问地址:** 不在仓库里记录(README 说"见交付时单独给出的步骤说明")。恢复上下文时如果需要线上地址,去 Railway 面板看,或问用户。
- **已知问题:国内访问不稳。** 线上平台的反代还可能有一层请求体上限,大文件 POST 可能 `413`;在线版建议视频控制在 5–10s、10MB 以内最稳。

**数据持久化 & 按设备隔离历史:**

- **持久化:** SQLite + 上传文件都放 `DATA_DIR`(默认项目内 `./data`,Docker 内默认 `/app/data`)。**⚠️ Railway 容器文件系统是临时的,重新部署/重启后 SQLite 和上传文件都会丢——这不是 bug。** 要保留数据必须在 Railway 加 Volume 挂到某路径,再把 `DATA_DIR` 设成那个挂载点。**目前是否已挂 Volume,需去 Railway 面板确认**(仓库里看不出来)。
- **按设备隔离:** 已实现。前端首次访问生成随机 `client_id`(UUID)存 localStorage,之后 `/analyze`、`/history` 都带上。`/history` 只返回该设备自己的记录;没有 client_id 的老数据不返回给任何人。`store.py` 有对应迁移(老库 `ALTER TABLE` 补 `client_id` 列)。
  - **⚠️ 注意:** 这套 client_id 隔离 + EXPORT_KEY 保护的代码,截至本文写作时**在工作区里已写好但尚未 commit**(涉及 `app.py`、`index.html`、`pipeline/store.py`、`.env.example`、`README.md`)。新会话如果 `git log` 里看不到它,先 `git status` 看是否还挂在工作区未提交。

---

## 五、已知限制与未验证项

**只在合成/少量素材上验证过,缺真实宠物数据验证的部分——逐条列出该重点看什么:**

1. **运动强度档位阈值**(`video.py` 的 `_classify_intensity`,轻微/中等/剧烈)是按 320×180 灰度帧差经验定的。不同拍摄距离、光照、分辨率下基线会整体偏移。**重点看:** 攒到一批真实素材后,同一动作在不同拍摄条件下被分到的档位是否一致;偏了就重标定阈值。
2. **周期性检测**(`_dominant_frequency`,FFT 主频)理论上能抓摇尾巴这类周期动作,但没在真实摇尾视频上验证过主频落点是否合理。**重点看:** 真实摇尾/颤抖素材的 `periodicity_hz` 是否落在预期频段。
3. **智能抽帧是否真的抓到了关键瞬间。** 峰值检测的 height/distance 参数是经验值。**重点看:** `python -m pipeline.video pet.mp4` 会把抽帧结果存到 `data/frames/` 供肉眼检查——真实素材里"扑咬/僵直/龇牙"那一帧有没有被抽中。
4. **各模态置信度上限(0.75/0.7/0.65)是拍脑袋定的**,没有真实数据支撑这些具体数值。**重点看:** 攒够反馈后,对照"模型判断 vs 用户实际反馈",看每个模态的实际准确率与置信度是否匹配,再回调上限。
5. **充分性判据(尤其单图"主轴≥1 且可观测≥2")的松紧**没经真实数据检验。**重点看:** 真实单图里判 insufficient 的比例是否合理——太高说明门槛仍偏严,太低说明在证据不足时硬给了结论。
6. **护栏的误杀/漏杀。** 否定感知和邻近判定的窗口大小(8 字 / 14 字)是经验值。**重点看:** 导出的 suggestion 里有没有被误杀的正确建议,或漏网的"惩罚警告信号"类建议。
7. **知识库本身的覆盖度。** 编码表是行为学通识的编码,不是诊断工具;真实个体的品种形态学差异(垂耳/断尾/扁脸/长毛)对通道置信度的削弱是否被正确降级,缺真实样本验证。

---

## 六、下一步计划(明确)

**当前阶段是"用一周攒校准反馈",不是继续加功能。** 功能面已经够用,现在缺的是真实数据。不要急着加新模式/新通道。

1. **攒反馈。** 让真实使用产生 records + feedback(用户在结果底部"实际情况是什么"里填)。目标二三十条带反馈的记录。
2. **导出。** 攒够后 `GET /export?key=<EXPORT_KEY>` 导出**全部设备**的 records+feedback 宽表 CSV(每条一行,含设备ID/时间/模式/模型判断/置信度/用户实际反馈/是否信息不足等),直接丢表格软件回顾。
3. **基于真实数据做个体校准:** 找系统性偏差(比如"这只狗的低吼总被判成威胁,实际是玩耍""这只猫的某姿势老被高估紧张度")→ 把发现写进知识库。
4. **个体校准写在哪里(见第七节末):** 目前知识库**还没有**专门的"个体档案"段落。可选两条路:(a) 直接在对应物种的 yaml 里调 values / value_confidence / context_modifiers / conflict_rules;(b) 新增一个个体档案段落并在 `prompts.py` 里注入(目前 `profiles` 表存了 `calibration_notes` 但**没有喂给模型**,见待办)。

**待办:**

- **确认数据持久化是否真的生效**(Railway 是否挂了 Volume 并把 `DATA_DIR` 指过去)。没挂的话,攒的反馈会在下次部署时丢——这直接威胁第六节整个计划,**优先级最高**。
- **确认 client_id 隔离 + EXPORT_KEY 那批改动已 commit 并 push**(见第四节末的警告)。
- **把 `profiles` 表接进判读链路。** 现在 `/profiles` 能存 `calibration_notes`,但 `analyze.py`/`prompts.py` 完全没读它,个体校准备注等于白存。要么把它注入 prompt,要么明确走"改 yaml"这条路、把 profiles 表下掉。

---

## 七、知识库结构说明

`knowledge/dog.yaml`(419 行)、`knowledge/cat.yaml`(431 行)。两个文件结构一致,顶层字段:

| 字段 | 含义 | 能不能手调 |
|---|---|---|
| `meta` | 元信息:`species`、`framework`(犬=Dog Body Language Ladder)、`confidence_scale`(高/中/低各自定义)、`reading_order`(判读主轴顺序:先体态→再距离调节→最后单点信号)、`hard_constraints`(硬约束,如"任一单通道不足以定论,至少三通道一致")。 | `reading_order`、`hard_constraints`、`confidence_scale` 都可调,但它们影响全局判读逻辑,慎改。 |
| `signals` | 视觉信号编码表,按通道组织。犬通道:`ear_position`、`tail`、`body_posture`、`mouth`、`eye`、`displacement_signals`;猫通道:`ear_position`、`tail`、`pupil`、`whisker_position`、`body_posture`、`affiliative_signals`、`displacement_signals`。每个通道有 `values`(允许取值)、`confidence`/`value_confidence`(通道或取值的置信度)、`caveats`(品种形态学等注意事项);`tail` 等有 `dimensions` 子维度。 | **主要手调对象。** 调 values 措辞、value_confidence 等级、caveats 都在这里。 |
| `acoustic` | 声学模式表(pattern 列表)。犬 9 种(低频单音长吠/低吼/玩耍性低吼/呜咽…),猫 10 种(purr/meow/hiss/growl/yowl/pain shriek…)。每项有 f0_range、confidence 等。 | 可调 f0_range 分档、confidence。 |
| `context_modifiers` | 情境改写规则:命中 `override_when` 条件时按 `override_to` 重新解读(处理"形态相同、含义相反"的情形)。 | **个体校准的主要落点之一。** |
| `conflict_rules` | 信号冲突裁决:出现 `conflicting` 信号组合时按 `resolution` 裁决,遵守 `priority` 通道优先级。 | **个体校准的主要落点之一。** |
| `escalation_ladder` | 情绪升级阶梯(犬 level 1–7)。用于判断帧间变化方向(升级/缓解/横向)和识别"警告信号"(level≥5 的信号进护栏保护名单)。 | 可调,但改动会连带影响方向判定和护栏。 |
| `output_constraints` | 逐条注入 system prompt 的输出约束。 | 可调。 |

**个体校准写在哪里(明确回答):** 当前 yaml **没有**独立的"个体档案 / individual profile"段落。做个体校准时,把发现的系统性偏差写进上表标了"个体校准落点"的位置——优先 `context_modifiers`(条件化改写)和 `conflict_rules`(冲突裁决),其次直接调对应通道的 `value_confidence`。如果要做成"每只宠物一份档案"的结构,需要新增段落 + 在 `prompts.py::build_system_prompt` / `omni_prompt` 里注入(目前没有这个注入点,`profiles` 表也还没接进链路——见第六节待办)。

---

## 附:快速自测命令

```bash
# 不开服务跑一次真实判读(打印原始输出/token/耗时/脱敏 payload)
.venv/bin/python -m pipeline.providers --video pet.mp4 --mode native --species dog --context "客厅,快递员按门铃"

# 接新端点先探测 base64 支持
.venv/bin/python -m pipeline.providers --probe --video pet.mp4

# 视频预处理自测(抽帧存 data/frames/ 供肉眼检查)
.venv/bin/python -m pipeline.video pet.mp4

# mock 模式(不发真实请求,验证链路自洽):把 config.yaml 的 mock 改 true,或
.venv/bin/python -m pipeline.providers --video pet.mp4 --mock

# 启动服务
.venv/bin/python app.py     # 默认 0.0.0.0:7860
```
