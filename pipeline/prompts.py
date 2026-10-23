"""三段链路的 prompt 模板与知识库注入。

设计约定
--------
* 知识库按 species 加载并按模态裁剪,``confidence`` / ``value_confidence``
  标注原样保留,让模型知道哪些通道可以单独作为依据、哪些只能作佐证。
* ``output_constraints`` 拼进 system prompt。
* 视觉段与音频段只做"观察编码",禁止情绪结论;情绪候选只在融合段产生。
* 所有 prompt 明确要求裸 JSON 输出;解析端(providers.parse_json_response)
  仍然兼容模型包了 markdown 代码块的情况。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KNOWLEDGE_DIR = os.path.join(PROJECT_ROOT, "knowledge")

CONFIDENCE_FLOOR = 0.5

# 所有 prompt 共用的输出格式要求
JSON_ONLY_RULE = (
    "只输出一个 JSON 对象。不要用 markdown 代码块包裹,不要输出 ```json,"
    "不要在 JSON 前后添加任何说明、标题或解释文字。JSON 内部不要写注释。"
)

# 置信度分级的说明,随知识库一起注入
CONFIDENCE_LEGEND = (
    "知识库中的 confidence / value_confidence 表示该通道或取值的可靠程度:\n"
    "  高 = 可单独作为判读依据;\n"
    "  中 = 需与至少一个其他通道交叉验证;\n"
    "  低 = 不得单独支撑结论,只能作为佐证。\n"
    "引用信号时必须连同其置信度等级一起给出,不得把低置信度信号当作高置信度使用。"
)

_KNOWLEDGE_CACHE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Prompt 容器
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Prompt:
    """system + user 两段文本,附带供 provider 使用的元信息。

    ``len(prompt)`` 返回两段合计字符数,便于自测时打印 prompt 体积。
    """

    system: str
    user: str
    species: str = ""
    task: str = ""          # vision / audio / fusion
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.system) + len(self.user)

    def messages(self) -> list[dict]:
        """OpenAI 兼容格式的消息数组。"""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def as_text(self) -> str:
        """拼成单段文本,给不支持 system role 的接口使用。"""
        return f"{self.system}\n\n{self.user}"


# ---------------------------------------------------------------------------
# 知识库加载
# ---------------------------------------------------------------------------

def load_knowledge(species: str, knowledge_dir: str | None = None) -> dict:
    """按 species 加载 ``knowledge/<species>.yaml``,结果带进程内缓存。"""
    directory = knowledge_dir or DEFAULT_KNOWLEDGE_DIR
    key = f"{directory}::{species}"
    if key in _KNOWLEDGE_CACHE:
        return _KNOWLEDGE_CACHE[key]

    path = os.path.join(directory, f"{species}.yaml")
    if not os.path.isfile(path):
        available = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(directory)
            if f.endswith(".yaml")
        ) if os.path.isdir(directory) else []
        raise FileNotFoundError(
            f"找不到 {species} 的知识库: {path};当前可用: {available or '无'}"
        )

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    _KNOWLEDGE_CACHE[key] = data
    return data


def _dump_yaml(node: Any) -> str:
    return yaml.safe_dump(node, allow_unicode=True, sort_keys=False, width=100).rstrip()


def _dump_json(node: Any) -> str:
    return json.dumps(node, ensure_ascii=False, indent=2)


def _output_constraints(knowledge: dict) -> str:
    items = knowledge.get("output_constraints") or []
    if not items:
        return ""
    lines = "\n".join(f"  - {c}" for c in items)
    return f"【输出约束】(来自知识库 output_constraints,必须逐条遵守)\n{lines}"


def _hard_constraints(knowledge: dict) -> str:
    items = (knowledge.get("meta") or {}).get("hard_constraints") or []
    if not items:
        return ""
    lines = "\n".join(f"  - {c}" for c in items)
    return f"【硬约束】(来自知识库 meta.hard_constraints)\n{lines}"


# ---------------------------------------------------------------------------
# 按模态裁剪知识库
#
# 整个知识库 dump 进 system prompt 每物种约 1.5 万字符,加上 pain_indicators /
# temporal_signals 后继续膨胀,直接推高延迟与成本。用不上的分块按模态剔掉:
# 单图没有音轨,纯音频没有画面,非视频读不到时序信号。
# ---------------------------------------------------------------------------

# 任何模态都不得剔除:判读框架、输出约束、裁决规则与素材类型无关,
# behavior_patterns 更是文字/图片模式能出候选的关键入口。
_ALWAYS_KEEP = ("meta", "output_constraints", "conflict_rules",
                "context_modifiers", "behavior_patterns")

# 各模态要剔除的顶层键。
_PRUNE_BY_MODALITY: dict[str, tuple[str, ...]] = {
    "image": ("acoustic", "temporal_signals"),      # 单图无音轨、无时间轴
    "audio": ("signals", "temporal_signals"),       # 纯音频无画面
    "text": ("temporal_signals",),                  # 文字无时间轴
    "video": (),                                    # 视频全保留
}

# 纯音频模式下 pain_indicators 里仍然保留的键:全部动作单元与面部线索都依赖画面,
# 但"这是提示不是诊断"的边界约束与"犬类无已验证量表"的说明必须留着。
_PAIN_KEEP_FOR_AUDIO = ("scale_name", "evidence_level", "hard_boundary",
                        "no_validated_scale", "what_this_measures")


def prune_knowledge(knowledge: dict, modality: str = "video") -> dict:
    """按模态裁掉用不上的知识库分块,返回**新** dict(不修改传入对象)。

    ``load_knowledge`` 的结果是进程内缓存的共享对象,这里绝不能就地改。
    未知 modality 一律按 video 处理(全量保留),宁可长也不要漏掉判读依据。
    """
    drop = set(_PRUNE_BY_MODALITY.get(modality, ()))
    pruned = {k: v for k, v in knowledge.items()
              if k in _ALWAYS_KEEP or k not in drop}

    pain = pruned.get("pain_indicators")
    if modality == "audio" and isinstance(pain, dict):
        pruned["pain_indicators"] = {k: v for k, v in pain.items()
                                     if k in _PAIN_KEEP_FOR_AUDIO}
    return pruned


def build_system_prompt(species: str, knowledge: dict, task: str,
                        modality: str = "video") -> str:
    """三段共用的 system prompt:角色 + 知识库(按模态裁剪)+ 约束 + 输出格式。

    ``modality`` 默认 "video"(全量保留),保证 vision/audio/fusion 三个旧入口
    的行为不变;只有 omni_prompt 会按实际素材类型传值。
    """
    meta = knowledge.get("meta") or {}
    framework = meta.get("framework", "")
    role = {
        "vision": "你是动物行为学的视觉信号编码员。你的工作是把画面中可观察的形态编码到给定的编码表中,不是解读情绪。",
        "audio": "你是动物行为学的声学信号编码员。你的工作是描述可测量的声学特征并匹配到给定的模式表中,不是解读情绪。",
        "fusion": "你是动物行为学的判读整合员。你在已有的视觉与声学编码结果之上做时序对齐与规则裁决,你不能引入编码结果里不存在的观察。",
    }.get(task, "你是动物行为信号的编码员。")

    parts = [
        role,
        f"物种: {species}    判读框架: {framework}",
        "",
        "【知识库】以下 YAML 是本次判读的唯一依据。不得引入知识库之外的行为学主张,"
        "不得编造文献、作者或年份。带 source 字段的条目,其 DOI 只能原样引用,"
        "不得改写、不得为没有 source 的条目补一个出处。",
        "```yaml",
        _dump_yaml(prune_knowledge(knowledge, modality)),
        "```",
        "",
        CONFIDENCE_LEGEND,
    ]

    hard = _hard_constraints(knowledge)
    if hard:
        parts += ["", hard]

    constraints = _output_constraints(knowledge)
    if constraints:
        parts += ["", constraints]

    parts += ["", f"【输出格式】{JSON_ONLY_RULE}"]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 1. 视觉段
# ---------------------------------------------------------------------------

VISION_SCHEMA = {
    "per_frame": [
        {
            "t": "该帧时间戳(秒,浮点),必须与输入的帧列表一一对应",
            "role": "该帧的角色,原样抄输入值",
            "signals": {
                "<通道名>": "取值必须来自知识库该通道的 values 列表;有 dimensions 的通道写成嵌套对象;不可观测写 \"不可观测\"",
            },
            "signal_confidence": {
                "<通道名>": "该取值在知识库中的 value_confidence,取 高/中/低;不可观测写 \"不可观测\"",
            },
            "uncertain": ["列出本帧因分辨率、遮挡、角度、品种形态学而无法可靠判读的通道名"],
        }
    ],
    "temporal_change": [
        {
            "from_t": "变化起点帧的时间戳",
            "to_t": "变化终点帧的时间戳",
            "channel": "发生变化的通道名",
            "from": "变化前取值",
            "to": "变化后取值",
            "direction": "升级 | 缓解 | 横向变化 | 无变化",
            "basis": "依据 escalation_ladder 的哪一级判断的方向,写出级别数字与依据",
        }
    ],
    "overall_signals": {
        "<通道名>": {
            "dominant": "整段中占比最高的取值",
            "confidence": "高 | 中 | 低",
            "frames_observed": "该通道可观测的帧数(整数)",
        }
    },
}


def vision_prompt(
    species: str,
    context: str,
    frames_meta: list[dict],
    motion_features: dict,
    knowledge: dict,
) -> Prompt:
    """逐帧信号编码 + 帧间变化对比。

    ``frames_meta`` 是 ``video.select_frames`` 的输出去掉 ``image_b64`` 后的部分,
    用于让模型知道每张图对应的时间戳与角色(图像本身由 provider 另行附加)。
    """
    system = build_system_prompt(species, knowledge, "vision")

    frames_desc = "\n".join(
        f"  第 {i + 1} 张  t={f.get('t')}s  role={f.get('role')}"
        for i, f in enumerate(frames_meta)
    ) or "  (无帧)"

    channels = list((knowledge.get("signals") or {}).keys())

    user = f"""【任务】对以下 {len(frames_meta)} 张按时间顺序排列的帧做信号编码。

【拍摄情境】(由用户提供,可能不准确,与画面冲突时以画面为准)
{context or "（未提供）"}

【帧清单】图像按此顺序附在本条消息中,第 N 张图对应第 N 行:
{frames_desc}

【运动学指标】由帧差曲线计算得出,可作为判断动作幅度与节律的旁证:
{_dump_json(motion_features)}

【必须完成的三件事】

1. 逐帧填空。对每一帧,遍历知识库 signals 下的全部通道并逐个填写:
   {channels}
   - 取值必须严格来自该通道的 values 列表,不得自创描述。
   - 带 dimensions 的通道(如 tail)要按其子维度分别填写。
   - 该通道在本帧无法可靠判读时,取值写 "不可观测" 并把通道名放进 uncertain。
   - 因品种形态学而结构性不可用的通道(见各通道 caveats),同样写 "不可观测"。
   - 每个取值同时给出它在知识库里的 value_confidence 等级。

2. 对比帧间变化。按时间顺序两两比较相邻帧,列出发生改变的通道。
   对每一处变化,依据 escalation_ladder 判断方向是"升级"还是"缓解":
   - 变化后的取值出现在更高 level 的 signals 中 -> 升级
   - 出现在更低 level 中 -> 缓解
   - 同级内的取值替换 -> 横向变化
   必须写出你依据的是哪一级。取值未出现在阶梯中的通道,方向写"横向变化"。

3. 汇总整段。给出每个通道在整段中占比最高的取值及其置信度等级。

【禁止事项】
- 禁止输出任何情绪词(开心、生气、害怕、焦虑、放松、享受等),包括在 basis 字段里。
  本段只做形态编码,情绪判读由后续环节负责。
- 禁止用"看起来像""似乎在"这类推测句式描述形态。
- 禁止为了填满字段而猜测:看不清就是 "不可观测",这不是失败。
- 禁止跨帧脑补:每一帧只编码该帧画面中实际可见的内容。

【输出 JSON 结构】(字段说明仅为解释用途,实际输出填真实值)
{_dump_json(VISION_SCHEMA)}

{JSON_ONLY_RULE}"""

    return Prompt(
        system=system,
        user=user,
        species=species,
        task="vision",
        meta={"frame_count": len(frames_meta), "channels": channels},
    )


# ---------------------------------------------------------------------------
# 2. 音频段
# ---------------------------------------------------------------------------

AUDIO_SCHEMA = {
    "events": [
        {
            "t_start": "事件起点(秒,浮点,相对音频文件起点)",
            "t_end": "事件终点(秒,浮点)",
            "acoustic": {
                "duration_s": "事件时长(秒)",
                "f0_contour": "基频走向:上升 | 下降 | 先升后降 | 先降后升 | 平稳 | 起伏 | 无明确基频",
                "f0_range_hz": "[估计下界, 估计上界],无法估计的一侧写 null",
                "syllable_count": "音节数(整数);连续无分段写 1",
                "tremor": "有无颤抖/震颤成分,true 或 false",
                "energy_envelope": "能量包络:陡起陡落 | 陡起缓落 | 缓起缓落 | 持续平稳 | 脉冲串",
                "noise_component": "是否包含明显宽带噪声成分,true 或 false",
            },
            "matched_pattern": "知识库 acoustic 表中的 pattern 名称;都不匹配写 \"未匹配\"",
            "match_confidence": "该匹配的可靠程度:高 | 中 | 低,参考所匹配 pattern 自带的 confidence",
            "match_basis": "说明是哪几项声学特征把它匹配到该 pattern 的",
            "alternatives": ["声学上无法排除的其他 pattern 名称,没有则空数组"],
        }
    ],
    "background_noise": "对背景噪声的一句话描述,影响判读时须说明",
    "audible_events_found": "是否检测到任何可判读的发声事件,true 或 false",
}


def audio_prompt(species: str, context: str, knowledge: dict) -> Prompt:
    """声学特征描述 + 模式匹配 + 事件时间戳。"""
    system = build_system_prompt(species, knowledge, "audio")

    patterns = [p.get("pattern") for p in (knowledge.get("acoustic") or [])]

    user = f"""【任务】对附带的音频做声学事件切分与模式匹配。音频为 16kHz 单声道。

【拍摄情境】(可能不准确,与音频冲突时以音频为准)
{context or "（未提供）"}

【必须完成的三件事】

1. 切分并描述。找出音频中每一个可判读的发声事件,对每个事件描述:
   时长、基频走向、估计基频范围、音节数、有无颤抖成分、能量包络、有无宽带噪声成分。
   基频无法估计时明确写 null,不要给编造的数值。

2. 匹配模式。把每个事件匹配到知识库 acoustic 表中的一项:
   {patterns}
   - 匹配依据必须是声学特征本身,不是情境推测。
   - 声学上无法区分的多个 pattern,主选一个,其余放进 alternatives。
   - 都不匹配就写 "未匹配",这比强行归类更有价值。
   - match_confidence 不得高于所匹配 pattern 在知识库中自带的 confidence。

3. 标时间戳。每个事件给出 t_start / t_end,精确到 0.1 秒。

【注意事项】
- 知识库 acoustic 中的 f0_range 是形态学粗分档,不是精确阈值,不要因为实测值
  落在边界外几十赫兹就拒绝匹配;但偏离一个数量级时应写 "未匹配"。
- 低频事件(如 purr 的 25-30Hz)可能低于录音设备与视频压缩的有效响应下限。
  音频中未检测到不等于未发生,此时应在 background_noise 里说明,而不是断言其不存在。
- 环境噪声(人声、电视、交通、风噪)不是事件,不要编码;但要在 background_noise 中说明
  它是否掩盖了判读。
- 音频中没有任何目标物种的发声时,events 输出空数组,audible_events_found 写 false。
  这是合法结果,不要为了凑数而把噪声当成发声。

【禁止事项】
- 禁止输出情绪结论。本段只做声学描述与模式匹配。
- 禁止根据情境反推声学特征(例如因为"在打针"就把某段声音标成疼痛尖叫)。

【输出 JSON 结构】(字段说明仅为解释用途,实际输出填真实值)
{_dump_json(AUDIO_SCHEMA)}

{JSON_ONLY_RULE}"""

    return Prompt(
        system=system,
        user=user,
        species=species,
        task="audio",
        meta={"pattern_count": len(patterns)},
    )


# ---------------------------------------------------------------------------
# 3. 融合段
# ---------------------------------------------------------------------------

FUSION_SCHEMA = {
    "timeline": [
        {
            "t_range": "[起点秒, 终点秒]",
            "visual": "该时间段内的视觉信号摘要,只引用视觉编码结果中已有的内容",
            "audio": "该时间段内的声学事件摘要;无事件写 null",
            "motion": "该时间段的运动强度描述,引用运动学指标",
        }
    ],
    "conflicts": [
        {
            "signals": ["冲突的信号名"],
            "rule": "命中的 conflict_rules 或 context_modifiers 条目原文",
            "resolution": "按该规则裁决的结果",
            "resolved": "是否裁决成功,true 或 false;规则未覆盖时写 false",
        }
    ],
    "candidates": [
        {
            "label": "状态候选标签",
            "confidence": "0 到 1 的浮点数,严格小于 0.5 的候选不得出现在输出中",
            "evidence": [
                {
                    "channel": "通道名或声学 pattern 名",
                    "observation": "具体观察到的取值",
                    "t": "该观察出现的时间戳或时间段",
                    "knowledge_basis": "知识库中支持该推断的条目原文",
                    "signal_confidence": "该信号的置信度等级 高/中/低",
                }
            ],
        }
    ],
    "insufficient_evidence": "true 或 false",
    "insufficient_reason": "insufficient_evidence 为 true 时必填,说明缺什么;否则写 null",
    "translation": "一句话的拟人化表达",
    "suggestion": "主人现在可以做什么,一到三句",
}


def fusion_prompt(
    visual_result: dict,
    audio_result: dict,
    motion_features: dict,
    context: str,
    knowledge: dict,
    species: str = "",
) -> Prompt:
    """时序对齐 + 规则裁决 + 带依据的情绪候选。"""
    species = species or (knowledge.get("meta") or {}).get("species") or ""
    system = build_system_prompt(species, knowledge, "fusion")

    ladder = knowledge.get("escalation_ladder") or []
    warning_levels = [lv for lv in ladder if int(lv.get("level", 0)) >= 5]
    warning_signals: list[str] = []
    for lv in warning_levels:
        warning_signals.extend(lv.get("signals") or [])

    user = f"""【任务】把视觉与声学编码结果整合成一次完整判读。

【拍摄情境】
{context or "（未提供）"}

【视觉编码结果】
{_dump_json(visual_result)}

【声学编码结果】
{_dump_json(audio_result)}

【运动学指标】
{_dump_json(motion_features)}

【必须完成的五件事】

1. 时序对齐。把视觉帧与声学事件按时间轴合并成 timeline。
   - 时间段的切分以声学事件边界和视觉信号发生变化的时刻为准,不要均匀切分。
   - timeline 中只能出现上面两份编码结果里已有的观察。禁止引入新观察。
   - 某段没有音频事件时 audio 写 null,不要用"可能有"填充。

2. 应用规则。逐条检查知识库的 context_modifiers 与 conflict_rules:
   - context_modifiers:命中 override_when 条件时,必须按 override_to 解读,
     并把这次改写记进 conflicts,rule 字段写出该条目原文。
   - conflict_rules:出现 conflicting 中列出的信号组合时,按 resolution 裁决,
     并遵守其 priority 指定的通道优先级。
   - 规则未覆盖的冲突也要记进 conflicts,resolved 写 false。

3. 给出候选。每个候选必须带 confidence 与 evidence:
   - confidence 严格小于 0.5 的候选直接不要输出,不要输出后再标注低置信度。
   - 每条 evidence 必须包含:通道名、具体观察值、时间戳、知识库依据原文、该信号的置信度等级。
   - 低置信度(低)的信号不得单独支撑一个候选,至少要有一条中或高置信度证据。
   - 候选数量不设下限,可以为空数组。

4. 判定证据是否充分。下列任一条成立时,insufficient_evidence 必须为 true,
   candidates 必须为空数组,并在 insufficient_reason 中说明:
   - 可观测通道少于 3 个;
   - 知识库 meta.reading_order 中作为判读主轴的通道不可观测;
   - conflicts 中存在 resolved=false 且该冲突涉及判读主轴;
   - 所有候选的 confidence 都低于 0.5。
   证据不足是合法且有价值的结论。禁止为了给出答案而降低标准、拼凑证据或提高置信度。

5. 写 translation 与 suggestion。

【translation 的规则】
- 这是本次输出中唯一允许拟人化的字段,一句话,用第一人称。
- 内容必须与置信度最高的候选一致,不得引入候选中没有的状态。
- insufficient_evidence 为 true 时,translation 必须表达"这段素材看不出来",
  禁止给出任何具体情绪。

【suggestion 的规则】(违反任意一条即为错误输出)
- 禁止建议惩罚、呵斥、压制或以任何方式阻止警告信号。本物种的警告信号包括:
  {warning_signals}
  知识库明确指出:惩罚这些信号会使个体在后续情境中跳过警告直接升级。
- 当判读结果落在 escalation_ladder 的任一级上时,首选建议永远是移除压力源、
  恢复个体的撤离路径、停止当前互动,而不是让个体"适应"或"克服"。
- 禁止给出医疗诊断或用药建议。出现疼痛、持续性行为异常的迹象时,
  只描述观察到的形态,并建议由执业兽医或行为专业人员评估。
- insufficient_evidence 为 true 时,suggestion 应给出如何补拍更可判读的素材
  (角度、光照、时长、是否需要同时录音),而不是行为建议。

【输出 JSON 结构】(字段说明仅为解释用途,实际输出填真实值)
{_dump_json(FUSION_SCHEMA)}

{JSON_ONLY_RULE}"""

    return Prompt(
        system=system,
        user=user,
        species=species,
        task="fusion",
        meta={"warning_signals": warning_signals},
    )


# ---------------------------------------------------------------------------
# 4. 全模态单次调用(MiMo-V2.6-Flash 等原生多模态模型)
# ---------------------------------------------------------------------------

OMNI_SCHEMA = {
    "visual_signals": [
        {
            "t": "该观察对应的时间戳(秒,浮点)",
            "signals": {
                "<通道名>": "取值必须来自知识库该通道的 values 列表;有 dimensions 的通道写成嵌套对象;不可观测写 \"不可观测\"",
            },
            "signal_confidence": {"<通道名>": "高 | 中 | 低 | 不可观测"},
            "uncertain": ["因分辨率、遮挡、角度、品种形态学而无法可靠判读的通道名"],
        }
    ],
    "audio_events": [
        {
            "t_start": "事件起点(秒)",
            "t_end": "事件终点(秒)",
            "acoustic": {
                "duration_s": "时长(秒)",
                "f0_contour": "上升 | 下降 | 先升后降 | 先降后升 | 平稳 | 起伏 | 无明确基频",
                "f0_range_hz": "[估计下界, 估计上界],无法估计的一侧写 null",
                "syllable_count": "音节数(整数)",
                "tremor": "true 或 false",
                "energy_envelope": "陡起陡落 | 陡起缓落 | 缓起缓落 | 持续平稳 | 脉冲串",
                "noise_component": "true 或 false",
            },
            "matched_pattern": "知识库 acoustic 表中的 pattern 名称;都不匹配写 \"未匹配\"",
            "match_confidence": "高 | 中 | 低,不得高于该 pattern 自带的 confidence",
            "match_basis": "把它匹配到该 pattern 的声学依据",
        }
    ],
    "timeline": [
        {
            "t_range": "[起点秒, 终点秒]",
            "visual": "该时段的视觉信号摘要",
            "audio": "该时段的声学事件摘要;无事件写 null",
            "motion": "该时段的运动强度描述",
            "direction": "相对上一时段的方向:升级 | 缓解 | 横向变化 | 无变化,并注明依据的 escalation_ladder 级别",
        }
    ],
    "conflicts": [
        {
            "signals": ["冲突的信号名"],
            "rule": "命中的 conflict_rules 或 context_modifiers 条目原文",
            "resolution": "按该规则裁决的结果",
            "resolved": "true 或 false;规则未覆盖时写 false",
        }
    ],
    "candidates": [
        {
            "label": "状态候选标签",
            "confidence": "0 到 1 的浮点数,严格小于 0.5 的候选不得出现",
            "evidence": [
                {
                    "channel": "通道名或声学 pattern 名",
                    "observation": "具体观察到的取值",
                    "t": "时间戳或时间段",
                    "knowledge_basis": "知识库中支持该推断的条目原文",
                    "signal_confidence": "高 | 中 | 低",
                }
            ],
        }
    ],
    "segment_consistency": {
        "segments": [
            {
                "t_range": "[起点秒, 终点秒]",
                "reading": "该时间段单独看得出的状态",
            }
        ],
        "agreement": "consistent | mixed | contradictory",
        "note": "各段是否一致的一句话说明;单图/纯音频只有一段时写明无法评估稳定性",
        "confidence_effect": "一致→上调 / 矛盾→下调 / 无法评估→维持,并说明对最终置信度的影响",
    },
    "counter_evidence": {
        "counter_hypotheses": [
            {
                "alternative": "与主判断相反或不同的假设",
                "would_look_like": "若该假设成立,画面/声音里本应出现什么信号",
                "present_in_media": "上述反方信号在本次素材里是否真的出现:是 | 否 | 无法确认",
            }
        ],
        "verdict": "主判断在排除以上反方后是否仍然成立:成立 | 减弱 | 存疑",
        "note": "一句话总结反方论证的结论",
    },
    "missing_signals": [
        {
            "signal": "描述/素材里缺失但对判读重要的通道名(如 tail、ear_position)",
            "why_it_helps": "补充这个信号为什么能提高判断可靠性",
        }
    ],
    "insufficient_evidence": "true 或 false",
    "insufficient_reason": "为 true 时必填,说明缺什么;否则写 null",
    "translation": "一句话的拟人化表达",
    "suggestion": "主人现在可以做什么,一到三句",
}


def mimo_system_preamble(now: datetime | None = None) -> str:
    """MiMo 官方建议的系统提示词开头,日期动态生成。

    只作为前缀,后面接项目自己的知识库与输出约束,不替换它们。
    """
    now = now or datetime.now()
    week = "星期" + "一二三四五六日"[now.weekday()]
    date = f"{now.year}年{now.month}月{now.day}日"
    return (
        "你是MiMo(中文名称也是MiMo),是小米公司研发的AI智能助手。\n"
        f"今天的日期:{date} {week},你的知识截止日期是2024年12月。"
    )


# 各模态下"结构性不可观测"的通道:不是没看清,而是这种素材里根本不存在该信息。
# 判定证据是否充分时,这些通道必须从分母里剔除,否则单图/纯音频会被必然判为证据不足。
_STRUCTURALLY_ABSENT = {
    "image": {
        "channels": ["displacement_signals", "affiliative_signals",
                     "tail.motion", "ear.motion"],
        "note": "位移信号与亲和信号都是动作序列,单帧无法判定;尾部/耳部的 motion 子维度同理。",
    },
    "audio": {
        "channels": ["__all_visual__"],
        "note": "纯音频没有任何画面,全部视觉通道都不适用。",
    },
    "text": {"channels": [], "note": ""},
    "video": {"channels": [], "note": ""},
}

# ---- 单图模式的通道分类 ----------------------------------------------------
# 主轴通道:这两个中至少一个可读,单帧就具备判读基础。
_IMAGE_AXIS_CHANNELS = ["body_posture", "ear_position"]

# 永不进分母:单帧里天然读不到,或极易因遮挡/照度/毛量而读不到。
# 读不到不算"证据不足",读到了则计入可观测数(加分项,不是及格项)。
_IMAGE_NEVER_COUNTED = [
    "displacement_signals",   # 位移信号:动作序列,单帧不成立
    "affiliative_signals",    # 亲和信号:同上
    "tail 的 motion 子维度",  # 摆动是时间量
    "pupil",                  # 瞳孔:照度未知时不可用(知识库已标 confidence=低)
    "whisker_position",       # 胡须:长毛/角度/分辨率极易不可辨
    "eye 的 pupil 子维度",    # 犬类 eye.pupil 同理
]


def _pain_block(modality: str, knowledge: dict) -> str:
    """疼痛面部指标在各模态下的读法。按物种分叉:猫有已验证量表(FGS),犬没有。

    两边的输出边界一致 —— 只提示、不诊断。差别在于犬侧连"评分"这一步都不允许。
    """
    pain = knowledge.get("pain_indicators") or {}
    if not pain:
        return ""

    # 猫:FGS 有 action_units,且经效度验证;犬:没有等效量表,只有恐惧研究的面部线索。
    has_scale = bool(pain.get("action_units"))

    if not has_scale:
        return """**疼痛/不适线索(本物种【没有】已验证的面部疼痛量表):**
知识库 pain_indicators 收录的面部线索(耳根后拉、张口增多)来自**恐惧研究而非疼痛研究**,
且本物种不存在等效于 Feline Grimace Scale 的已验证量表。因此:
- 这些线索**只能**按"恐惧 / 负性唤醒"解读,与 signals 的防御性判读相互印证;
- **禁止**据此评分、设阈值,或套用其他物种的 cut-off;
- **禁止**输出任何疼痛结论。仅当出现躯体性迹象(运动受限、定点舔舐、姿势异常、触碰躲避)
  且已排除体温调节与品种结构因素时,才按 pain_indicators.hard_boundary 的措辞提示
  『建议由执业兽医评估』,并且只描述观察到的形态,不做归因。"""

    if modality == "text":
        return """**疼痛面部指标(FGS)在文字模式下的读法:**
用户若描述了面部特征(耳朵扁平外旋、眯眼、胡须前伸变直、低头/下颏收向胸部),
可以把它们映射到 pain_indicators 的对应动作单元。但:
- confidence **上限仍是 0.65,不得因命中 FGS 而上调** —— 静态图片与实时观察的等效性
  (source 10.7717/peerj.8967)只适用于**直接看到画面**的情形,不适用于主人转述;
- evidence 里**必须写明"依据主人转述,非直接观测"**;
- **只有命中 ≥3 个动作单元**时,才允许提示"可能存在不适,建议由执业兽医评估";
  命中 1–2 个时只作佐证,不得提示疼痛;
- ⚠️ 输出边界:只能写『观察到符合疼痛的面部特征,建议由执业兽医评估』,
  **禁止**断言疼痛(如"你的猫在痛")、推测病因或给出用药建议。"""

    if modality == "image":
        return """**(d) 疼痛面部指标(FGS)—— 单图模式下【不因静态素材而降级】:**
pain_indicators 是本知识库中少见的**高证据等级**通道。画面能读到其中多个动作单元
(耳位、眼眶紧缩、口鼻紧张度、胡须变化、头部位置)时即可据此出候选:
- 依据 pain_indicators.static_image_validity:静态图片评分与实时观察几乎等效
  (偏差 -0.057,一致性区间 -0.351 至 0.237),**因此 FGS 相关判读不因"只有一张照片"
  而降级** —— 这是单图模式中少数不受静态局限影响的通道;
- 可观测的动作单元计入上面 (b) 的可观测通道总数;
- 达到 pain_indicators.threshold 时应给出就医提示;
- ⚠️ 输出边界:只能写『观察到符合疼痛的面部特征,建议由执业兽医评估』,
  **禁止**断言疼痛、推测病因或给用药建议;
- ⚠️ **不要与 signals 的同名通道混用**(见 conflict_rules 中 FGS 那一条):
  同一处观察不得在两套体系里各记一次当作两条独立证据。"""

    return """**疼痛面部指标(FGS):**
可读到多个动作单元时按 pain_indicators 评估,达到 threshold 应给出就医提示。
⚠️ 只能写『观察到符合疼痛的面部特征,建议由执业兽医评估』,禁止断言疼痛、推测病因
或给用药建议;不要与 signals 的同名通道混用(见 conflict_rules)。"""


def _pain_read_instruction(modality: str, knowledge: dict) -> str:
    """第一步里"顺带把疼痛面部指标读了"的指令。只对有画面的模态生成。"""
    pain = knowledge.get("pain_indicators") or {}
    if not pain or modality not in ("image", "video"):
        return ""

    if pain.get("action_units"):
        units = [u.get("unit") for u in pain["action_units"]]
        return f"""
- **同时读疼痛面部指标(pain_indicators):** 逐个评估 {units} 这 5 个动作单元,
  各记 0 / 1 / 2 分,读不到的记"不可观测"(**不要猜分**)。这是知识库中少见的
  **高证据等级**通道,且静态图片评分与实时观察几乎等效,单图同样适用。
  ⚠️ 与 signals 的同名通道(ear_position / whisker_position)读的**不是同一件事**:
  那边读唤醒与防御意图,这边读疼痛。分别编码,**不得**互相替代填值,也不得把同一处
  观察在两边各记一次当作两条独立证据(裁决见 conflict_rules)。
  ⚠️ 结论边界:达到阈值时只能写『观察到符合疼痛的面部特征,建议由执业兽医评估』,
  **禁止**断言疼痛、推测病因或给用药建议。"""

    return """
- **同时留意疼痛/不适线索(pain_indicators):** 本物种**没有**已验证的面部疼痛量表,
  该节收录的线索(耳根后拉、张口增多)来自**恐惧研究而非疼痛研究**。
  因此只按"恐惧 / 负性唤醒"记录,**禁止**评分、设阈值或输出疼痛结论;
  出现躯体性迹象且已排除体温调节与品种结构因素时,才按 hard_boundary 的措辞提示就医。"""


def _temporal_read_instruction(modality: str, knowledge: dict) -> str:
    """第一步里读 temporal_signals 的指令。只有 video 有连续画面,其余模态一律不生成。"""
    temporal = knowledge.get("temporal_signals") or {}
    if not temporal or modality != "video":
        return ""
    names = [c.get("channel") for c in (temporal.get("channels") or [])]
    return f"""
- **同时读时序信号(temporal_signals):** {names}。
  这一层**只有视频读得到** —— 它读的是**随时间的变化方向**(升级 / 缓解),
  不是某一帧的取值。逐条评估,读不到的写 "不可判定"(不影响充分性判定)。
  ⚠️ 带 limitation 字段的通道必须在 evidence 里如实写明该限制,
  confidence 不得超过该通道自带的 confidence / confidence_cap。"""


def _sufficiency_rules(modality: str, knowledge: dict) -> str:
    """按模态生成"证据是否充分"的判定规则。

    视频有完整的时间+声音维度,门槛最高;单图与纯音频天然少若干通道,
    沿用同一套门槛会导致"只要不是视频就必然证据不足",那是规则的缺陷而非事实。
    """
    if modality == "text":
        return """【文字描述的充分性判据 —— 只基于描述里的客观观察,不套用媒体门槛】

**允许出候选的条件:** 用户描述里包含**至少一项主轴级客观观察** ——
体态/身体姿势、耳朵位置,或一个明确的可复核行为动作(如"往后退""扑上来""贴着人蹭")。
满足即正常出候选(低置信度)。

**命中 behavior_patterns 同样满足出候选条件(重要):**
用户描述的往往是**复合行为**("抱着我的腿咬""追自己尾巴""踩奶"),而 signals 编码的是
**静态原子信号**。描述命中知识库 behavior_patterns 里某个 pattern 的 typical_phrases 时:
- **按该 pattern 的 candidates 并列给出候选**(不要硬选一个),这本身即构成可判读的客观行为;
- confidence **不得超过该 pattern 的 confidence_cap**,且同时不得超过文字模态上限 0.65,
  两者取更小值;
- 把该 pattern 的 distinguishing_signals 指向的通道写进 **missing_signals**,
  说明补充它们才能在几个候选之间区分;
- evidence 里 knowledge_basis 引用该 pattern 条目(pattern 名与 notes),
  channel 可写 "behavior_pattern"。
- 此时**不得**判 insufficient_evidence —— 用户已经描述了一个可映射的客观行为,
  只是缺少用于区分候选的原子信号,那是**降低置信度并写进 missing_signals** 的理由,
  不是拒绝判读的理由。

**insufficient_evidence 必须为 true(并引导补充)当:**
- 描述里只有用户自己的结论(如"它好像不开心""感觉怪怪的""它在闹脾气"),
  没有任何可映射到编码表的客观形态/动作,**也没有命中任何 behavior_pattern**;
  (注意:纯结论不构成行为描述 —— "它好像不太开心"没有描述任何动作,
   不得因为情绪词与某个 pattern 的候选相近就算作命中 typical_phrases。
   命中的前提是用户描述了一个**可复核的动作**。)
- 或描述空泛到读不出任何主轴信号;
- 或所有候选 confidence 都低于 0.5。
  此时 insufficient_reason 说明"描述里缺少可判读的客观观察",
  suggestion 具体引导用户补充:耳朵朝向、尾巴位置与动作、身体姿势/重心、有没有发声、
  当时在做什么、对方(人/动物)在做什么。

**置信度:** 文字模式 confidence **上限 0.65**(文字描述必然有信息损失、且依赖用户观察的
准确度),仅凭单一主轴观察时落 **0.5–0.6**。在 evidence 里如实写明这是转述而非直接观测。

**客观观察 vs 用户结论:** 只有客观观察能作为判读依据。用户的结论("它很紧张""它生气了")
记录下来供参考,但**绝不作为**候选的证据,也不得因为用户已经下了结论就顺着给高置信度。

""" + _pain_block(modality, knowledge)

    if modality == "image":
        never = "\n".join(f"    · {c}" for c in _IMAGE_NEVER_COUNTED)
        return f"""【单图的充分性判据 —— 按下面这一套执行,不要套用视频的门槛】

**允许出候选的条件(满足即出,不要再加码):**
  (a) {_IMAGE_AXIS_CHANNELS} 这两个主轴通道中**至少一个**可读;**且**
  (b) 可观测通道总数 **≥ 2**。
满足即正常输出 candidates(低置信度),**不得**判 insufficient_evidence。

例:只有"耳位 + 体态"两个可读,其余全不可观测 —— 这是**充分**的,照常出候选。
耳位与体态是行为学判读的主轴信号,它们可读时单张照片已经足以支撑一个低置信度判断。

**(c) 画面命中 behavior_patterns 时同样满足出候选条件(与 (a)(b) 并列,任一满足即可):**
画面里呈现的往往是一个**复合动作**(抱咬后腿踢、作揖邀玩、低伏摆臀),而 signals 编码的是
**静态原子信号**。画面内容命中知识库 behavior_patterns 里某个 pattern 的 typical_phrases
所描述的动作形态时:
- **按该 pattern 的 candidates 并列给出候选**(不要硬选一个);
- confidence **不得超过该 pattern 的 confidence_cap**,且同时不得超过单图上限 0.75,
  两者取更小值;
- 把该 pattern 的 distinguishing_signals 指向的通道写进 **missing_signals**,
  说明补充它们才能在几个候选之间区分;
- evidence 里 knowledge_basis 引用该 pattern 条目,channel 可写 "behavior_pattern"。
- 此时**不得**判 insufficient_evidence:画面已呈现可映射的客观行为,缺的只是用于区分候选的
  原子信号 —— 那是**降低置信度并写进 missing_signals** 的理由,不是拒绝判读的理由。
- 注意单帧局限:静态画面可能无法确认动作的完整序列(如是否真有后腿踢、是否有摆尾蓄力),
  不确定时应在 evidence 中写明,并让候选保持并列。

{_pain_block(modality, knowledge)}

**永不计入分母的通道**(读不到是素材类型使然,不是证据不足):
{never}
  这些通道写 "不可观测" 即可。**读不到不扣分;读到了则计入 (b) 的可观测总数(加分项)。**

**insufficient_evidence 只在下列情况为 true:**
- 画面中没有目标物种;
- 个体严重模糊 / 过曝 / 欠曝到读不出形态;
- 主体被完全遮挡,主轴通道(体态、耳位)一个都读不出,**且**画面也未命中任何 behavior_pattern;
- 可观测通道总数 < 2,**且**画面未命中任何 behavior_pattern;
- 所有候选的 confidence 都低于 0.5。
除此之外**不得**以"只有一张照片""缺少时间维度""通道数偏少"为由判证据不足。

**置信度校准:** 静态素材无法区分"瞬间姿态"与"持续状态",也看不到变化方向。
单图 confidence **上限 0.75**;仅靠两个主轴通道出的候选建议落在 **0.5–0.65**。
在 evidence 中如实写明该局限,但这是**降低置信度**的理由,不是**拒绝判读**的理由。
结论只描述该瞬间的状态,不要推断此前或此后会发生什么。"""

    if modality == "audio":
        return """insufficient_evidence 必须为 true(且 candidates 置空、insufficient_reason 必填)当:
- 音频中没有目标物种的发声(只有环境噪声、人声、静音);
- 所有发声事件都无法匹配到知识库 acoustic 表中的任何 pattern;
- 所有候选的 confidence 都低于 0.5。

【纯音频判定的重要校准】
- 本次没有任何画面。**全部视觉通道都属于结构性不可观测**,visual_signals 输出空数组,
  **不要**因为"看不到视觉信号"判证据不足——那是素材类型决定的,不是信息缺失。
- 判读完全建立在声学证据上。知识库中若干发声(如犬的低吼、猫的 hiss)本身
  confidence 就是"高",单独即可支撑候选。
- 但缺少视觉通道意味着无法交叉验证,也无法应用大部分 context_modifiers。
  因此纯音频的 confidence **上限建议 0.7**,并在 insufficient_reason 或 evidence 中说明该局限。
- 声学上无法区分的模式(如犬的玩耍性低吼与威胁性低吼)**必须**并列为多个候选或明确指出
  无法区分,不得只挑一个。"""

    return """insufficient_evidence 必须为 true(且 candidates 置空、insufficient_reason 必填)当:
- 可观测通道少于 3 个;
- 知识库 meta.reading_order 中作为判读主轴的通道不可观测;
- conflicts 中存在 resolved=false 且该冲突涉及判读主轴;
- 所有候选的 confidence 都低于 0.5。

【时序信号层(temporal_signals)—— 只有本模态可用】
视频是唯一能读到 temporal_signals 的素材类型,务必用上这一层:
- 逐条检查 temporal_signals.channels,读**随时间的变化方向**(升级 / 缓解),
  而不是某一帧的取值 —— 单图读不到的正是这个维度;
- 每条通道的 confidence 不得超过其自带的 confidence / confidence_cap;
- 带 limitation 字段的通道(如摆尾左右不对称)**必须**在 evidence 里如实写明该限制,
  不得把"观察者反应"的证据说成"摆尾个体内在状态"的直接测量;
- 读不到的通道写 "不可判定" 即可,**不影响**上面的充分性判定,不计入分母。

""" + _pain_block(modality, knowledge)


def _consistency_block(modality: str) -> str:
    """多帧一致性判据。视频要真的分段自检,单图/纯音频只有一段无法评估稳定性。"""
    if modality == "video":
        return """把视频按时间切成 2–4 段(与第三步的 timeline 时段可以对齐),
**对每一段单独得出一个状态读数**,填进 segment_consistency.segments。然后:
- 各段读数一致 → agreement = "consistent",这提高了判断的可靠性,
  最终候选的 confidence 可以**上调**(仍不超过该模态的上限)。
- 各段读数分歧 → agreement = "mixed" 或 "contradictory",说明状态在变化或本就不稳定,
  最终候选的 confidence 必须**下调**,并在 note 里说明是哪几段冲突。
- confidence_effect 里写清楚这次一致性检查最终把置信度往哪个方向调了。
目的:让置信度反映**跨时间的稳定性**,而不是被某一个瞬间的强烈表情带偏。"""
    if modality == "text":
        return """本次是文字描述,没有时间轴,无法做跨时段一致性自检。
segment_consistency.segments 留空或只放一段,agreement 写 "unassessable",
note 写"文字描述无时间维度,稳定性无法评估",confidence_effect 写"维持,不因一致性加分"。"""
    label = "单张照片只有一个瞬间" if modality == "image" else "纯音频不分视觉时段"
    return f"""本次是{'单图' if modality == 'image' else '纯音频'},{label},无法做跨时段的一致性自检。
segment_consistency.segments 只放一段(或留空),agreement 写 "unassessable",
note 说明"素材只有单一时间切面,稳定性无法评估",confidence_effect 写"维持,不因一致性加分"。
**不得**因为"看起来很一致"而给单一时间切面的判断加置信度。"""


def omni_prompt(
    species: str,
    context: str,
    knowledge: dict,
    motion_features: dict,
    media_hint: str = "",
    now: datetime | None = None,
    modality: str = "video",
) -> Prompt:
    """全模态单次调用:视觉编码 + 声学编码 + 时序对齐 + 规则裁决 + 候选输出。

    ``modality`` ∈ {video, image, audio},决定"证据是否充分"的判定门槛:
    单图与纯音频天然缺少某些通道,不能沿用视频的门槛。

    ``media_hint`` 用于说明本次附带的媒体形态(整段视频 / 若干抽帧 + 音轨),
    让模型知道时间戳的参照系。

    system 以 MiMo 官方建议的身份/日期提示词开头,项目自己的知识库、硬约束与
    输出约束接在其后。
    """
    system = (
        mimo_system_preamble(now)
        + "\n\n"
        + build_system_prompt(species, knowledge, "fusion", modality=modality)
    )

    channels = list((knowledge.get("signals") or {}).keys())
    patterns = [p.get("pattern") for p in (knowledge.get("acoustic") or [])]

    ladder = knowledge.get("escalation_ladder") or []
    warning_signals: list[str] = []
    for lv in ladder:
        try:
            if int(lv.get("level", 0)) >= 5:
                warning_signals.extend(lv.get("signals") or [])
        except (TypeError, ValueError):
            continue

    absent = _STRUCTURALLY_ABSENT.get(modality, _STRUCTURALLY_ABSENT["video"])
    modality_label = {"video": "视频(含音轨)", "image": "单张静态照片",
                      "audio": "纯音频(无画面)",
                      "text": "文字描述(无媒体)"}.get(modality, modality)

    if modality == "text":
        step1 = f"""════════ 第一步:从文字描述提取视觉信号(填入 visual_signals) ════════

把用户描述里的**客观观察**映射到知识库 signals 编码表:
{channels}
- 只映射描述中明确提到的观察;没提到的通道写 "不可观测",并放进 uncertain。
- 取值尽量落到该通道的 values 列表;描述太笼统对不上具体取值时写 "不可观测"。
- 所有记录 t 统一写 0(文字没有时间轴)。
- **严格区分客观观察与用户结论**:
  · 客观观察 = 可被他人复核的形态/动作,例:"耳朵向后贴""尾巴夹在腿间""压低身体往后退""对着门低吼"。
    → 映射到编码表,作为判读依据。
  · 用户结论 = 用户自己的判断,例:"它很紧张""它生气了""它不开心"。
    → 记下供参考,**绝不作为依据**,更不能因为用户已下结论就顺着给高置信度。
- **主动指出缺失(填 missing_signals):** 把描述里没覆盖、但对本物种判读重要的通道
  (尤其主轴的体态、耳位,以及尾巴、发声)列进 missing_signals,并说明补充它为何能提高可靠性。

- **除了映射到 signals 编码表,也要把描述里的复合动作匹配到知识库的 behavior_patterns。**
  用户说的常常是**行为**("抱着我的腿咬""追自己尾巴""踩奶"),而 signals 里只有**静态原子信号**,
  直接找不到落点 —— 这时**不要**因为通道对不上就全标"不可观测"。
  命中某个 pattern 的 typical_phrases 时:按其 candidates **并列**给出低置信度候选
  (confidence 取 该 pattern 的 confidence_cap 与文字模态上限 0.65 中的较小值),
  并把其 distinguishing_signals 指向的通道逐条列进 missing_signals。
  命中 pattern 即已具备判读基础,**不得**据此判证据不足(判据见第五步)。"""
    elif modality == "audio":
        step1 = """════════ 第一步:视觉信号编码(填入 visual_signals) ════════

本次是纯音频,没有任何画面。**visual_signals 直接输出空数组 `[]`**,不要编造视觉观察。
timeline 中的 visual 字段一律写 null。这不是信息缺失,是素材类型决定的。"""
    else:
        step1 = f"""════════ 第一步:视觉信号编码(填入 visual_signals) ════════

{"本次是单张静态照片,visual_signals 只输出一条记录,t 写 0。" if modality == "image" else "在时间轴上选取信号发生变化的时刻(不要均匀采样),"}对每个时刻遍历知识库 signals
下的全部通道并逐个填写:
{channels}
- 取值必须严格来自该通道的 values 列表,不得自创描述。
- 带 dimensions 的通道(如 tail)按其子维度分别填写。
- 无法可靠判读时取值写 "不可观测",并把通道名放进 uncertain。
- 因品种形态学而结构性不可用的通道(见各通道 caveats)同样写 "不可观测"。
- 每个取值同时给出它在知识库里的 value_confidence 等级。
- 每条记录必须带时间戳 t。
{f"- 本次素材中以下通道属于结构性不可观测,写 “不可观测” 即可,不必勉强判读:{absent['channels']}。{absent['note']}" if absent["channels"] else ""}
{'''- **重点读主轴:** 优先把 body_posture(整体重心与肌肉张力)和 ear_position 读准,
  这两个是判读主轴,单帧里通常都能读到。其余通道读不到就写 "不可观测",不必勉强,
  也不要因为它们读不到就认为整体证据不足 —— 充分性判据见第五步。
- **除了映射到 signals 编码表,也要把画面里的复合动作匹配到知识库的 behavior_patterns。**
  画面里常常是一个**动作**(抱咬后腿踢、作揖邀玩、低伏摆臀),而 signals 里只有**静态原子信号**,
  直接找不到落点 —— 这时**不要**因为通道对不上就全标 "不可观测"。
  命中某个 pattern 时:按其 candidates **并列**给出低置信度候选(confidence 取该 pattern 的
  confidence_cap 与单图上限 0.75 中的较小值),并把其 distinguishing_signals 指向的通道
  逐条列进 missing_signals。命中 pattern 即已具备判读基础,**不得**据此判证据不足。''' if modality == "image" else ""}
{_pain_read_instruction(modality, knowledge)}{_temporal_read_instruction(modality, knowledge)}"""

    if modality == "text":
        step2 = f"""════════ 第二步:从描述提取声音(填入 audio_events) ════════

如果描述提到叫声 / 呼噜 / 低吼 / 嚎叫等,提取出来并尽量匹配知识库 acoustic 表:
{patterns}
- 只依据描述里的声音线索,**不要编造**基频、音节数这类描述没提到的声学细节
  (没提就写 null 或 "未描述")。
- 描述没提到任何声音 → audio_events 输出空数组。"""
    elif modality == "image":
        step2 = """════════ 第二步:声学事件编码(填入 audio_events) ════════

单张照片没有音轨,**audio_events 直接输出空数组 `[]`**。不要编造发声。"""
    else:
        step2 = f"""════════ 第二步:声学事件编码(填入 audio_events) ════════

找出音轨中每一个可判读的发声事件,描述其时长、基频走向、估计基频范围、音节数、
有无颤抖成分、能量包络、有无宽带噪声成分,然后匹配到知识库 acoustic 表中的一项:
{patterns}
- 匹配依据必须是声学特征本身,不是情境推测。
- 都不匹配就写 "未匹配",这比强行归类更有价值。
- match_confidence 不得高于所匹配 pattern 在知识库中自带的 confidence。
- 每个事件必须带 t_start / t_end,精确到 0.1 秒。
- 知识库中的 f0_range 是粗分档不是精确阈值,偏离几十赫兹不影响匹配,偏离一个数量级时写 "未匹配"。
- 低频事件(如 purr 的 25-30Hz)可能低于录音与压缩的有效响应下限,未检测到不等于未发生。
- 无任何目标物种发声时 audio_events 输出空数组,这是合法结果,不要把环境噪声当成发声。"""

    if modality == "text":
        task_line = ("【任务】仅根据用户的文字描述做一次完整的行为学判读:把描述里的客观观察"
                     "映射到编码表,给出带依据的状态候选,并指出描述里缺了哪些关键信号。")
        media_block = ("【用户的文字描述】(这是本次判读的唯一输入;描述里可能混有用户自己的结论,"
                       f"按第一步的规则区分对待)\n{media_hint or '(空)'}")
        situation_label = "【背景情境】(用户额外提供的场景标签/说明,可能不准确)"
    else:
        task_line = ("【任务】对附带的媒体做一次完整判读:视觉信号编码、声学事件编码、时序对齐、"
                     "规则裁决、给出带依据的状态候选。一次输出全部结果。")
        media_block = f"【本次媒体】\n{media_hint or '见本条消息附带的内容'}"
        situation_label = "【拍摄情境】(由用户提供,可能不准确;与媒体内容冲突时以媒体为准)"

    user = f"""{task_line}

【本次素材类型】{modality_label}

{media_block}

{situation_label}
{context or "（未提供）"}

【运动学指标】由帧差曲线计算得出,可作为动作幅度与节律的旁证:
{_dump_json(motion_features)}

{step1}

{step2}

════════ 第三步:时序对齐(填入 timeline) ════════

把视觉观察与声学事件按同一时间轴合并。
- 时段切分以声学事件边界和视觉信号发生变化的时刻为准,不要均匀切分。
- timeline 中只能出现第一、二步已经记录的观察,不得引入新观察。
- 每个时段相对上一时段的方向,依据 escalation_ladder 判断:变化后的取值出现在更高
  level 中为"升级",更低 level 为"缓解",同级替换为"横向变化",并注明依据的级别。

════════ 第四步:应用规则(填入 conflicts) ════════

逐条检查知识库的 context_modifiers 与 conflict_rules:
- context_modifiers:命中 override_when 条件时必须按 override_to 解读,
  并把这次改写记进 conflicts,rule 字段写出该条目原文。
- conflict_rules:出现 conflicting 列出的信号组合时按 resolution 裁决,
  并遵守其 priority 指定的通道优先级。
- 规则未覆盖的冲突同样要记录,resolved 写 false。

════════ 第五步:候选与结论 ════════

candidates:
- confidence 严格小于 0.5 的候选直接不要输出,不要输出后再标注低置信度。
- 每条 evidence 必须包含:通道名、具体观察值、时间戳、知识库依据原文、信号置信度等级。
- 低置信度(低)的信号不得单独支撑一个候选,至少要有一条中或高置信度证据。
- 候选可以为空数组。

{_sufficiency_rules(modality, knowledge)}

证据不足是合法且有价值的结论。禁止为了给出答案而降低标准、拼凑证据或提高置信度。
反过来也一样:**证据确实充分时不得为了显得谨慎而判证据不足**,那同样是错误输出。

════════ 第六步:多帧一致性(填入 segment_consistency) ════════
{_consistency_block(modality)}

════════ 第七步:反方论证(填入 counter_evidence)—— 强制,不可跳过 ════════

给出主判断后,必须反过来论证"它在什么情况下是错的"。这一步的目的是压制
"看着自信、实则证据不足"的判断。

- 针对置信度最高的候选,列出 1–3 个 counter_hypotheses(反方假设):
  可以是完全相反的状态,也可以是"同一形态的另一种常见解释"(参考知识库的
  context_modifiers 与 conflict_rules,那里记录了大量"形态相同、含义相反"的情形)。
- 对每个反方假设,写出 would_look_like:如果它成立,画面/声音里**本应**出现哪些信号。
- 再写 present_in_media:那些反方信号在本次素材里**是否真的出现**(是 / 否 / 无法确认)。
- verdict:排除这些反方后,主判断是"成立 / 减弱 / 存疑"。
- **若某个反方假设的支持信号确实出现(present_in_media=是),必须相应下调主候选的
  confidence,或把该反方也列为一个候选。** 不允许一边承认反方信号存在、一边维持高置信度。
- insufficient_evidence 为 true 时,counter_evidence 可只写一句"证据不足,无主判断可供反驳"。

translation:
- 这是本次输出中唯一允许拟人化的字段,一句话,用第一人称。
- 必须与置信度最高的候选一致,不得引入候选中没有的状态。
- insufficient_evidence 为 true 时必须表达"这段素材看不出来",禁止给出具体情绪。

suggestion(违反任意一条即为错误输出):
- 禁止建议惩罚、呵斥、压制或以任何方式阻止警告信号。本物种的警告信号包括:
  {warning_signals}
  知识库明确指出:惩罚这些信号会使个体在后续情境中跳过警告直接升级。
- 判读落在 escalation_ladder 任一级上时,首选建议永远是移除压力源、恢复撤离路径、
  停止当前互动,而不是让个体"适应"或"克服"。
- 禁止给出医疗诊断或用药建议。出现疼痛或持续性行为异常的迹象时,只描述观察到的形态,
  并建议由执业兽医或行为专业人员评估。
- insufficient_evidence 为 true 时,suggestion 应给出如何补拍更可判读的素材
  (角度、光照、时长、是否需要同时录音),而不是行为建议。

════════ 全局禁止事项 ════════
- 第一、二步只做形态与声学编码,其中禁止出现任何情绪词。情绪只允许出现在
  candidates 的 label、translation 和 suggestion 中。
- 禁止引入知识库之外的行为学主张,禁止编造文献、作者或年份。
- 禁止为了填满字段而猜测:看不清就是 "不可观测",这不是失败。

【输出 JSON 结构】(字段说明仅为解释用途,实际输出填真实值)
{_dump_json(OMNI_SCHEMA)}

{JSON_ONLY_RULE}"""

    return Prompt(
        system=system,
        user=user,
        species=species,
        task="omni",
        meta={"channels": channels, "patterns": patterns,
              "warning_signals": warning_signals},
    )


if __name__ == "__main__":  # 快速查看三段 prompt 的体积构成
    for sp in ("dog", "cat"):
        kb = load_knowledge(sp)
        v = vision_prompt(sp, "客厅,主人接近", [{"t": 0.0, "role": "baseline"}], {}, kb)
        a = audio_prompt(sp, "客厅,主人接近", kb)
        f = fusion_prompt({}, {}, {}, "客厅,主人接近", kb, species=sp)
        print(f"[{sp}] system={len(v.system)}  vision_user={len(v.user)}  "
              f"audio_user={len(a.user)}  fusion_user={len(f.user)}")

