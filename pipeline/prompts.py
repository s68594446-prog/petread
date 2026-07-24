"""三段链路的 prompt 模板与知识库注入。

设计约定
--------
* 知识库按 species 全量注入(暂不裁剪),``confidence`` / ``value_confidence``
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


def build_system_prompt(species: str, knowledge: dict, task: str) -> str:
    """三段共用的 system prompt:角色 + 知识库全量 + 约束 + 输出格式。"""
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
        "不得编造文献、作者或年份。",
        "```yaml",
        _dump_yaml(knowledge),
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
# 4. 全模态单次调用(MiMo-V2.5 等原生多模态模型)
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


def omni_prompt(
    species: str,
    context: str,
    knowledge: dict,
    motion_features: dict,
    media_hint: str = "",
    now: datetime | None = None,
) -> Prompt:
    """全模态单次调用:视觉编码 + 声学编码 + 时序对齐 + 规则裁决 + 候选输出。

    ``media_hint`` 用于说明本次附带的媒体形态(整段视频 / 若干抽帧 + 音轨),
    让模型知道时间戳的参照系。

    system 以 MiMo 官方建议的身份/日期提示词开头,项目自己的知识库、硬约束与
    输出约束接在其后。
    """
    system = (
        mimo_system_preamble(now)
        + "\n\n"
        + build_system_prompt(species, knowledge, "fusion")
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

    user = f"""【任务】对附带的媒体做一次完整判读:视觉信号编码、声学事件编码、时序对齐、
规则裁决、给出带依据的状态候选。一次输出全部结果。

【本次媒体】
{media_hint or "见本条消息附带的内容"}

【拍摄情境】(由用户提供,可能不准确;与媒体内容冲突时以媒体为准)
{context or "（未提供）"}

【运动学指标】由帧差曲线计算得出,可作为动作幅度与节律的旁证:
{_dump_json(motion_features)}

════════ 第一步:视觉信号编码(填入 visual_signals) ════════

在时间轴上选取信号发生变化的时刻(不要均匀采样),对每个时刻遍历知识库 signals
下的全部通道并逐个填写:
{channels}
- 取值必须严格来自该通道的 values 列表,不得自创描述。
- 带 dimensions 的通道(如 tail)按其子维度分别填写。
- 无法可靠判读时取值写 "不可观测",并把通道名放进 uncertain。
- 因品种形态学而结构性不可用的通道(见各通道 caveats)同样写 "不可观测"。
- 每个取值同时给出它在知识库里的 value_confidence 等级。
- 每条记录必须带时间戳 t。

════════ 第二步:声学事件编码(填入 audio_events) ════════

找出音轨中每一个可判读的发声事件,描述其时长、基频走向、估计基频范围、音节数、
有无颤抖成分、能量包络、有无宽带噪声成分,然后匹配到知识库 acoustic 表中的一项:
{patterns}
- 匹配依据必须是声学特征本身,不是情境推测。
- 都不匹配就写 "未匹配",这比强行归类更有价值。
- match_confidence 不得高于所匹配 pattern 在知识库中自带的 confidence。
- 每个事件必须带 t_start / t_end,精确到 0.1 秒。
- 知识库中的 f0_range 是粗分档不是精确阈值,偏离几十赫兹不影响匹配,偏离一个数量级时写 "未匹配"。
- 低频事件(如 purr 的 25-30Hz)可能低于录音与压缩的有效响应下限,未检测到不等于未发生。
- 无任何目标物种发声时 audio_events 输出空数组,这是合法结果,不要把环境噪声当成发声。

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

insufficient_evidence 必须为 true(且 candidates 置空、insufficient_reason 必填)当:
- 可观测通道少于 3 个;
- 知识库 meta.reading_order 中作为判读主轴的通道不可观测;
- conflicts 中存在 resolved=false 且该冲突涉及判读主轴;
- 所有候选的 confidence 都低于 0.5。
证据不足是合法且有价值的结论。禁止为了给出答案而降低标准、拼凑证据或提高置信度。

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
