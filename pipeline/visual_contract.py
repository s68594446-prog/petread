"""Evidence-first visual revision; independent of frozen CatMeows v1.1.

These checks verify structure/provenance, not whether an image interpretation is
true. No guessed facts, score repairs, or automatic retries are introduced.
"""
import math
import re

VERSION = "petread.visual.v2.1"
CHANNELS = {"body_posture", "ear_position", "tail", "eye", "pupil",
            "whisker_position", "mouth", "coat", "position", "vocalization"}
MENTAL_WORDS = re.compile(
    r"开心|生气|害怕|焦虑|放松|享受|紧张|恐惧|愤怒|疼痛|痛苦|敌意|攻击性|"
    r"求关注|求食|想要|想吃|讨好|分离焦虑|愉快|抑郁|"
    r"\b(happy|angry|afraid|anxious|relaxed|enjoying|aggressive|painful|attention.seeking)\b",
    re.IGNORECASE,
)
IMAGE_MOTION = re.compile(r"喘气|呼吸急促|摇尾|摆尾|持续|反复|一直|正在走|来回|"
                          r"\b(panting|wagging|repeatedly|continuously)\b", re.IGNORECASE)


def requirements(modality: str) -> str:
    return f"""
【视觉证据契约 {VERSION}，优先于旧模板的同名字段说明】
旧 visual_signals 是知识库编码，不是客观观察正文。知识库的“放松侧卧”等混合枚举
只能留在编码层；不能直接拷贝成事实。编码不匹配不等于画面上什么都看不到。
不要降低原有目标、画质、主轴或置信度门槛，也不要强迫给出情绪候选。

额外输出 visual_assessment_version="{VERSION}" 和 literal_observations 数组。
每项严格为 {{"id":"o1", "source":"visual", "t":0, "channel":"body_posture",
"status":"observed", "description":"身体侧面接触地面，四肢可见", "reason":null}}。
以上只演示格式，不是本次事实，不得照抄。id 唯一；source 只取 visual/audio；
channel 只取 {sorted(CHANNELS)}。status 只取 observed/unobservable。
observed 时 description 是非空、字面的形态或声学描述，reason=null。
unobservable 时 description=null，reason 明确遮挡/裁切等具体原因；不能补全。
体态看得见而张力无法判断时，保留可见形态的 observed 记录，在 limitations 说明
张力不明；可以没有候选，不要把整只动物写成不可见。观察层不写意图、情绪或诊断，
例如“口部张开、舌头外露”不能改写成“开心/喘气”。未知写 null，不补零。
每条 visual 的 t 必须是实际传入帧的时间。抽帧之间的连续动作不能当作已观察事实。
本次 modality={modality}。单图仅 t=0、source=visual，audio_events=[]；单图不能证明
呼吸节律、持续状态或摆尾。视频音轨若不可用，不得创建 audio 观察。

input_assessment 保留目标/质量字段，新增 target_description（唯一目标的可见位置与
外观；不能唯一确定时 null）。animal_count 是画面动物数，不是所选物种数。
target_species_present=false 时 target_unambiguous=false。目标存在且 count=0 是矛盾。
quality 只描述成像是否足够清楚；没有猫狗不是画质差。多动物且不能唯一定位目标时拒判，
不混合不同个体的线索。observable_main_axes 仅列真正能可靠编码的主轴，每个轴必须有
同通道 observed 的字面记录；未命中知识库编码不删除字面观察。

候选 evidence 每项新增 observation_ids（非空 id 数组），只能引用 observed 记录；
evidence.observation 必须逐字等于这些记录的 description 用“；”连接后的文字。
knowledge_basis 写知识库依据，不要放进 observation。候选必须有至少一项中/高可靠性
依据。每个候选新增 alternative_explanations（非空字符串数组，写不同解释及区分所缺
证据），不能仅重复候选 label。分数仍是未校准估计，不是正确概率；不得自动加分。
translation 改为第三人称、带“可能”的条件性摘要，不使用第一人称替宠物说话；
拒判时只说明不能判断及所缺信息。建议不得含医疗诊断/用药或惩罚警告信号。

本版本未验证疼痛量表评分：pain_indicators 必须严格为
{{"status":"not_assessed","reason":"说明本次未进行可靠量表评估的原因"}}，不得输出
action_units 数值、total、normalized_score 或自创临床阈值。可见面部形态留在字面观察，
不把无法评分解释为没有不适。此规则覆盖旧模板要求自动打分的段落。
input_assessment.limitations 至少说明单图/抽帧的局限；insufficient_evidence=true 时
还必须有非空 insufficient_reason。这些新字段不能通过省略来规避检查。
"""


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def errors(result: dict, modality: str, *, frame_times=None, audio_available=False) -> list[str]:
    out = []
    if result.get("visual_assessment_version") != VERSION:
        out.append(f"visual_assessment_version 必须为 {VERSION}")
    observations = result.get("literal_observations")
    if not isinstance(observations, list):
        out.append("literal_observations 必须为数组")
        observations = []
    known, observed_channels = {}, set()
    for i, obs in enumerate(observations):
        path = f"literal_observations[{i}]"
        if not isinstance(obs, dict):
            out.append(f"{path} 必须为对象")
            continue
        oid = obs.get("id")
        if not _text(oid) or oid in known:
            out.append(f"{path}.id 必须非空且唯一")
        else:
            known[oid] = obs
        source, channel, status = obs.get("source"), obs.get("channel"), obs.get("status")
        if not isinstance(channel, str) or channel not in CHANNELS:
            out.append(f"{path}.channel 非原生枚举")
        if source not in ("visual", "audio") or (source == "audio" and not audio_available):
            out.append(f"{path}.source 与实际输入不符")
        t = obs.get("t")
        if type(t) not in (int, float) or not math.isfinite(t) or t < 0:
            out.append(f"{path}.t 必须为非负有限数值")
        elif source == "visual":
            times = [0] if modality == "image" else frame_times
            if times is not None and not any(abs(t - ft) < 0.001 for ft in times):
                out.append(f"{path}.t 不是实际传入帧的时间")
        description = obs.get("description")
        if status == "observed":
            if not _text(description) or obs.get("reason") is not None:
                out.append(f"{path} observed 必须有字面 description 且 reason=null")
            elif MENTAL_WORDS.search(description) or (modality == "image" and IMAGE_MOTION.search(description)):
                out.append(f"{path}.description 混入状态/意图或单图不可确认的持续动作")
            if source == "visual" and isinstance(channel, str):
                observed_channels.add(channel)
        elif status == "unobservable":
            if description is not None or not _text(obs.get("reason")):
                out.append(f"{path} unobservable 必须 description=null 且说明原因")
        else:
            out.append(f"{path}.status 非原生枚举")

    assessment = result.get("input_assessment")
    if isinstance(assessment, dict):
        present, unique = assessment.get("target_species_present"), assessment.get("target_unambiguous")
        count = assessment.get("animal_count")
        if (present is True and count == 0) or (present is not True and unique is True):
            out.append("input_assessment 目标存在/数量/唯一性互相矛盾")
        if unique is True and not _text(assessment.get("target_description")):
            out.append("input_assessment.target_description 必须定位唯一目标")
        if unique is not True and assessment.get("target_description") is not None:
            out.append("目标不唯一时 target_description 必须为 null")
        axes = assessment.get("observable_main_axes")
        for axis in axes if isinstance(axes, list) else []:
            if isinstance(axis, str) and axis not in observed_channels:
                out.append(f"input_assessment.observable_main_axes.{axis} 缺少同通道字面观察")
        limitations = assessment.get("limitations")
        if not isinstance(limitations, list) or not limitations or any(not _text(s) for s in limitations):
            out.append("input_assessment.limitations 必须明确素材局限")
    if result.get("insufficient_evidence") is True and not _text(result.get("insufficient_reason")):
        out.append("insufficient_reason 必须说明为何拒判")
    candidates = result.get("candidates")
    for i, cand in enumerate(candidates if isinstance(candidates, list) else []):
        if not isinstance(cand, dict):
            continue
        path = f"candidates[{i}]"
        confidence = cand.get("confidence")
        if type(confidence) in (int, float) and confidence < 0.5:
            out.append(f"{path}.confidence 低于候选门槛，不可通过修复提高")
        alternatives = cand.get("alternative_explanations")
        if not isinstance(alternatives, list) or not alternatives or any(
            not _text(a) or a == cand.get("label") for a in alternatives
        ):
            out.append(f"{path}.alternative_explanations 必须提供不同解释")
        evidence = cand.get("evidence")
        evidence = evidence if isinstance(evidence, list) else []
        reliable_channels = set()
        if not any(isinstance(e, dict) and e.get("signal_confidence") in ("中", "高") for e in evidence):
            out.append(f"{path} 缺少中/高可靠性依据")
        for j, e in enumerate(evidence):
            if not isinstance(e, dict):
                continue
            ep = f"{path}.evidence[{j}]"
            ids = e.get("observation_ids")
            if not isinstance(ids, list) or not ids or any(not _text(o) for o in ids):
                out.append(f"{ep}.observation_ids 必须引用字面观察")
                continue
            refs = [known.get(oid) for oid in ids]
            if any(not r or r.get("status") != "observed" or not _text(r.get("description")) for r in refs):
                out.append(f"{ep} 引用了不存在/不可观测的证据")
            elif e.get("observation") != "；".join(r["description"] for r in refs):
                out.append(f"{ep}.observation 必须原样引用，不能新增推断")
            elif e.get("signal_confidence") in ("中", "高"):
                reliable_channels.update(r["channel"] for r in refs
                    if isinstance(r.get("channel"), str) and r["channel"] in
                    {"body_posture", "ear_position", "tail", "eye", "pupil", "whisker_position", "mouth"})
            if not _text(e.get("knowledge_basis")):
                out.append(f"{ep}.knowledge_basis 不得为空")
        if modality == "image":
            axes = assessment.get("observable_main_axes", []) if isinstance(assessment, dict) else []
            axes = axes if isinstance(axes, list) else []
            if len(reliable_channels) < 2 or not any(a in reliable_channels for a in axes if isinstance(a, str)):
                out.append(f"{path} 单图候选需要两个独立可靠通道且含可编码主轴；不满足时不得猜测")
    translation = result.get("translation")
    if _text(translation):
        if re.search(r"(^|[，。！？：\s])我(们|想|要|很|正|这|在|是|觉得|需要|不|没)", translation):
            out.append("translation 不得以第一人称替宠物表达内心")
        if candidates and "可能" not in translation:
            out.append("translation 必须明确候选只是可能的解释")
    pain = result.get("pain_indicators")
    if (not isinstance(pain, dict) or set(pain) != {"status", "reason"}
            or pain.get("status") != "not_assessed" or not _text(pain.get("reason"))):
        out.append("pain_indicators 必须为 not_assessed/reason，不输出未经验证的评分")
    if modality == "image" and result.get("audio_events") != []:
        out.append("单图 audio_events 必须为空")
    return out
