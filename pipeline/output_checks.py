"""Versioned production output checks for the OpenAI adapter (not CatMeows v1.1)."""
import math

VISUAL_REQUIREMENTS = """
【OpenAI 接入的目标与画质契约】
输出额外 input_assessment 对象，逐项基于本次画面而不是用户预选物种：
target_species_present: true/false/"unknown"；animal_count: 非负整数；
target_unambiguous: boolean（多只动物且未能确定唯一目标时为 false）；
quality: "sufficient"/"insufficient"/"unknown"；
observable_main_axes: 能真正观察到的 body_posture / ear_position 数组；
limitations: 字符串数组，明确裁切、遮挡、模糊、过曝、目标歧义或采样局限。
没有目标物种、目标不明确、画质不足或未知、两个主轴都不可观测时：
insufficient_evidence=true，candidates=[]，并解释补拍要求。看不到的姿态不能补全。
单张图片没有运动信息，不得从静态姿态推断持续行为。置信度是未校准估计，不能宣称准确率。
"""


def output_errors(result: dict, modality: str) -> list[str]:
    errors = []
    if not isinstance(result.get("insufficient_evidence"), bool):
        errors.append("insufficient_evidence 必须为 boolean")
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        errors.append("candidates 必须为数组")
        candidates = []
    if result.get("insufficient_evidence") is True and candidates:
        errors.append("证据不足时 candidates 必须为空")
    if result.get("insufficient_evidence") is False and not candidates:
        errors.append("证据充分时 candidates 不得为空")
    cap = {"image": 0.75, "text": 0.65, "video": 1.0}.get(modality, 1.0)
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            errors.append(f"candidates[{index}] 必须为对象")
            continue
        confidence = item.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= cap):
            errors.append(f"candidates[{index}].confidence 必须在 0..{cap}")
        if (not isinstance(item.get("label"), str) or not item["label"].strip()
                or not isinstance(item.get("evidence"), list) or not item["evidence"]
                or any(not isinstance(e, dict) for e in item["evidence"])):
            errors.append(f"candidates[{index}] 必须有 label 和 evidence")
    for field in ("translation", "suggestion"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            errors.append(f"{field} 必须为非空字符串")
    if modality in {"image", "video"}:
        assessment = result.get("input_assessment")
        if not isinstance(assessment, dict):
            return errors + ["缺少 input_assessment"]
        present = assessment.get("target_species_present")
        if type(present) is not bool and present != "unknown":
            errors.append("input_assessment.target_species_present 类型错误")
        count = assessment.get("animal_count")
        if type(count) is not int or count < 0:
            errors.append("input_assessment.animal_count 必须为非负整数")
        unambiguous = assessment.get("target_unambiguous")
        if type(unambiguous) is not bool:
            errors.append("input_assessment.target_unambiguous 必须为 boolean")
        quality = assessment.get("quality")
        if not isinstance(quality, str) or quality not in {"sufficient", "insufficient", "unknown"}:
            errors.append("input_assessment.quality 非原生枚举")
        axes = assessment.get("observable_main_axes")
        if not isinstance(axes, list) or any(not isinstance(a, str) or a not in {"body_posture", "ear_position"} for a in axes):
            errors.append("input_assessment.observable_main_axes 非原生枚举")
            axes = []
        limitations = assessment.get("limitations")
        if not isinstance(limitations, list) or any(not isinstance(s, str) for s in limitations):
            errors.append("input_assessment.limitations 必须为字符串数组")
        blocked = present is not True or unambiguous is not True or quality != "sufficient" or not axes or count == 0
        if blocked and (result.get("insufficient_evidence") is not True or candidates):
            errors.append("目标/画质/主轴 gate 未通过，必须拒判")
        if blocked and not limitations:
            errors.append("拒判必须明确记录 input_assessment.limitations")
    return errors
