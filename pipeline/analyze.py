"""判读编排层:把 video / prompts / providers 串起来,供服务端调用。

与 ``providers.run_native`` / ``run_frames`` 的区别:那两个是 CLI 调试入口,
会往 stdout 打大量内容;这里是程序化路径,只通过 ``progress`` 回调汇报进度,
返回结构化结果。
"""

from __future__ import annotations

import os
import time
from typing import Callable

from . import prompts as prompts_mod
from . import providers as providers_mod

Progress = Callable[[str, int, str], None]

MODES = ("native", "frames", "quick", "audio")

# 各模式的进度锚点,用于前端画进度条。模型调用那一段是大头,单独留出区间。
_STAGE_PLAN = {
    "native": [("prepare", 10), ("motion", 25), ("model", 90), ("persist", 100)],
    "frames": [("prepare", 8), ("motion", 18), ("frames", 30),
               ("audio", 38), ("model", 90), ("persist", 100)],
    "quick": [("prepare", 15), ("model", 90), ("persist", 100)],
    "audio": [("prepare", 15), ("model", 90), ("persist", 100)],
}


def _noop(stage: str, pct: int, message: str) -> None:
    pass


def analyze(
    media_path: str,
    species: str,
    context: str,
    mode: str,
    provider,
    knowledge: dict,
    *,
    max_frames: int = 8,
    progress: Progress | None = None,
) -> dict:
    """执行一次判读。

    返回 ``{"result", "usage", "tokens_in", "tokens_out", "latency_s",
    "provider", "model", "motion_features", "mode"}``。
    模型调用失败时 ``result`` 里带 ``error`` 字段,不抛异常。
    """
    report = progress or _noop
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一,收到 '{mode}'")

    started = time.time()
    report("prepare", 5, "准备媒体…")

    if mode == "quick":
        out = _run_quick(media_path, species, context, provider, knowledge, report)
    elif mode == "audio":
        out = _run_audio(media_path, species, context, provider, knowledge, report)
    elif mode == "native":
        out = _run_native(media_path, species, context, provider, knowledge, report)
    else:
        out = _run_frames(media_path, species, context, provider, knowledge,
                          max_frames, report)

    usage = out.get("usage") or {}
    out.update({
        "mode": mode,
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
        "latency_s": round(time.time() - started, 2),
        "provider": provider.name,
        "model": provider.cfg.model,
    })
    return out


# ---------------------------------------------------------------------------

def _finish(provider, result: dict, motion: dict) -> dict:
    return {
        "result": result,
        "usage": getattr(provider, "last_usage", None) or {},
        "motion_features": motion,
        "raw_text": getattr(provider, "last_raw_text", None),
    }


def _run_native(media_path, species, context, provider, knowledge, report) -> dict:
    from .video import motion_features

    report("motion", 15, "计算运动曲线…")
    motion = motion_features(media_path)

    size_mb = os.path.getsize(media_path) / 1024 / 1024
    hint = (f"整段视频文件({size_mb:.1f} MB),由你按 fps={provider.cfg.fps} 自行采样;"
            f"音轨包含在视频内。时间戳以视频起点为 0。")
    prompt = prompts_mod.omni_prompt(species, context, knowledge, motion,
                                     media_hint=hint)

    report("model", 30, "模型判读中,通常 20-30 秒…")
    result = provider.call_omni(media_path, "video", prompt)
    report("model", 90, "模型返回,整理结果…")
    return _finish(provider, result, motion)


def _run_quick(media_path, species, context, provider, knowledge, report) -> dict:
    motion = {
        "duration": None,
        "motion_peaks": [],
        "motion_intensity": "不可观测",
        "motion_pattern": "不可观测",
        "periodicity_hz": None,
        "_note": "单张照片没有时间维度,运动学指标全部不可用。",
    }
    hint = ("单张静态照片,没有时间维度也没有音轨。"
            "visual_signals 只输出一条记录,t 写 0;audio_events 输出空数组;"
            "timeline 只输出一段,direction 写「无变化」。"
            "静态体态本身是可判读的,按单图的判定门槛处理。")
    prompt = prompts_mod.omni_prompt(species, context, knowledge, motion,
                                     media_hint=hint, modality="image")

    report("model", 30, "模型判读中…")
    result = provider.call_omni(media_path, "image", prompt)
    report("model", 90, "模型返回,整理结果…")
    return _finish(provider, result, motion)


def _run_audio(media_path, species, context, provider, knowledge, report) -> dict:
    """纯音频:只做声学分析,visual_signals 留空。"""
    motion = {
        "duration": None,
        "motion_peaks": [],
        "motion_intensity": "不可观测",
        "motion_pattern": "不可观测",
        "periodicity_hz": None,
        "_note": "纯音频没有画面,运动学指标全部不可用。",
    }
    hint = ("一段纯音频,没有任何画面。只做声学事件的切分、描述与模式匹配,"
            "visual_signals 输出空数组,timeline 的 visual 字段一律写 null。")
    prompt = prompts_mod.omni_prompt(species, context, knowledge, motion,
                                     media_hint=hint, modality="audio")

    report("model", 30, "声学分析中…")
    result = provider.call_omni(media_path, "audio", prompt)
    report("model", 90, "模型返回,整理结果…")

    # 兜底:纯音频下视觉一定为空,不允许模型凭空补出视觉观察
    if isinstance(result, dict) and "error" not in result:
        result["visual_signals"] = []
    return _finish(provider, result, motion)


def _run_frames(media_path, species, context, provider, knowledge,
                max_frames, report) -> dict:
    from .video import analysis_window, extract_audio, motion_features, select_frames

    report("motion", 12, "计算运动曲线…")
    motion = motion_features(media_path)

    report("frames", 22, "智能抽帧…")
    frames = select_frames(media_path, max_frames=max_frames)
    frames_meta = [{k: v for k, v in f.items() if k != "image_b64"} for f in frames]

    report("audio", 32, "提取音轨…")
    start, dur = analysis_window(media_path)
    wav = os.path.splitext(media_path)[0] + "_16k.wav"
    audio_path = extract_audio(media_path, wav, start=start, duration=dur)

    report("model", 40, f"视觉编码({len(frames)} 帧)…")
    v_prompt = prompts_mod.vision_prompt(species, context, frames_meta, motion, knowledge)
    v_result = provider.call_omni(frames, "image", v_prompt, capability="vision")
    tokens = [getattr(provider, "last_usage", None) or {}]

    if audio_path:
        report("model", 60, "声学编码…")
        a_prompt = prompts_mod.audio_prompt(species, context, knowledge)
        a_result = provider.call_omni(audio_path, "audio", a_prompt, capability="audio")
        tokens.append(getattr(provider, "last_usage", None) or {})
    else:
        a_result = {"events": [], "audible_events_found": False,
                    "background_noise": "视频无音轨"}

    report("model", 78, "融合判读…")
    f_prompt = prompts_mod.fusion_prompt(v_result, a_result, motion, context,
                                         knowledge, species=species)
    result = provider.call_text(f_prompt)
    tokens.append(getattr(provider, "last_usage", None) or {})

    if "error" not in result:
        result.setdefault("visual_signals", v_result.get("per_frame", []))
        result.setdefault("audio_events", a_result.get("events", []))

    report("model", 90, "模型返回,整理结果…")
    out = _finish(provider, result, motion)
    # 三次调用的用量累加,便于和 native 模式直接比较
    out["usage"] = {
        "prompt_tokens": sum(t.get("prompt_tokens") or 0 for t in tokens),
        "completion_tokens": sum(t.get("completion_tokens") or 0 for t in tokens),
        "total_tokens": sum(t.get("total_tokens") or 0 for t in tokens),
        "_calls": len(tokens),
    }
    out["stage_results"] = {"visual": v_result, "audio": a_result}
    return out
