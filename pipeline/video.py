"""视频预处理:运动曲线、智能抽帧、运动学特征、音轨提取。

设计约定
--------
* 所有对外函数对损坏视频 / 超短视频 / 无音轨视频做优雅降级,不抛异常。
* ``motion_curve`` 返回的是**整段视频**的逐帧运动强度;
  "超过 15 秒截取运动最密集的 10 秒窗口" 这条规则在分析层
  (``select_frames`` / ``motion_features``)生效,由 ``_pick_window`` 统一计算。
* 运动强度单位是 320x180 灰度图上相邻帧 absdiff 的均值(0~255)。
"""

from __future__ import annotations

import base64
import io
import math
import os
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

try:  # scipy 是抽帧策略的依赖,缺失时退化到自带的峰值检测
    from scipy.signal import find_peaks as _scipy_find_peaks
except Exception:  # pragma: no cover - 环境缺 scipy 时才走到
    _scipy_find_peaks = None

try:
    import ffmpeg
except Exception:  # pragma: no cover - 环境缺 ffmpeg-python 时才走到
    ffmpeg = None


# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

DIFF_SIZE = (320, 180)          # 帧差计算的缩放尺寸
FRAME_LONG_EDGE = 768           # 输出帧长边
JPEG_QUALITY = 85
DEFAULT_FPS = 25.0              # 元数据里 fps 不可信时的兜底值

MAX_RAW_SECONDS = 15.0          # 超过这个时长就裁窗口
WINDOW_SECONDS = 10.0           # 裁出来的窗口长度

PEAK_OFFSET = 0.2               # 峰值前后各抽一帧的偏移(秒)
DEDUP_TOLERANCE = 0.12          # 时间戳去重容差(秒)
STATIC_VARIANCE = 0.05          # 曲线方差低于此值视为"完全静止"

PERIODICITY_BAND = (0.5, 8.0)   # FFT 主频搜索范围(Hz)

_ROLE_PRIORITY = {"peak": 3, "pre": 2, "post": 2, "baseline": 1}


# --------------------------------------------------------------------------
# 内部工具
# --------------------------------------------------------------------------

def _safe_fps(cap: "cv2.VideoCapture") -> float:
    """读 fps,遇到 0 / nan / 离谱值时兜底。"""
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    except Exception:
        fps = 0.0
    if not math.isfinite(fps) or fps <= 1.0 or fps > 240.0:
        return DEFAULT_FPS
    return fps


def _open(video_path: str) -> "cv2.VideoCapture | None":
    """打开视频,失败返回 None(不抛异常)。"""
    if not video_path or not os.path.isfile(video_path):
        return None
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception:
        return None
    if not cap or not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    return cap


def _motion_curve_raw(video_path: str) -> tuple[np.ndarray, float]:
    """返回 (曲线, fps)。曲线第 i 项对应第 i 帧与第 i+1 帧之间的差异。

    读取失败 / 空视频 / 只有一帧时返回空数组。
    """
    cap = _open(video_path)
    if cap is None:
        return np.zeros(0, dtype=np.float32), DEFAULT_FPS

    fps = _safe_fps(cap)
    diffs: list[float] = []
    prev: np.ndarray | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            try:
                small = cv2.resize(frame, DIFF_SIZE, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            except Exception:
                # 单帧解码出问题就跳过,不影响整体
                continue
            if prev is not None:
                diffs.append(float(np.mean(cv2.absdiff(gray, prev))))
            prev = gray
    except Exception:
        # 中途解码崩了也返回已经拿到的部分
        pass
    finally:
        cap.release()

    return np.asarray(diffs, dtype=np.float32), fps


def _curve_time(index: int | np.ndarray, fps: float):
    """曲线下标 -> 时间戳(秒)。第 i 项是第 i 与 i+1 帧之差,取中点。"""
    return (np.asarray(index, dtype=np.float64) + 1.0) / fps


def _pick_window(curve: np.ndarray, fps: float) -> tuple[int, int]:
    """视频超过 15 秒时,返回运动最密集的 10 秒窗口 [start, end) 的曲线下标。

    不超过 15 秒则返回整段。
    """
    n = curve.size
    if n == 0:
        return 0, 0
    duration = n / fps
    if duration <= MAX_RAW_SECONDS:
        return 0, n

    win = max(1, int(round(WINDOW_SECONDS * fps)))
    if win >= n:
        return 0, n

    # 滑动窗口能量和,取最大者
    cumulative = np.concatenate(([0.0], np.cumsum(curve, dtype=np.float64)))
    sums = cumulative[win:] - cumulative[:-win]
    start = int(np.argmax(sums))
    return start, start + win


def _find_peaks(y: np.ndarray, height: float, distance: int) -> np.ndarray:
    """scipy.signal.find_peaks 的薄封装,scipy 缺失时用等价的简易实现。"""
    if y.size < 3:
        return np.zeros(0, dtype=int)

    if _scipy_find_peaks is not None:
        try:
            peaks, _ = _scipy_find_peaks(y, height=height, distance=max(1, distance))
            return np.asarray(peaks, dtype=int)
        except Exception:
            pass

    # 兜底:局部极大 + 阈值 + 按幅值贪心地施加最小间距
    candidates = [
        i for i in range(1, y.size - 1)
        if y[i] >= y[i - 1] and y[i] > y[i + 1] and y[i] >= height
    ]
    candidates.sort(key=lambda i: float(y[i]), reverse=True)
    kept: list[int] = []
    for i in candidates:
        if all(abs(i - j) >= max(1, distance) for j in kept):
            kept.append(i)
    return np.asarray(sorted(kept), dtype=int)


def _detect_peaks(curve: np.ndarray, fps: float) -> np.ndarray:
    """在曲线上找运动峰值,返回曲线下标。已内置最小间距,避免峰值扎堆。"""
    if curve.size < 3:
        return np.zeros(0, dtype=int)
    # 与 select_frames 的"完全静止"判据保持一致,避免出现
    # motion_intensity=静止 却报出峰值这种自相矛盾的结果
    if float(np.var(curve)) < STATIC_VARIANCE:
        return np.zeros(0, dtype=int)
    mean = float(np.mean(curve))
    std = float(np.std(curve))
    if std <= 1e-6:
        return np.zeros(0, dtype=int)
    height = mean + 0.5 * std
    distance = max(1, int(round(0.4 * fps)))  # 至少间隔 0.4s
    return _find_peaks(curve, height=height, distance=distance)


def _encode_frame(frame: np.ndarray) -> str:
    """BGR ndarray -> 长边 768 的 JPEG base64。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    long_edge = max(img.size)
    if long_edge > FRAME_LONG_EDGE:
        scale = FRAME_LONG_EDGE / float(long_edge)
        new_size = (max(1, int(round(img.width * scale))),
                    max(1, int(round(img.height * scale))))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _grab_frames(video_path: str, frame_indices: Sequence[int]) -> dict[int, np.ndarray]:
    """按帧号取帧。先试随机 seek,失败的帧再用一次顺序扫描补齐。"""
    out: dict[int, np.ndarray] = {}
    wanted = sorted({int(i) for i in frame_indices if i >= 0})
    if not wanted:
        return out

    cap = _open(video_path)
    if cap is None:
        return out

    try:
        for idx in wanted:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            if ok and frame is not None:
                out[idx] = frame
    finally:
        cap.release()

    missing = [i for i in wanted if i not in out]
    if not missing:
        return out

    # 有些容器 seek 不可靠,顺序扫一遍补齐
    cap = _open(video_path)
    if cap is None:
        return out
    try:
        pending = set(missing)
        pos = 0
        last_frame = None
        while pending:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            last_frame = frame
            if pos in pending:
                out[pos] = frame
                pending.discard(pos)
            pos += 1
        # 请求的帧号超出实际帧数时,用最后一帧兜底
        if pending and last_frame is not None:
            for idx in pending:
                out[idx] = last_frame
    finally:
        cap.release()

    return out


def _dedup(candidates: list[dict]) -> list[dict]:
    """按时间排序后去重,时间相近时保留角色优先级更高的那个。"""
    ordered = sorted(candidates, key=lambda c: (c["t"], -_ROLE_PRIORITY.get(c["role"], 0)))
    kept: list[dict] = []
    for cand in ordered:
        if kept and abs(cand["t"] - kept[-1]["t"]) < DEDUP_TOLERANCE:
            prev = kept[-1]
            if _ROLE_PRIORITY.get(cand["role"], 0) > _ROLE_PRIORITY.get(prev["role"], 0):
                kept[-1] = cand
            continue
        kept.append(cand)
    return kept


def _uniform_times(duration: float, count: int) -> list[float]:
    """静止视频的退化策略:均匀抽帧。"""
    count = max(1, count)
    if duration <= 0:
        return [0.0]
    if count == 1:
        return [duration / 2.0]
    step = duration / float(count - 1) if count > 1 else duration
    return [min(duration, round(i * step, 3)) for i in range(count)]


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------

def motion_curve(video_path: str) -> np.ndarray:
    """逐帧计算帧差,返回运动强度曲线(float32 一维数组)。

    缩放到 320x180 灰度后取 absdiff 均值。读取失败或视频为空时返回空数组。
    """
    curve, _fps = _motion_curve_raw(video_path)
    return curve


def select_frames(video_path: str, max_frames: int = 8) -> list[dict]:
    """基于运动曲线的智能抽帧。

    策略:峰值 ± 0.2s 三连拍 + 首尾基线帧,排序去重后按运动强度截断到
    ``max_frames``;曲线方差极低(几乎静止)时退化为均匀抽帧。

    返回 ``[{"t": float, "role": str, "image_b64": str}, ...]``,按时间升序。
    """
    max_frames = max(1, int(max_frames))
    curve, fps = _motion_curve_raw(video_path)

    if curve.size == 0:
        # 空视频 / 损坏 / 只有一帧:尽力抓第一帧
        frames = _grab_frames(video_path, [0])
        if 0 not in frames:
            return []
        return [{"t": 0.0, "role": "baseline", "image_b64": _encode_frame(frames[0])}]

    win_start, win_end = _pick_window(curve, fps)
    window = curve[win_start:win_end]
    t_first = float(_curve_time(win_start, fps)) if win_start else 0.0
    t_last = float(_curve_time(max(win_start, win_end - 1), fps))
    duration = max(0.0, float(t_last - t_first))

    candidates: list[dict] = []

    def add(t: float, role: str) -> None:
        t = float(min(max(t, t_first), t_last))
        candidates.append({"t": round(t, 3), "role": role})

    variance = float(np.var(window)) if window.size else 0.0
    is_static = window.size < 3 or variance < STATIC_VARIANCE

    if is_static:
        # 完全静止:均匀抽帧
        for t in _uniform_times(duration, max_frames):
            add(t_first + t, "baseline")
    else:
        peaks = _detect_peaks(window, fps)
        for p in peaks:
            t_peak = float(_curve_time(win_start + int(p), fps))
            add(t_peak - PEAK_OFFSET, "pre")
            add(t_peak, "peak")
            add(t_peak + PEAK_OFFSET, "post")
        # 首尾基线
        add(t_first, "baseline")
        add(t_last, "baseline")

    picked = _dedup(candidates)

    # 超过上限时按运动强度保留最强的;基线帧在预算允许时优先保住
    if len(picked) > max_frames:
        def intensity(item: dict) -> float:
            idx = int(round(item["t"] * fps)) - 1
            idx = int(np.clip(idx, 0, curve.size - 1))
            return float(curve[idx])

        baselines = [c for c in picked if c["role"] == "baseline"]
        others = [c for c in picked if c["role"] != "baseline"]

        if max_frames >= 4 and baselines:
            reserved = baselines[:2]
            budget = max_frames - len(reserved)
            others.sort(key=intensity, reverse=True)
            picked = reserved + others[:budget]
        else:
            picked.sort(key=intensity, reverse=True)
            picked = picked[:max_frames]
        picked.sort(key=lambda c: c["t"])

    # 时间戳 -> 帧号 -> 图片
    index_of = {}
    for item in picked:
        frame_idx = int(round(item["t"] * fps))
        index_of[id(item)] = max(0, frame_idx)
    frames = _grab_frames(video_path, list(index_of.values()))

    result: list[dict] = []
    for item in picked:
        frame = frames.get(index_of[id(item)])
        if frame is None:
            continue
        try:
            b64 = _encode_frame(frame)
        except Exception:
            continue
        result.append({"t": item["t"], "role": item["role"], "image_b64": b64})

    return result


def motion_features(video_path: str) -> dict:
    """提取运动学指标。

    返回 ``{"duration", "motion_peaks", "motion_intensity",
    "motion_pattern", "periodicity_hz"}``。视频不可读时返回一份保守的默认值。
    """
    curve, fps = _motion_curve_raw(video_path)

    if curve.size == 0:
        return {
            "duration": 0.0,
            "motion_peaks": [],
            "motion_intensity": "静止",
            "motion_pattern": "无",
            "periodicity_hz": None,
        }

    duration = round(float(curve.size + 1) / fps, 3)

    win_start, win_end = _pick_window(curve, fps)
    window = curve[win_start:win_end]

    peak_idx = _detect_peaks(window, fps)
    peaks = [round(float(_curve_time(win_start + int(p), fps)), 3) for p in peak_idx]

    return {
        "duration": duration,
        "motion_peaks": peaks,
        "motion_intensity": _classify_intensity(window, peaks, fps),
        "motion_pattern": _classify_pattern(peaks, window.size / fps if fps else 0.0),
        "periodicity_hz": _dominant_frequency(window, fps),
    }


def _classify_intensity(curve: np.ndarray, peaks: Sequence[float], fps: float) -> str:
    """用曲线均值 + 峰值密度归纳出四档强度。"""
    if curve.size == 0:
        return "静止"
    mean = float(np.mean(curve))
    duration = curve.size / fps if fps > 0 else 0.0
    peaks_per_sec = (len(peaks) / duration) if duration > 0 else 0.0

    if mean < 0.8 and float(np.var(curve)) < STATIC_VARIANCE:
        return "静止"
    if mean < 2.5:
        level = 1          # 轻微
    elif mean < 8.0:
        level = 2          # 中等
    else:
        level = 3          # 剧烈

    # 峰值又多又密说明动作更连贯,往上抬一档;几乎没有峰值则往下压一档
    if peaks_per_sec >= 1.2 and level < 3:
        level += 1
    elif len(peaks) == 0 and level > 1:
        level -= 1

    return {1: "轻微", 2: "中等", 3: "剧烈"}[level]


def _classify_pattern(peaks: Sequence[float], duration: float) -> str:
    """描述峰值在时间轴上的分布形态。"""
    n = len(peaks)
    if n == 0:
        return "无"
    if n == 1:
        return "单峰"
    if n == 2:
        return "双峰"

    gaps = np.diff(np.asarray(peaks, dtype=np.float64))
    mean_gap = float(np.mean(gaps))
    if mean_gap <= 1e-6:
        return "连续"
    cv = float(np.std(gaps)) / mean_gap          # 间隔的变异系数
    coverage = (peaks[-1] - peaks[0]) / duration if duration > 0 else 0.0

    # 间隔均匀且密集,并且铺满大半段视频 -> 连续动作;否则是一阵一阵的
    if mean_gap <= 1.2 and cv < 0.6 and coverage > 0.5:
        return "连续"
    return "间歇"


def _dominant_frequency(curve: np.ndarray, fps: float) -> float | None:
    """对运动曲线做 FFT,返回 0.5~8Hz 内的主频(Hz),否则 None。

    用于捕捉摇尾巴这类周期性动作。样本太短、超出奈奎斯特频率,
    或频谱里没有明显突出的峰时都返回 None。
    """
    n = curve.size
    if n < 8 or fps <= 0:
        return None
    if n / fps < 1.5:            # 不到 1.5 秒,谈不上周期性
        return None

    lo, hi = PERIODICITY_BAND
    if fps / 2.0 <= lo:          # 奈奎斯特频率都够不到下界
        return None

    try:
        signal = curve.astype(np.float64)
        signal = signal - signal.mean()
        if not np.any(signal):
            return None
        signal = signal * np.hanning(n)
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    except Exception:
        return None

    band = (freqs >= lo) & (freqs <= min(hi, fps / 2.0))
    if not np.any(band):
        return None

    band_power = spectrum[band]
    peak_pos = int(np.argmax(band_power))
    peak_power = float(band_power[peak_pos])

    # 主频要明显高过频谱底噪,否则认为没有周期性
    noise = float(np.median(spectrum[1:])) if spectrum.size > 1 else 0.0
    if peak_power <= 0 or (noise > 0 and peak_power < 3.0 * noise):
        return None

    return round(float(freqs[band][peak_pos]), 3)


def extract_audio(
    video_path: str,
    out_path: str,
    start: float | None = None,
    duration: float | None = None,
) -> str | None:
    """拆出音轨,转 16kHz 单声道 wav,返回输出路径;无音轨或失败返回 None。

    ``start`` / ``duration`` 可选,用于只截取分析窗口对应的那段音频。
    """
    if ffmpeg is None:
        return None
    if not video_path or not os.path.isfile(video_path):
        return None

    # 先探测有没有音轨,没有就直接返回,不让 ffmpeg 抛错
    try:
        probe = ffmpeg.probe(video_path)
        streams = probe.get("streams", []) or []
        if not any(s.get("codec_type") == "audio" for s in streams):
            return None
    except Exception:
        return None

    try:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        in_kwargs: dict = {}
        if start is not None and start > 0:
            in_kwargs["ss"] = float(start)
        if duration is not None and duration > 0:
            in_kwargs["t"] = float(duration)

        (
            ffmpeg
            .input(video_path, **in_kwargs)
            .output(out_path, format="wav", acodec="pcm_s16le", ac=1, ar=16000, vn=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except Exception:
        return None

    # 只有一个 44 字节的 wav 头说明其实没解出东西
    if not os.path.isfile(out_path) or os.path.getsize(out_path) <= 44:
        return None
    return out_path


def analysis_window(video_path: str) -> tuple[float, float]:
    """返回实际参与分析的时间窗口 ``(起点秒, 时长秒)``。

    视频不超过 15 秒时就是整段;超过时是运动最密集的 10 秒。
    """
    curve, fps = _motion_curve_raw(video_path)
    if curve.size == 0:
        return 0.0, 0.0
    start, end = _pick_window(curve, fps)
    return round(float(start) / fps, 3), round(float(end - start) / fps, 3)


# --------------------------------------------------------------------------
# 自测入口
# --------------------------------------------------------------------------

def _self_test(video_path: str, max_frames: int, out_dir: str) -> int:
    import json

    if not os.path.isfile(video_path):
        print(f"[x] 文件不存在: {video_path}")
        return 1

    print(f"=== {video_path} ===")

    curve = motion_curve(video_path)
    print(f"\n[运动曲线] 长度={curve.size}")
    if curve.size:
        print(f"  均值={curve.mean():.3f}  最大={curve.max():.3f}  "
              f"方差={curve.var():.3f}")

    win_start, win_dur = analysis_window(video_path)
    print(f"\n[分析窗口] 起点={win_start}s  时长={win_dur}s")

    feats = motion_features(video_path)
    print("\n[运动特征]")
    print(json.dumps(feats, ensure_ascii=False, indent=2))

    frames = select_frames(video_path, max_frames=max_frames)
    print(f"\n[抽帧] 共 {len(frames)} 帧")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    for i, f in enumerate(frames):
        name = f"{stem}_{i:02d}_{f['t']:.2f}s_{f['role']}.jpg"
        path = os.path.join(out_dir, name)
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(f["image_b64"]))
        print(f"  t={f['t']:>7.2f}s  role={f['role']:<8} "
              f"b64={len(f['image_b64']):>7d}B  -> {path}")

    wav_path = os.path.join(out_dir, f"{stem}_16k.wav")
    audio = extract_audio(video_path, wav_path, start=win_start, duration=win_dur)
    print(f"\n[音轨] {audio if audio else '无音轨或提取失败(已降级为 None)'}")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="petread 视频预处理自测")
    parser.add_argument("video", help="待测视频路径")
    parser.add_argument("--max-frames", type=int, default=8, help="最多抽几帧")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "frames",
        ),
        help="抽帧输出目录,默认 data/frames/",
    )
    args = parser.parse_args()

    raise SystemExit(_self_test(args.video, args.max_frames, args.out_dir))
