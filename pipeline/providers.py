"""模型调用层:provider 可切换,模型名全部来自 config,不在代码里硬编码。

结构
----
* ``BaseProvider``          能力方法的抽象基类,不支持的能力返回统一的错误 dict。
* ``OpenAICompatibleProvider``  text / vision / audio / video / omni,
  支持 text、image_url、video_url、input_audio 四种 content type。
  当前主力 MiMo-V2.6-Flash 走这一条(type: openai)。
* ``DeepSeekProvider``      OpenAI 兼容的子类,备选
* ``GeminiProvider``        audio,备选
* ``MockProvider``          按 prompt.task 分派,生成对应 schema 的假数据

约定
----
* 所有公开调用只返回 dict,不抛异常。解析失败返回 ``{"error": ..., "raw": ...}``。
* API key 只从环境变量读,config 里只写变量名。
* 重试:指数退避,最多 ``max_retries`` 次;只对网络错误和 5xx 重试,4xx 不重试。
* MiMo 的 thinking / max_completion_tokens 在顶层；fps / media_resolution 在视频 part。
* base64 内联能力必须用 ``probe_base64_support`` 实测,不做静默降级。
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import yaml

from . import prompts as prompts_mod
from .prompts import Prompt, load_knowledge

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

DEFAULT_CONFIDENCE_FLOOR = 0.5
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 1.0
MAX_BACKOFF = 30.0

# 按需求只对网络错误和 5xx 重试。若将来要对限流重试,把 429 加进这个集合即可。
EXTRA_RETRY_STATUS: frozenset[int] = frozenset()


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """provider 层的基类异常,不会泄漏到公开方法之外。"""


class ProviderNetworkError(ProviderError):
    """连接失败、超时、DNS 等传输层错误,可重试。"""


class ProviderHTTPError(ProviderError):
    """HTTP 非 2xx。5xx 可重试,4xx 不可。"""

    def __init__(self, status: int, body: str = ""):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body

    @property
    def retryable(self) -> bool:
        return self.status >= 500 or self.status in EXTRA_RETRY_STATUS


class ProviderConfigError(ProviderError):
    """配置缺失或非法,不可重试。"""


# ---------------------------------------------------------------------------
# JSON 解析:兼容模型仍然包了 markdown 代码块的情况
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """剥掉 markdown 代码块。有多个时取最长的一段。"""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def _extract_outermost_object(text: str) -> str | None:
    """从自由文本中截出第一个完整的顶层 JSON 对象,按括号配对扫描。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json_response(text: str) -> dict:
    """把模型返回的文本解析成 dict。失败时返回 ``{"error", "raw"}``。"""
    if text is None:
        return {"error": "模型返回为空", "raw": ""}
    if isinstance(text, dict):
        return text

    raw = str(text)
    candidates = [_strip_fences(raw), raw]
    extracted = _extract_outermost_object(_strip_fences(raw))
    if extracted:
        candidates.insert(0, extracted)

    for cand in candidates:
        if not cand:
            continue
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
        return {"error": f"模型返回的是 {type(parsed).__name__},不是 JSON 对象", "raw": raw}

    return {"error": "无法解析为 JSON", "raw": raw}


def filter_low_confidence(result: dict, floor: float = DEFAULT_CONFIDENCE_FLOOR) -> dict:
    """丢弃置信度低于 floor 的候选。prompt 里已经要求过,这里再兜一次。

    全部候选都被丢弃时,置 ``insufficient_evidence=true``。
    """
    if not isinstance(result, dict) or "candidates" not in result:
        return result
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        return result

    kept, dropped = [], []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        try:
            conf = float(cand.get("confidence"))
        except (TypeError, ValueError):
            dropped.append(cand)          # 置信度不可解析,按不合格处理
            continue
        (kept if conf >= floor else dropped).append(cand)

    result["candidates"] = kept
    if dropped:
        result.setdefault("_dropped_candidates", len(dropped))
    if not kept and not result.get("insufficient_evidence"):
        result["insufficient_evidence"] = True
        result.setdefault(
            "insufficient_reason",
            f"所有候选的置信度均低于 {floor},已按证据不足处理。",
        )
    return result


# ---------------------------------------------------------------------------
# suggestion 护栏:知识库要求"不得惩罚警告信号",prompt 里写了,这里再兜一次
# ---------------------------------------------------------------------------

# 在本领域内任何语境下都不该出现的处置动词,不需要邻近判定
_ALWAYS_BANNED_VERBS = ("惩罚", "呵斥", "训斥", "责骂", "责备", "凶它", "打它", "教训")

# 只有当它作用在警告信号上时才算违规的动词,需要邻近判定
# (例如"阻止孩子继续接近"是正确建议,"阻止它低吼"才是错误建议)
_CONTEXTUAL_VERBS = ("制止", "压制", "阻止", "纠正", "打断", "喝止", "抑制", "让它别")

# 否定词:出现在动词前方窗口内时,该动词属于劝阻用法,不算违规
_NEGATORS = ("不要", "不能", "不可", "不应", "不得", "别", "禁止", "避免", "勿", "切勿", "而不是")

_PROXIMITY = 14          # 邻近判定的字符半径
_NEGATION_WINDOW = 8     # 否定词回看窗口


def _negated(text: str, verb_start: int) -> bool:
    window = text[max(0, verb_start - _NEGATION_WINDOW):verb_start]
    return any(neg in window for neg in _NEGATORS)


def warning_signal_terms(knowledge: dict, min_level: int = 5) -> list[str]:
    """知识库 escalation_ladder 中 level >= min_level 的信号名。"""
    terms: list[str] = ["警告", "警告信号"]
    for level in knowledge.get("escalation_ladder") or []:
        try:
            if int(level.get("level", 0)) < min_level:
                continue
        except (TypeError, ValueError):
            continue
        for sig in level.get("signals") or []:
            # 去掉括号补充说明,只留核心词
            terms.append(re.split(r"[(（]", str(sig))[0].strip())
    return [t for t in dict.fromkeys(terms) if t]


def check_suggestion_safety(text: str, knowledge: dict) -> list[str]:
    """检查 suggestion 是否违反"不得惩罚警告信号"的约束。

    返回违规描述列表,空列表表示通过。做了否定感知(“不要制止它低吼”不算违规)
    与作用对象邻近判定(“阻止孩子接近”不算违规)。
    """
    if not text:
        return []
    text = str(text)
    violations: list[str] = []

    for verb in _ALWAYS_BANNED_VERBS:
        for match in re.finditer(re.escape(verb), text):
            if not _negated(text, match.start()):
                violations.append(f"出现处置性动词「{verb}」且未被否定")

    terms = warning_signal_terms(knowledge)
    for term in terms:
        for hit in re.finditer(re.escape(term), text):
            lo = max(0, hit.start() - _PROXIMITY)
            hi = min(len(text), hit.end() + _PROXIMITY)
            neighborhood = text[lo:hi]
            for verb in _CONTEXTUAL_VERBS:
                for vmatch in re.finditer(re.escape(verb), neighborhood):
                    if not _negated(neighborhood, vmatch.start()):
                        violations.append(f"「{verb}」作用于警告信号「{term}」")

    return sorted(set(violations))


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    type: str
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    timeout: int = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR
    mock: bool = False
    role: str = ""
    knowledge_dir: str = ""
    # 鉴权头形式:bearer -> Authorization: Bearer xxx;api-key -> api-key: xxx
    auth_header: str = "bearer"
    # thinking / 输出上限在顶层；媒体参数位置按协议选择。
    thinking: bool | None = None
    fps: float | None = None
    media_resolution: str = ""
    media_param_placement: str = "extra_body"   # extra_body | content | content_part
    media_format: str = "openai"              # openai | mimo
    max_completion_tokens: int | None = None

    def __post_init__(self):
        limit = self.max_completion_tokens
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 131072):
            raise ProviderConfigError("max_completion_tokens 必须是 1..131072 的整数")
        if self.media_format not in ("openai", "mimo"):
            raise ProviderConfigError("media_format 必须是 openai 或 mimo")
        if self.media_param_placement not in ("extra_body", "content", "content_part"):
            raise ProviderConfigError("未知 media_param_placement")
        if self.media_format == "mimo":
            if limit is None:
                raise ProviderConfigError("MiMo 必须显式配置 max_completion_tokens")
            if self.media_param_placement != "content_part":
                raise ProviderConfigError("MiMo 视频参数必须使用 content_part")
            if self.fps is not None and not 0.1 <= self.fps <= 10:
                raise ProviderConfigError("MiMo fps 必须在 0.1..10")
            if self.media_resolution not in ("", "default", "max"):
                raise ProviderConfigError("MiMo media_resolution 必须是 default 或 max")

    def extra_body(self) -> dict:
        """按 OpenAI SDK 的 extra_body 语义,这些键直接并入请求体顶层。"""
        body: dict = {}
        if self.thinking is not None:
            body["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
        if self.max_completion_tokens is not None:
            body["max_completion_tokens"] = self.max_completion_tokens
        if self.media_param_placement == "extra_body":
            if self.fps is not None:
                body["fps"] = self.fps
            if self.media_resolution:
                body["media_resolution"] = self.media_resolution
        return body

    def media_params(self) -> dict:
        """媒体参数；content 放在媒体对象内，content_part 放在视频 part 上。"""
        if self.media_param_placement not in ("content", "content_part"):
            return {}
        params: dict = {}
        if self.fps is not None:
            params["fps"] = self.fps
        if self.media_resolution:
            params["media_resolution"] = self.media_resolution
        return params


def load_config(path: str | None = None) -> dict:
    """读 config.yaml。缺文件时抛 ProviderConfigError,信息里带上查找路径。"""
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(cfg_path):
        raise ProviderConfigError(f"找不到配置文件: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "providers" not in cfg:
        raise ProviderConfigError(f"{cfg_path} 缺少 providers 段")
    return cfg


def build_providers(config: dict) -> dict[str, "BaseProvider"]:
    """按 config 中实际声明的角色构建 provider。

    支持两种形态:
      * 全模态:只声明 ``omni``,一次调用完成三段;
      * 三段式:声明 ``vision`` / ``audio`` / ``text``。
    两者可以共存,按需取用。``config["mock"]`` 为 true 时全部替换为 MockProvider。
    """
    mock = bool(config.get("mock", False))
    knowledge_dir = config.get("knowledge_dir") or ""
    if knowledge_dir and not os.path.isabs(knowledge_dir):
        knowledge_dir = os.path.join(PROJECT_ROOT, knowledge_dir)

    declared = {k: v for k, v in (config.get("providers") or {}).items() if v}
    if not declared:
        raise ProviderConfigError(
            "providers 段为空。至少需要声明 omni,或同时声明 vision / audio / text。"
        )
    if "omni" not in declared and not {"vision", "audio", "text"} <= set(declared):
        raise ProviderConfigError(
            f"providers 声明不完整:当前只有 {sorted(declared)};"
            "需要 omni,或者 vision / audio / text 三个都在。"
        )

    built: dict[str, BaseProvider] = {}
    for role, raw in declared.items():
        ptype = str(raw.get("type") or "").strip().lower()
        thinking = raw.get("thinking")
        cfg = ProviderConfig(
            type=ptype,
            model=str(raw.get("model") or "").strip(),
            base_url=str(raw.get("base_url") or "").strip(),
            api_key_env=str(raw.get("api_key_env") or "").strip(),
            timeout=int(config.get("timeout", DEFAULT_TIMEOUT)),
            max_retries=int(config.get("max_retries", DEFAULT_MAX_RETRIES)),
            backoff_base=float(config.get("retry_backoff_base", DEFAULT_BACKOFF_BASE)),
            confidence_floor=float(
                config.get("confidence_floor", DEFAULT_CONFIDENCE_FLOOR)
            ),
            mock=mock,
            role=role,
            knowledge_dir=knowledge_dir,
            auth_header=str(raw.get("auth_header") or "bearer").strip().lower(),
            thinking=None if thinking is None else bool(thinking),
            fps=None if raw.get("fps") is None else float(raw["fps"]),
            media_resolution=str(raw.get("media_resolution") or "").strip(),
            media_param_placement=str(
                raw.get("media_param_placement") or "extra_body"
            ).strip().lower(),
            media_format=str(raw.get("media_format") or "openai").strip().lower(),
            max_completion_tokens=raw.get("max_completion_tokens"),
        )

        if mock:
            built[role] = MockProvider(cfg)
            continue

        cls = PROVIDER_REGISTRY.get(ptype)
        if cls is None:
            raise ProviderConfigError(
                f"{role} 的 provider 类型 '{ptype}' 未注册;"
                f"可用: {sorted(PROVIDER_REGISTRY)}"
            )
        if not cfg.model:
            raise ProviderConfigError(
                f"{role} 的 model 为空。mock 为 false 时必须在 config.yaml 中填写模型名。"
            )
        built[role] = cls(cfg)

    return built


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> str:
    """POST JSON,返回响应体文本。非 2xx 抛 ProviderHTTPError,传输失败抛 ProviderNetworkError。"""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:          # HTTPError 是 URLError 子类,必须先接
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise ProviderHTTPError(exc.code, detail) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderNetworkError(str(exc)) from exc


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class BaseProvider(ABC):
    """三个能力方法的抽象基类。子类只实现自己支持的能力。"""

    name = "base"
    supports: tuple[str, ...] = ()

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg

    # -- 能力方法 ----------------------------------------------------------

    def call_vision(self, frames: Sequence[dict], prompt: Prompt | str) -> dict:
        return self._unsupported("vision")

    def call_audio(self, audio_path: str, prompt: Prompt | str) -> dict:
        return self._unsupported("audio")

    def call_text(self, prompt: Prompt | str) -> dict:
        return self._unsupported("text")

    # -- 工具 --------------------------------------------------------------

    def _unsupported(self, capability: str) -> dict:
        return {
            "error": f"provider '{self.name}' 不支持 {capability} 能力;"
                     f"它支持 {list(self.supports)}",
            "raw": "",
            "_meta": {"provider": self.name, "capability": capability},
        }

    @property
    def api_key(self) -> str:
        env = self.cfg.api_key_env
        if not env:
            raise ProviderConfigError(f"provider '{self.name}' 未配置 api_key_env")
        key = os.environ.get(env, "").strip()
        if not key:
            raise ProviderConfigError(
                f"环境变量 {env} 未设置或为空。api_key 只能来自环境变量,不要写进 config.yaml。"
            )
        return key

    @staticmethod
    def _prompt_parts(prompt: Prompt | str) -> tuple[str, str]:
        if isinstance(prompt, Prompt):
            return prompt.system, prompt.user
        return "", str(prompt)

    def _with_retry(self, fn: Callable[[], str], label: str) -> tuple[str, dict]:
        """执行 fn,按需重试。返回 (响应文本, 调用元信息)。"""
        attempts = 0
        started = time.time()
        last_exc: Exception | None = None

        for attempt in range(1, max(1, self.cfg.max_retries) + 1):
            attempts = attempt
            try:
                text = fn()
                return text, {
                    "provider": self.name,
                    "model": self.cfg.model,
                    "capability": label,
                    "attempts": attempts,
                    "latency_s": round(time.time() - started, 3),
                }
            except ProviderHTTPError as exc:
                last_exc = exc
                if not exc.retryable:
                    break                       # 4xx 立即失败
            except ProviderNetworkError as exc:
                last_exc = exc
            except ProviderConfigError:
                raise
            except Exception as exc:            # 预期外错误不重试,直接上抛给调用点包装
                last_exc = exc
                break

            if attempt < max(1, self.cfg.max_retries):
                delay = min(self.cfg.backoff_base * (2 ** (attempt - 1)), MAX_BACKOFF)
                time.sleep(delay + random.uniform(0, delay * 0.3))

        raise last_exc or ProviderError("调用失败但没有捕获到具体异常")

    def _run(self, fn: Callable[[], str], label: str, *, floor: float | None = None) -> dict:
        """统一出口:执行 -> 解析 -> 过滤低置信度候选,任何失败都落成 error dict。"""
        try:
            text, meta = self._with_retry(fn, label)
        except ProviderError as exc:
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "raw": getattr(exc, "body", "") or "",
                "_meta": {"provider": self.name, "model": self.cfg.model,
                          "capability": label},
            }
        except Exception as exc:
            return {
                "error": f"未预期的错误 {type(exc).__name__}: {exc}",
                "raw": "",
                "_meta": {"provider": self.name, "model": self.cfg.model,
                          "capability": label},
            }

        result = parse_json_response(text)
        result = filter_low_confidence(
            result, floor if floor is not None else self.cfg.confidence_floor
        )
        result.setdefault("_meta", {}).update(meta)
        return result


# ---------------------------------------------------------------------------
# OpenAI 兼容(DeepSeek 走这一套)
# ---------------------------------------------------------------------------

_IMAGE_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif",
}
_VIDEO_MIME = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
}

MEDIA_TYPES = ("image", "audio", "video")


def _is_url(value: str) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _read_b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def sanitize_payload(node: Any, keep: int = 50) -> Any:
    """递归截断超长字符串,用于打印实际发送的 payload 结构。

    base64 / data URI 只保留前 ``keep`` 个字符,其余长文本(prompt)同样截断并标注原长度。
    """
    if isinstance(node, dict):
        return {k: sanitize_payload(v, keep) for k, v in node.items()}
    if isinstance(node, list):
        return [sanitize_payload(v, keep) for v in node]
    if isinstance(node, str) and len(node) > keep * 2:
        return f"{node[:keep]}...<截断,原长 {len(node)} 字符>"
    return node


class OpenAICompatibleProvider(BaseProvider):
    """/chat/completions 接口。

    支持四种 content type:
      text / image_url / video_url / input_audio
    图像与视频用 ``{"url": ...}``,可以是公网 URL 也可以是 base64 data URI;
    音频按 OpenAI 的 input_audio 约定传裸 base64 + format。

    每次调用后在实例上留下调试痕迹:``last_payload`` / ``last_raw_text`` /
    ``last_usage`` / ``last_latency`` / ``last_error_body``。
    """

    name = "openai_compatible"
    supports = ("vision", "audio", "video", "text", "omni")
    default_base_url = "https://api.openai.com/v1"

    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        self.last_payload: dict | None = None
        self.last_raw_text: str | None = None
        self.last_usage: dict | None = None
        self.last_latency: float | None = None
        self.last_error_body: str | None = None
        # 实际被服务端接受的鉴权头形式,首次成功调用后确定
        self.effective_auth_header: str | None = None
        self._auth_fallback_announced = False

    @property
    def endpoint(self) -> str:
        base = (self.cfg.base_url or self.default_base_url).rstrip("/")
        return f"{base}/chat/completions"

    def _headers(self, style: str | None = None) -> dict:
        """按鉴权头形式构造 header。

        官方 curl 示例用 ``api-key: $KEY``,Python SDK 示例走标准 OpenAI Bearer,
        两者不一致,因此做成可切换,并在 401 时自动试另一种。
        """
        style = (style or self.effective_auth_header
                 or self.cfg.auth_header or "bearer").lower()
        if style in ("api-key", "api_key", "apikey"):
            return {"api-key": self.api_key}
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _other_auth_style(style: str) -> str:
        return "bearer" if style in ("api-key", "api_key", "apikey") else "api-key"

    def _extract_text(self, body: str) -> str:
        data = json.loads(body)
        self.last_usage = data.get("usage")
        choice = data["choices"][0]
        reason = choice.get("finish_reason")
        if reason != "stop" and (reason is not None or self.cfg.media_format == "mimo"):
            raise ProviderError(f"模型输出未正常完成: {reason}")
        message = choice["message"]
        content = message.get("content")
        # 推理内容不是最终判读；绝不能把它当作最终 JSON。
        if not content:
            raise ProviderError("模型未返回最终正文 content")
        return content

    # -- content part 构造 -------------------------------------------------

    def _media_part(self, media_type: str, source: str, *, b64: bool = True) -> dict:
        """把一份媒体构造成一个 content part。

        ``source`` 是本地路径或公网 URL。本地路径且 ``b64=True`` 时转 data URI /
        裸 base64;``b64=False`` 且传的是本地路径会直接报错,避免静默发出无效请求。
        """
        if media_type not in MEDIA_TYPES:
            raise ProviderConfigError(
                f"media_type 必须是 {MEDIA_TYPES} 之一,收到 '{media_type}'"
            )

        def url_part(url: str) -> dict:
            part = {"type": f"{media_type}_url", f"{media_type}_url": {"url": url}}
            if self.cfg.media_param_placement == "content":
                part[f"{media_type}_url"].update(self.cfg.media_params())
            elif self.cfg.media_param_placement == "content_part" and media_type == "video":
                part.update(self.cfg.media_params())
            return part

        if _is_url(source):
            if media_type == "image":
                return url_part(source)
            if media_type == "video":
                return url_part(source)
            if self.cfg.media_format == "mimo":
                return {"type": "input_audio", "input_audio": {"data": source}}
            raise ProviderConfigError(
                "input_audio 不接受 URL,音频必须以 base64 内联;请先把音频下载到本地。"
            )

        if not os.path.isfile(source):
            raise ProviderConfigError(f"媒体文件不存在: {source}")
        if not b64:
            raise ProviderConfigError(
                f"已禁用 base64 内联,但 {source} 是本地文件。"
                "请改传公网 URL,或把 base64 支持打开。"
            )

        ext = os.path.splitext(source)[1].lower()
        if media_type == "image":
            mime = _IMAGE_MIME.get(ext, "image/jpeg")
            return url_part(f"data:{mime};base64,{_read_b64(source)}")
        if media_type == "video":
            mime = _VIDEO_MIME.get(ext, "video/mp4")
            return url_part(f"data:{mime};base64,{_read_b64(source)}")
        if self.cfg.media_format == "mimo":
            mime = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
                    ".m4a": "audio/mp4", ".ogg": "audio/ogg"}.get(ext)
            if not mime:
                raise ProviderConfigError("MiMo 音频请使用 WAV/MP3/FLAC/M4A/OGG，其他格式需先转换")
            return {"type": "input_audio", "input_audio": {
                "data": f"data:{mime};base64,{_read_b64(source)}"}}
        # audio:OpenAI 的 input_audio 约定是裸 base64 + format,不带 data: 前缀
        fmt = ext.lstrip(".") or "wav"
        return {
            "type": "input_audio",
            "input_audio": {"data": _read_b64(source), "format": fmt},
        }

    @staticmethod
    def _image_part_from_b64(b64: str, mime: str = "image/jpeg") -> dict:
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

    # -- 请求 --------------------------------------------------------------

    def _build_payload(self, system: str, content: Any, *, json_mode: bool = True) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        payload: dict = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.cfg.extra_body())
        return payload

    def _chat(self, payload: dict) -> str:
        self.last_payload = payload
        self.last_raw_text = None
        self.last_usage = None
        self.last_error_body = None
        started = time.time()

        style = (self.effective_auth_header or self.cfg.auth_header or "bearer").lower()
        try:
            body = _post_json(self.endpoint, payload,
                              self._headers(style), self.cfg.timeout)
        except ProviderHTTPError as exc:
            if exc.status != 401:
                self.last_error_body = exc.body
                self.last_latency = round(time.time() - started, 3)
                raise
            # 401:换另一种鉴权头再试一次。这一次不占用 max_retries 配额。
            alt = self._other_auth_style(style)
            print(f"[auth] {style} 形式返回 401,改用 {alt} 形式重试一次 ...")
            try:
                body = _post_json(self.endpoint, payload,
                                  self._headers(alt), self.cfg.timeout)
            except ProviderHTTPError as exc2:
                self.last_error_body = exc2.body
                self.last_latency = round(time.time() - started, 3)
                if exc2.status == 401:
                    print(f"[auth] {alt} 形式同样返回 401 —— 两种形式都不通,"
                          f"多半是 key 本身的问题,不是 header 形式。")
                raise
            self.effective_auth_header = alt
            print(f"[auth] 生效的鉴权头形式: {alt}"
                  f"(可把 config.yaml 的 auth_header 改成 {alt} 省掉这次试探)")
            self._auth_fallback_announced = True
        else:
            if self.effective_auth_header is None:
                self.effective_auth_header = style
                print(f"[auth] 生效的鉴权头形式: {style}")

        self.last_latency = round(time.time() - started, 3)
        text = self._extract_text(body)
        self.last_raw_text = text
        return text

    # -- 能力方法 ----------------------------------------------------------

    def call_text(self, prompt: Prompt | str) -> dict:
        system, user = self._prompt_parts(prompt)
        return self._run(lambda: self._chat(self._build_payload(system, user)), "text")

    def call_vision(self, frames: Sequence[dict], prompt: Prompt | str) -> dict:
        system, user = self._prompt_parts(prompt)
        content: list[dict] = [{"type": "text", "text": user}]
        for frame in frames:
            b64 = frame.get("image_b64")
            if b64:
                content.append(self._image_part_from_b64(b64))
        return self._run(lambda: self._chat(self._build_payload(system, content)), "vision")

    def call_audio(self, audio_path: str, prompt: Prompt | str) -> dict:
        return self.call_omni(audio_path, "audio", prompt, capability="audio")

    def call_omni(
        self,
        media: str | Sequence[str] | Sequence[dict],
        media_type: str,
        prompt: Prompt | str,
        *,
        capability: str = "omni",
        allow_base64: bool = True,
    ) -> dict:
        """全模态调用。

        ``media`` 可以是:
          * 单个本地路径或公网 URL;
          * 路径 / URL 的列表;
          * ``select_frames`` 输出的帧字典列表(含 ``image_b64``,仅 media_type=image)。
        """
        system, user = self._prompt_parts(prompt)
        content: list[dict] = [{"type": "text", "text": user}]

        try:
            items: Sequence = [media] if isinstance(media, (str, bytes)) else list(media)
            for item in items:
                if isinstance(item, dict):
                    if media_type != "image":
                        raise ProviderConfigError(
                            f"帧字典只能用于 media_type=image,当前是 {media_type}"
                        )
                    b64 = item.get("image_b64")
                    if b64:
                        content.append(self._image_part_from_b64(b64))
                    continue
                content.append(self._media_part(media_type, str(item), b64=allow_base64))
        except ProviderConfigError as exc:
            return {
                "error": f"ProviderConfigError: {exc}",
                "raw": "",
                "_meta": {"provider": self.name, "model": self.cfg.model,
                          "capability": capability},
            }

        return self._run(lambda: self._chat(self._build_payload(system, content)),
                         capability)


class DeepSeekProvider(OpenAICompatibleProvider):
    name = "deepseek"
    supports = ("vision", "text")
    default_base_url = "https://api.deepseek.com/v1"


# ---------------------------------------------------------------------------
# Gemini(音频)
# ---------------------------------------------------------------------------

_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


class GeminiProvider(BaseProvider):
    """generateContent 接口,音频以 inline_data 内联。"""

    name = "gemini"
    supports = ("audio", "text")
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _endpoint(self) -> str:
        base = (self.cfg.base_url or self.default_base_url).rstrip("/")
        return f"{base}/models/{self.cfg.model}:generateContent"

    def _headers(self) -> dict:
        return {"x-goog-api-key": self.api_key}

    @staticmethod
    def _extract_text(body: str) -> str:
        data = json.loads(body)
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)

    def _payload(self, system: str, parts: list[dict]) -> dict:
        payload: dict = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    def call_audio(self, audio_path: str, prompt: Prompt | str) -> dict:
        system, user = self._prompt_parts(prompt)

        if not audio_path or not os.path.isfile(audio_path):
            return {
                "error": f"音频文件不存在: {audio_path}",
                "raw": "",
                "_meta": {"provider": self.name, "capability": "audio"},
            }

        ext = os.path.splitext(audio_path)[1].lower()
        mime = _AUDIO_MIME.get(ext, "audio/wav")
        with open(audio_path, "rb") as fh:
            audio_b64 = base64.b64encode(fh.read()).decode("ascii")

        parts = [
            {"text": user},
            {"inline_data": {"mime_type": mime, "data": audio_b64}},
        ]

        def do() -> str:
            return self._extract_text(
                _post_json(self._endpoint(), self._payload(system, parts),
                           self._headers(), self.cfg.timeout)
            )

        return self._run(do, "audio")

    def call_text(self, prompt: Prompt | str) -> dict:
        system, user = self._prompt_parts(prompt)

        def do() -> str:
            return self._extract_text(
                _post_json(self._endpoint(), self._payload(system, [{"text": user}]),
                           self._headers(), self.cfg.timeout)
            )

        return self._run(do, "text")


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

class MockProvider(BaseProvider):
    """返回符合各自 schema 的假数据,不发任何请求。

    取值直接从知识库 YAML 现场取,因此改了 knowledge/*.yaml 之后 mock 输出会跟着变,
    可以用来验证编码表与 prompt、解析链路是否仍然自洽。
    """

    name = "mock"
    supports = ("vision", "audio", "text")

    # -- 内部工具 ----------------------------------------------------------

    def _knowledge(self, species: str) -> dict:
        return load_knowledge(species, self.cfg.knowledge_dir or None)

    @staticmethod
    def _species_of(prompt: Prompt | str, default: str = "dog") -> str:
        if isinstance(prompt, Prompt) and prompt.species:
            return prompt.species
        text = prompt if isinstance(prompt, str) else prompt.as_text()
        for sp in ("dog", "cat"):
            if f"物种: {sp}" in text:
                return sp
        return default

    @staticmethod
    def _pick(values: Sequence, rotation: int):
        """按帧序号轮换取值,保证跨帧有变化且结果可复现。"""
        if not values:
            return "不可观测"
        return values[rotation % len(values)]

    def _signals_for(self, knowledge: dict, rotation: int) -> tuple[dict, dict]:
        """按知识库编码表生成一帧的 signals 与对应的置信度标注。"""
        signals: dict[str, Any] = {}
        confidence: dict[str, Any] = {}

        for channel, block in (knowledge.get("signals") or {}).items():
            if not isinstance(block, dict):
                continue
            dims = block.get("dimensions")
            if isinstance(dims, dict):
                sub_val, sub_conf = {}, {}
                for dname, dblock in dims.items():
                    value = self._pick(dblock.get("values") or [], rotation)
                    sub_val[dname] = value
                    sub_conf[dname] = (dblock.get("value_confidence") or {}).get(
                        value, dblock.get("confidence", "中")
                    )
                signals[channel] = sub_val
                confidence[channel] = sub_conf
            else:
                value = self._pick(block.get("values") or [], rotation)
                signals[channel] = value
                confidence[channel] = (block.get("value_confidence") or {}).get(
                    value, block.get("confidence", "中")
                )
        return signals, confidence

    # -- 能力方法 ----------------------------------------------------------

    def call_vision(self, frames: Sequence[dict], prompt: Prompt | str) -> dict:
        species = self._species_of(prompt)
        knowledge = self._knowledge(species)
        channels = list((knowledge.get("signals") or {}).keys())

        per_frame = []
        for i, frame in enumerate(frames):
            signals, confidence = self._signals_for(knowledge, i)
            # 制造一个稳定的不可观测通道,验证下游对缺失的处理
            uncertain = [channels[-1]] if channels and i % 3 == 2 else []
            for ch in uncertain:
                signals[ch] = "不可观测"
                confidence[ch] = "不可观测"
            per_frame.append({
                "t": frame.get("t", round(i * 0.5, 2)),
                "role": frame.get("role", "peak"),
                "signals": signals,
                "signal_confidence": confidence,
                "uncertain": uncertain,
            })

        temporal_change = []
        for prev, curr in zip(per_frame, per_frame[1:]):
            for ch in channels:
                before, after = prev["signals"].get(ch), curr["signals"].get(ch)
                if before == after or isinstance(before, dict):
                    continue
                temporal_change.append({
                    "from_t": prev["t"],
                    "to_t": curr["t"],
                    "channel": ch,
                    "from": before,
                    "to": after,
                    "direction": ("升级", "缓解", "横向变化")[len(temporal_change) % 3],
                    "basis": "[mock] 未做真实的阶梯比对",
                })

        overall = {}
        for ch in channels:
            values = [f["signals"].get(ch) for f in per_frame
                      if f["signals"].get(ch) not in (None, "不可观测")]
            flat = [v for v in values if not isinstance(v, dict)]
            overall[ch] = {
                "dominant": max(set(flat), key=flat.count) if flat else "不可观测",
                "confidence": "中",
                "frames_observed": len(values),
            }

        return {
            "per_frame": per_frame,
            "temporal_change": temporal_change[:6],
            "overall_signals": overall,
            "_meta": {"provider": self.name, "model": "mock", "capability": "vision",
                      "attempts": 1, "species": species, "frames": len(frames)},
        }

    def call_audio(self, audio_path: str, prompt: Prompt | str) -> dict:
        species = self._species_of(prompt)
        knowledge = self._knowledge(species)
        patterns = knowledge.get("acoustic") or []

        if not audio_path:
            return {
                "events": [],
                "background_noise": "[mock] 未提供音频文件",
                "audible_events_found": False,
                "_meta": {"provider": self.name, "model": "mock", "capability": "audio",
                          "attempts": 1, "species": species},
            }

        events = []
        for i, pattern in enumerate(patterns[:2]):
            f0 = pattern.get("f0_range") or [None, None]
            lo, hi = (f0 + [None, None])[:2]
            start = round(0.8 + i * 2.4, 2)
            events.append({
                "t_start": start,
                "t_end": round(start + 0.6, 2),
                "acoustic": {
                    "duration_s": 0.6,
                    "f0_contour": "先升后降" if i % 2 == 0 else "平稳",
                    "f0_range_hz": [lo, hi],
                    "syllable_count": 1 + i,
                    "tremor": bool(i % 2),
                    "energy_envelope": "陡起缓落" if i % 2 == 0 else "持续平稳",
                    "noise_component": bool(i % 2),
                },
                "matched_pattern": pattern.get("pattern"),
                "match_confidence": pattern.get("confidence", "中"),
                "match_basis": "[mock] 未做真实的声学分析",
                "alternatives": [],
            })

        return {
            "events": events,
            "background_noise": "[mock] 背景安静",
            "audible_events_found": bool(events),
            "_meta": {"provider": self.name, "model": "mock", "capability": "audio",
                      "attempts": 1, "species": species,
                      "audio_exists": os.path.isfile(audio_path)},
        }

    def call_text(self, prompt: Prompt | str) -> dict:
        species = self._species_of(prompt)
        knowledge = self._knowledge(species)

        ladder = knowledge.get("escalation_ladder") or []
        rules = knowledge.get("conflict_rules") or []
        channels = list((knowledge.get("signals") or {}).keys())
        level3 = next((lv for lv in ladder if int(lv.get("level", 0)) == 3), {})
        level3_signals = (level3.get("signals") or ["未知信号"])[:2]

        timeline = [
            {
                "t_range": [0.0, 1.2],
                "visual": f"[mock] {channels[0] if channels else '通道'} 处于基线取值",
                "audio": None,
                "motion": "[mock] 运动强度低",
            },
            {
                "t_range": [1.2, 3.0],
                "visual": f"[mock] {', '.join(level3_signals)} 出现",
                "audio": "[mock] 出现一次匹配到知识库模式的发声",
                "motion": "[mock] 运动强度升高",
            },
        ]

        conflicts = []
        if rules:
            rule = rules[0]
            conflicts.append({
                "signals": rule.get("conflicting", []),
                "rule": rule.get("resolution", ""),
                "resolution": f"[mock] 按 priority={rule.get('priority', '')} 裁决",
                "resolved": True,
            })

        candidates = [
            {
                "label": "[mock] 高置信候选",
                "confidence": 0.68,
                "evidence": [
                    {
                        "channel": channels[0] if channels else "unknown",
                        "observation": level3_signals[0],
                        "t": "1.2-3.0s",
                        "knowledge_basis": level3.get("notes", "")[:60],
                        "signal_confidence": level3.get("confidence", "中"),
                    }
                ],
            },
            {
                "label": "[mock] 中置信候选",
                "confidence": 0.52,
                "evidence": [
                    {
                        "channel": channels[-1] if channels else "unknown",
                        "observation": "[mock] 位移信号密度升高",
                        "t": "2.0-3.0s",
                        "knowledge_basis": "[mock] density_rule",
                        "signal_confidence": "中",
                    }
                ],
            },
            {
                "label": "[mock] 应被过滤掉的低置信候选",
                "confidence": 0.31,
                "evidence": [],
            },
        ]

        result = {
            "timeline": timeline,
            "conflicts": conflicts,
            "candidates": candidates,
            "insufficient_evidence": False,
            "insufficient_reason": None,
            "translation": "[mock] 这里是一句拟人化表达。",
            "suggestion": "[mock] 先停止当前互动并让它有路可退,不要制止它的警告信号。",
            "_meta": {"provider": self.name, "model": "mock", "capability": "text",
                      "attempts": 1, "species": species},
        }
        return filter_low_confidence(result, self.cfg.confidence_floor)

    def call_omni(
        self,
        media: str | Sequence[str] | Sequence[dict],
        media_type: str,
        prompt: Prompt | str,
        *,
        capability: str = "omni",
        allow_base64: bool = True,
    ) -> dict:
        """全模态 mock。

        真实 provider 的返回形态由 prompt 决定,mock 必须跟着 ``prompt.task`` 分派,
        否则 frames 模式下会拿到 omni 形态而不是 vision/audio 形态。
        """
        species = self._species_of(prompt)
        items = [media] if isinstance(media, (str, bytes)) else list(media)
        frames = [i for i in items if isinstance(i, dict)] or _fixture_frames(3)

        task = prompt.task if isinstance(prompt, Prompt) else ""
        if task == "vision":
            return self.call_vision(frames, prompt)
        if task == "audio":
            path = next((str(i) for i in items if isinstance(i, str)), "mock.wav")
            return self.call_audio(path, prompt)

        visual = self.call_vision(frames, prompt)
        audio = self.call_audio("mock.wav", prompt)
        fused = self.call_text(prompt)

        fused["visual_signals"] = visual.get("per_frame", [])
        fused["audio_events"] = audio.get("events", [])
        fused["_meta"] = {"provider": self.name, "model": "mock", "capability": capability,
                          "attempts": 1, "species": species,
                          "media_type": media_type, "media_count": len(items)}
        return fused


# ---------------------------------------------------------------------------
# 注册表:新增 provider 只需在这里登记,模型名始终来自 config
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "deepseek": DeepSeekProvider,
    "openai": OpenAICompatibleProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "gemini": GeminiProvider,
    "mock": MockProvider,
}


# ---------------------------------------------------------------------------
# base64 data URI 支持探测
#
# 官方示例只给了公网 URL,没有 base64 示例,所以内联能力必须实测。
# 探测分两步,用于区分"鉴权/模型有问题"和"确实不接受 base64":
#   step 0  纯文本 ping   -> 失败说明是 key / model / 端点的问题
#   step 1  base64 媒体   -> step 0 通过后仍失败,才是 base64 不被接受
# 任何一步失败都打印服务端原始返回,不静默降级。
# ---------------------------------------------------------------------------

# 1x1 透明 PNG,用于最小成本地测试图像 base64 通道
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg=="
)


def probe_base64_support(
    provider: OpenAICompatibleProvider,
    media_path: str | None = None,
    media_type: str = "image",
    verbose: bool = True,
) -> dict:
    """实测该端点是否接受 base64 data URI 内联的媒体。

    返回 ``{"text_ok", "base64_ok", "media_type", "detail", "http_status"}``。
    ``base64_ok`` 为 False 时 ``detail`` 里是服务端的原始报错。
    """
    def say(*args):
        if verbose:
            print(*args)

    report = {"text_ok": False, "base64_ok": False, "media_type": media_type,
              "detail": "", "http_status": None}

    # --- step 0:纯文本 ping ---
    say(f"[探测 0/2] 纯文本 ping  model={provider.cfg.model}  {provider.endpoint}")
    try:
        text = provider._chat(provider._build_payload(
            "", "回复两个字符:OK", json_mode=False))
        report["text_ok"] = True
        say(f"          -> 通过,返回 {text[:60]!r}  用量={provider.last_usage}")
    except ProviderConfigError as exc:
        report["detail"] = f"配置错误: {exc}"
        say(f"          -> 失败(配置): {exc}")
        return report
    except ProviderHTTPError as exc:
        report["detail"] = f"HTTP {exc.status}: {exc.body[:800]}"
        report["http_status"] = exc.status
        say(f"          -> 失败 HTTP {exc.status}")
        say(f"             服务端返回: {exc.body[:800]}")
        say("          纯文本都不通,先解决 key / model / base_url,再谈 base64。")
        return report
    except ProviderNetworkError as exc:
        report["detail"] = f"网络错误: {exc}"
        say(f"          -> 失败(网络): {exc}")
        return report

    # --- step 1:base64 媒体 ---
    if media_type == "image" and not media_path:
        source_desc = "内置 1x1 PNG"
        part = provider._image_part_from_b64(_TINY_PNG_B64, "image/png")
    else:
        if not media_path or not os.path.isfile(media_path):
            report["detail"] = f"media_type={media_type} 需要一个真实的本地文件,收到: {media_path}"
            say(f"[探测 1/2] 跳过:{report['detail']}")
            return report
        source_desc = f"{media_path} ({os.path.getsize(media_path) / 1024:.0f} KB)"
        try:
            part = provider._media_part(media_type, media_path, b64=True)
        except ProviderConfigError as exc:
            report["detail"] = str(exc)
            say(f"[探测 1/2] 构造失败: {exc}")
            return report

    say(f"[探测 1/2] base64 内联 {media_type}:{source_desc}")
    content = [{"type": "text", "text": "用两个字符回复:OK"}, part]
    try:
        text = provider._chat(provider._build_payload("", content, json_mode=False))
        report["base64_ok"] = True
        report["detail"] = text[:200]
        say(f"          -> 接受 base64,返回 {text[:60]!r}  "
            f"用量={provider.last_usage}  耗时={provider.last_latency}s")
    except ProviderHTTPError as exc:
        report["detail"] = f"HTTP {exc.status}: {exc.body[:1500]}"
        report["http_status"] = exc.status
        say(f"          -> 拒绝 HTTP {exc.status}")
        say(f"             服务端返回: {exc.body[:1500]}")
        say("")
        say("=" * 70)
        say(f"[!] 该端点不接受 base64 内联的 {media_type}。纯文本已确认可用,"
            f"所以问题出在媒体的传递方式,不是 key 或 model。")
        say("    可选处置:")
        say("      1. 把媒体上传到可公网访问的对象存储,改传 URL;")
        say("      2. 确认服务端是否要求先走文件上传接口拿 file_id;")
        say("      3. 若只是 video 不支持而 image 支持,改用 --mode frames。")
        say("=" * 70)
    except ProviderNetworkError as exc:
        report["detail"] = f"网络错误: {exc}"
        say(f"          -> 失败(网络): {exc}")
        say("[!] 媒体体积较大时可能是超时,可调大 config.yaml 的 timeout 再试。")

    return report


# ---------------------------------------------------------------------------
# 自测:mock 模式下跑完整三段链路
# ---------------------------------------------------------------------------

def _fixture_frames(count: int = 4) -> list[dict]:
    roles = ["baseline", "pre", "peak", "post"]
    return [
        {"t": round(i * 0.7, 2), "role": roles[i % len(roles)], "image_b64": "ZmFrZQ=="}
        for i in range(count)
    ]


_FIXTURE_MOTION = {
    "duration": 6.0,
    "motion_peaks": [1.2, 2.6, 4.1],
    "motion_intensity": "中等",
    "motion_pattern": "间歇",
    "periodicity_hz": None,
}


# ---------------------------------------------------------------------------
# 真实调用入口
#
# --mode native   整段视频直接交给模型,由它按 fps 自行采样
# --mode frames   走 video.select_frames 的智能抽帧 + 分段调用(视觉/声学/融合)
# 两种模式跑同一段素材,便于对比采样策略。
# ---------------------------------------------------------------------------

def _sep(title: str, char: str = "=") -> None:
    print(f"\n{char * 78}\n{title}\n{char * 78}")


def _report_call(provider: BaseProvider, label: str, prompt: Prompt,
                 result: dict, elapsed: float) -> None:
    """按要求打印:payload 结构(脱敏)、完整原始输出、token 用量、耗时。"""
    _sep(f"【{label}】", "-")

    payload = getattr(provider, "last_payload", None)
    print("① 实际发送的 payload 结构(base64 只留前 50 字符):")
    if payload is None:
        print("   (mock provider 未产生 payload)")
    else:
        print(json.dumps(sanitize_payload(payload), ensure_ascii=False, indent=2))

    print(f"\n② prompt 体积: system={len(prompt.system)} 字符  "
          f"user={len(prompt.user)} 字符  合计={len(prompt)} 字符")

    raw = getattr(provider, "last_raw_text", None)
    print("\n③ 完整原始输出(未做任何后处理):")
    if raw is None:
        print("   (无原始文本,mock provider 直接返回 dict)")
    else:
        print(raw)

    usage = getattr(provider, "last_usage", None)
    print("\n④ token 用量:")
    if usage:
        print(f"   input  = {usage.get('prompt_tokens')}")
        print(f"   output = {usage.get('completion_tokens')}")
        print(f"   total  = {usage.get('total_tokens')}")
        extra = {k: v for k, v in usage.items()
                 if k not in ("prompt_tokens", "completion_tokens", "total_tokens")}
        if extra:
            print(f"   其他   = {json.dumps(extra, ensure_ascii=False)}")
    else:
        print("   (响应中无 usage 字段)")

    net = getattr(provider, "last_latency", None)
    print(f"\n⑤ 耗时: 端到端 {elapsed:.2f}s"
          + (f"  其中网络往返 {net:.2f}s" if net else ""))

    if "error" in result:
        print(f"\n[x] 本次调用失败: {result['error']}")
        if result.get("raw"):
            print(f"    服务端返回: {str(result['raw'])[:1000]}")


def _postprocess_report(result: dict, knowledge: dict) -> None:
    """打印后处理结果:候选过滤、护栏检查、证据充分性。"""
    if "error" in result:
        return
    _sep("后处理与护栏", "-")
    cands = result.get("candidates") or []
    print(f"候选数: {len(cands)}"
          + (f"  (已过滤掉 {result['_dropped_candidates']} 个低于阈值的)"
             if result.get("_dropped_candidates") else ""))
    for c in cands:
        if isinstance(c, dict):
            print(f"  - {c.get('label')}  confidence={c.get('confidence')}  "
                  f"evidence={len(c.get('evidence') or [])} 条")
    print(f"insufficient_evidence = {result.get('insufficient_evidence')}")
    if result.get("insufficient_reason"):
        print(f"  原因: {result['insufficient_reason']}")
    print(f"translation: {result.get('translation')}")
    print(f"suggestion : {result.get('suggestion')}")
    hits = check_suggestion_safety(result.get("suggestion", ""), knowledge)
    print(f"suggestion 护栏: {'拦截 -> ' + '; '.join(hits) if hits else '通过'}")
    print(f"visual_signals 条数: {len(result.get('visual_signals') or [])}    "
          f"audio_events 条数: {len(result.get('audio_events') or [])}    "
          f"timeline 段数: {len(result.get('timeline') or [])}")


def run_native(provider, video: str, species: str, context: str,
               knowledge: dict, max_frames: int) -> dict:
    """整段视频直接送给模型,采样由 fps 参数控制。"""
    from .video import motion_features

    motion = motion_features(video)
    fps = provider.cfg.fps
    hint = (f"整段视频文件({os.path.getsize(video) / 1024 / 1024:.1f} MB),"
            f"由你按 fps={fps} 自行采样;音轨包含在视频内。"
            f"时间戳以视频起点为 0。")
    prompt = prompts_mod.omni_prompt(species, context, knowledge, motion, media_hint=hint)

    t0 = time.time()
    result = provider.call_omni(video, "video", prompt)
    elapsed = time.time() - t0

    _report_call(provider, "native 模式:整段视频单次调用", prompt, result, elapsed)
    _postprocess_report(result, knowledge)
    return result


def run_frames(provider, video: str, species: str, context: str,
               knowledge: dict, max_frames: int) -> dict:
    """智能抽帧 + 分段调用:视觉 -> 声学 -> 融合,三次都走同一个 omni provider。"""
    from .video import analysis_window, extract_audio, motion_features, select_frames

    print(f"[*] 智能抽帧中 (max_frames={max_frames}) ...")
    frames = select_frames(video, max_frames=max_frames)
    motion = motion_features(video)
    start, dur = analysis_window(video)
    wav = os.path.join(PROJECT_ROOT, "data", "omni_run_16k.wav")
    audio_path = extract_audio(video, wav, start=start, duration=dur)
    frames_meta = [{k: v for k, v in f.items() if k != "image_b64"} for f in frames]
    print(f"    抽出 {len(frames)} 帧,时间戳 {[f['t'] for f in frames]}")
    print(f"    分析窗口 {start}s 起 {dur}s,音轨 {audio_path or '无'}")

    # 1) 视觉
    v_prompt = prompts_mod.vision_prompt(species, context, frames_meta, motion, knowledge)
    t0 = time.time()
    v_result = provider.call_omni(frames, "image", v_prompt, capability="vision")
    _report_call(provider, "frames 模式 ①:抽帧视觉编码", v_prompt, v_result, time.time() - t0)

    # 2) 声学
    if audio_path:
        a_prompt = prompts_mod.audio_prompt(species, context, knowledge)
        t0 = time.time()
        a_result = provider.call_omni(audio_path, "audio", a_prompt, capability="audio")
        _report_call(provider, "frames 模式 ②:音轨声学编码", a_prompt, a_result,
                     time.time() - t0)
    else:
        a_result = {"events": [], "audible_events_found": False,
                    "background_noise": "视频无音轨"}
        print("\n[!] 视频无音轨,跳过声学段。")

    # 3) 融合
    f_prompt = prompts_mod.fusion_prompt(v_result, a_result, motion, context,
                                         knowledge, species=species)
    t0 = time.time()
    f_result = provider.call_text(f_prompt)
    _report_call(provider, "frames 模式 ③:融合判读", f_prompt, f_result, time.time() - t0)

    # 对齐 omni 的输出形态,便于两种模式直接对比
    if "error" not in f_result:
        f_result.setdefault("visual_signals", v_result.get("per_frame", []))
        f_result.setdefault("audio_events", a_result.get("events", []))
    _postprocess_report(f_result, knowledge)
    return f_result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="petread 全模态调用入口 / base64 支持探测")
    parser.add_argument("--video", help="待判读的视频路径")
    parser.add_argument("--species", default="dog", choices=["dog", "cat"])
    parser.add_argument("--context", default="", help="拍摄情境描述")
    parser.add_argument("--mode", default="native", choices=["native", "frames"],
                        help="native=整段视频交给模型采样;frames=智能抽帧+分段调用")
    parser.add_argument("--max-frames", type=int, default=8,
                        help="frames 模式的抽帧上限")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--role", default="omni", help="使用 config.providers 下的哪个角色")
    parser.add_argument("--probe", action="store_true",
                        help="只做 base64 支持探测,不跑判读")
    parser.add_argument("--mock", action="store_true", help="强制走 mock,不发真实请求")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.mock:
        config["mock"] = True

    try:
        providers = build_providers(config)
    except ProviderConfigError as exc:
        print(f"[x] 配置错误: {exc}")
        return 2

    if args.role not in providers:
        print(f"[x] config.providers 中没有 '{args.role}';当前可用: {sorted(providers)}")
        return 2
    provider = providers[args.role]

    # 真实模式下先确认 key 在环境里,给出可操作的报错而不是栈回溯
    if not config.get("mock"):
        env = provider.cfg.api_key_env
        if not os.environ.get(env, "").strip():
            print(f"[x] 环境变量 {env} 未设置,无法发起真实请求。")
            print(f"    export {env}='...' 之后重跑;或加 --mock 走假数据链路。")
            return 3

    knowledge = load_knowledge(args.species)

    _sep(f"petread  role={args.role}  provider={provider.name}  "
         f"model={provider.cfg.model}  mock={bool(config.get('mock'))}")
    print(f"物种={args.species}  模式={args.mode}  "
          f"thinking={provider.cfg.thinking}  fps={provider.cfg.fps}  "
          f"media_resolution={provider.cfg.media_resolution or '(未设)'}")
    print(f"extra_body={json.dumps(provider.cfg.extra_body(), ensure_ascii=False)}"
          f"  media_params={json.dumps(provider.cfg.media_params(), ensure_ascii=False)}")

    # --- base64 探测 ---
    if args.probe:
        if config.get("mock"):
            print("[x] mock 模式下探测没有意义,去掉 --mock 再跑。")
            return 2
        _sep("base64 data URI 支持探测")
        img = probe_base64_support(provider, None, "image")
        result = {"image": img}
        if args.video and os.path.isfile(args.video):
            print()
            result["video"] = probe_base64_support(provider, args.video, "video")
        _sep("探测结论")
        for k, v in result.items():
            print(f"  {k:<6} 纯文本={v['text_ok']}  base64={v['base64_ok']}"
                  + (f"  {v['detail'][:200]}" if not v["base64_ok"] else ""))
        return 0 if all(v["base64_ok"] for v in result.values()) else 1

    if not args.video:
        print("[x] 需要 --video 指定视频,或用 --probe 只做探测。")
        return 2
    if not os.path.isfile(args.video):
        print(f"[x] 视频不存在: {args.video}")
        return 2

    runner = run_native if args.mode == "native" else run_frames
    result = runner(provider, args.video, args.species, args.context,
                    knowledge, args.max_frames)
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())

