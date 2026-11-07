"""petread 服务端。

启动:
    uvicorn app:app --host 0.0.0.0 --port 8000
或:
    python app.py

接口
----
GET  /                      前端页面
POST /analyze               multipart 上传，返回 job_id
GET  /analyze/stream/{job_id} SSE 推送进度与最终结果
GET  /history               历史记录
GET  /record/{id}           单条记录详情
POST /feedback/{record_id}  用户自评
GET  /profiles, POST /profiles   宠物档案

POST 创建后台任务并返回 job_id；GET /analyze/stream/{job_id}
通过 EventSource 订阅进度，支持同一进程内的断线重连。
任务与 HTTP 连接独立；服务器签发凭证，按归属隔离任务、历史、反馈和档案。
"""

from __future__ import annotations

import asyncio
import csv
import hmac
import io
import json
import os
import re
import time
import uuid
import hashlib
from contextvars import ContextVar
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 启动时自动加载项目根目录的 .env,密钥不必每次在命令行传。
# 必须在导入 pipeline(会读 config / 环境变量)之前完成。
# 已存在于真实环境中的变量优先,不被 .env 覆盖(override=False)。
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)
except ImportError:
    pass  # 没装 python-dotenv 时退回到纯环境变量,不报错

from pipeline import analyze as analyze_mod
from pipeline import providers as providers_mod
from pipeline import store
from pipeline import runtime
from pipeline.http_limits import BodyLimitMiddleware
from pipeline.prompts import load_knowledge

INDEX_HTML = os.path.join(PROJECT_ROOT, "index.html")
RELEASE_ID = "20261107-visual-v2.1-free"
# 上传目录跟随 store.DATA_DIR,由 DATA_DIR 环境变量统一控制。
UPLOAD_DIR = os.path.join(store.DATA_DIR, "uploads")

from pipeline.media_limits import VIDEO_MAX_BYTES, VIDEO_MAX_SECONDS, VIDEO_MAX_FPS

MAX_VIDEO_BYTES = VIDEO_MAX_BYTES
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_AUDIO_BYTES = 15 * 1024 * 1024
CHUNK = 1024 * 1024


class MediaValidationError(ValueError):
    """A local media rejection; its details must not expose internal paths."""

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".caf", ".amr"}

@asynccontextmanager
async def lifespan(application):
    _startup()
    try:
        yield
    finally:
        if _TASKS:
            await asyncio.gather(*_TASKS, return_exceptions=True)
        lock = _STATE.pop('worker_lock', None)
        if lock:
            lock.close()


app = FastAPI(title="PetRead", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(BodyLimitMiddleware, limit=20 * 1024 * 1024 + 128 * 1024)
_IDENTITY: ContextVar[str | None] = ContextVar("petread_identity", default=None)

_STATE: dict[str, Any] = {}


def _startup() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    _STATE['worker_lock'] = runtime.acquire_worker_lock()
    store.init_db()
    runtime.init(recover=True)
    for jid, path in runtime.interrupted_uploads():
        # Only clean upload paths our own ledger registered, never arbitrary
        # files, historic assets, or untracked scripts/data directories.
        if (os.path.dirname(os.path.abspath(path)) == os.path.abspath(UPLOAD_DIR)
                and re.fullmatch(r'[a-f0-9]{32}\.[a-z0-9]+', os.path.basename(path))):
            _cleanup_media(path)
            if not os.path.exists(path):
                runtime.clear_upload_reference(jid)

    config = providers_mod.load_config()
    _STATE["config"] = config
    _STATE["providers"] = providers_mod.build_providers(config)
    _STATE["role"] = "omni" if "omni" in _STATE["providers"] else None

    provider = _STATE["providers"].get("omni")
    if provider is not None and not config.get("mock"):
        env = provider.cfg.api_key_env
        if not os.environ.get(env, "").strip():
            print(f"[!] 环境变量 {env} 未设置。/analyze 会返回 500,"
                  f"请 export {env}=... 后重启,或把 config.yaml 的 mock 改成 true。")
    print(f"[petread] provider={getattr(provider, 'name', None)} "
          f"model={getattr(getattr(provider, 'cfg', None), 'model', None)} "
          f"mock={bool(config.get('mock'))}")


def _provider():
    # Providers retain mutable response/usage traces. Never share an instance
    # across jobs, or one user's raw response and tokens can enter another job.
    config = _STATE.get("config")
    prov = providers_mod.build_providers(config).get("omni") if config else None
    if prov is None:
        raise HTTPException(503, "分析服务配置未就绪")
    if not config.get("mock") and not os.environ.get(prov.cfg.api_key_env, "").strip():
        raise HTTPException(503, "分析服务尚未配置 API 密钥")
    return prov


def _client_id(value: str) -> str:
    value = (value or "").strip()
    identity = _IDENTITY.get()
    if identity is not None:
        if value and value != identity:
            raise HTTPException(403, "设备凭证不匹配")
        return identity
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", value):
        raise HTTPException(422, "需要有效的设备 ID,请刷新页面后重试")
    return value


@app.middleware("http")
async def security(request: Request, call_next):
    # Reject cross-origin browser mutations, even when a bearer header exists.
    origin = request.headers.get("origin")
    expected = os.environ.get("PETREAD_PUBLIC_ORIGIN") or str(request.base_url).rstrip("/")
    mutation = request.method not in {"GET", "HEAD", "OPTIONS"}
    if mutation and origin and origin.rstrip("/") != expected.rstrip("/"):
        return JSONResponse({"detail": "不允许跨站请求"}, status_code=403)
    if mutation:
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"detail": "无效请求长度"}, status_code=400)
        if length > max(MAX_VIDEO_BYTES, MAX_IMAGE_BYTES, MAX_AUDIO_BYTES) + 128 * 1024:
            return JSONResponse({"detail": "请求超过上传上限"}, status_code=413)
    public = request.url.path in {"/", "/healthz", "/session", "/privacy", "/support"} or request.url.path.startswith("/assets/")
    bearer = request.headers.get("authorization", "")
    token = bearer[7:] if bearer.startswith("Bearer ") else request.cookies.get("petread_session", "")
    identity = runtime.authenticate(token) if token else None
    if not public and request.url.path != "/export" and identity is None:
        return JSONResponse({"detail": "请建立设备会话后重试"}, status_code=401)
    if mutation and not bearer and request.headers.get("x-petread-client") not in {"web", "ios"}:
        return JSONResponse({"detail": "需要同源客户端请求头"}, status_code=403)
    request.state.identity = identity
    scope = _IDENTITY.set(identity["client_id"] if identity else None)
    try:
        response = await call_next(request)
    finally:
        _IDENTITY.reset(scope)
    response.headers.update({
        "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
        "X-Frame-Options": "DENY", "Cache-Control": "no-store",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    })
    if request.method in {"GET", "HEAD"} and request.url.path.startswith("/assets/") and response.status_code == 200:
        response.headers["Cache-Control"] = "public, max-age=3600"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response


@app.post("/session")
def session(request: Request) -> JSONResponse:
    identity = request.state.identity
    token = None
    if identity is None:
        peer = request.client.host if request.client else "unknown"
        # Trust the actual ASGI peer, not arbitrary X-Forwarded-For headers.
        try:
            runtime.rate_limit("session:" + hashlib.sha256(peer.encode()).hexdigest(), 20, 3600)
        except runtime.AdmissionError as exc:
            raise HTTPException(429, str(exc))
        token, identity = runtime.create_session()
    response = JSONResponse({**identity, "required_consent_version": runtime.CONSENT_VERSION,
                             **({"token": token} if token else {})})
    if token:
        response.set_cookie("petread_session", token, max_age=runtime.SESSION_SECONDS,
                            httponly=True, secure=request.url.scheme == "https", samesite="strict")
    return response


@app.post("/consent")
def set_consent(payload: dict) -> dict:
    if type(payload.get("accepted")) is not bool or payload.get("version") != runtime.CONSENT_VERSION:
        raise HTTPException(422, "请确认当前版本的数据说明")
    runtime.consent(_client_id(""), payload["accepted"])
    return {"ok": True}


@app.get("/me/export")
def export_device() -> Response:
    cid = _client_id("")
    records = store.list_records(cid, limit=1000000, offset=0)
    for rec in records:
        rec.pop("media_path", None)
        rec["feedback"] = store.list_feedback(rec["id"])
    return Response(json.dumps({"records": records, "profiles": store.list_profiles(client_id=cid),
                                "attempts": runtime.attempts_for_device(cid)}, ensure_ascii=False),
                    media_type="application/json", headers={"Content-Disposition": 'attachment; filename="petread-data.json"'})


@app.delete("/me")
def delete_device() -> JSONResponse:
    cid = _client_id("")
    for jid, path in runtime.interrupted_uploads(cid):
        if (os.path.dirname(os.path.abspath(path)) == os.path.abspath(UPLOAD_DIR)
                and re.fullmatch(r'[a-f0-9]{32}\.[a-z0-9]+', os.path.basename(path))):
            _cleanup_media(path)
            if not os.path.lexists(path):
                runtime.clear_upload_reference(jid)
    try:
        counts = runtime.delete_device(cid)
    except runtime.AdmissionError as exc:
        raise HTTPException(409, str(exc))
    for jid in [jid for jid, job in _JOBS.items() if job["client_id"] == cid]:
        _JOBS.pop(jid, None)
    response = JSONResponse({"deleted": counts})
    response.delete_cookie("petread_session")
    return response


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = runtime.get_job(job_id, _client_id(""))
    if not job:
        raise HTTPException(404, "任务不存在")
    return {"job_id": job_id, "done": job["done"], "events": job["events"]}


@app.get("/assets/{name}")
def asset(name: str) -> FileResponse:
    if name not in {"app.js", "app.css", "mark.svg", "mark-yellow.png"}:
        raise HTTPException(404)
    return FileResponse(os.path.join(PROJECT_ROOT, "web", name))


@app.get("/privacy")
def privacy() -> FileResponse:
    return FileResponse(os.path.join(PROJECT_ROOT, "web", "privacy.html"))


@app.get("/support")
def support() -> JSONResponse:
    return JSONResponse({"support_email": os.environ.get("PETREAD_SUPPORT_EMAIL") or None,
                         "release_status": "pre-release", "note": "紧急健康问题请联系兽医；PetRead 不提供诊断或急救服务。"})


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    if not os.path.isfile(INDEX_HTML):
        raise HTTPException(500, "index.html 缺失")
    return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")


@app.get("/healthz")
def healthz() -> dict:
    prov = (_STATE.get("providers") or {}).get("omni")
    key_present = bool(os.environ.get(getattr(getattr(prov, "cfg", None), "api_key_env", "") or "", "").strip())
    mock = bool((_STATE.get("config") or {}).get("mock"))
    return {
        "ok": True,
        "release_id": RELEASE_ID,
        "mock": mock,
        "analysis_ready": bool(prov) and (mock or key_present),
        "provider": getattr(prov, "name", None),
        "model": getattr(getattr(prov, "cfg", None), "model", None),
        "key_present": key_present,
        "media_limits": {"video_bytes": MAX_VIDEO_BYTES, "video_seconds": VIDEO_MAX_SECONDS,
                         "video_fps": VIDEO_MAX_FPS, "video_long_edge": 1920, "video_short_edge": 1080},
    }


# ---------------------------------------------------------------------------
# 上传校验
# ---------------------------------------------------------------------------

async def _save_upload(upload: UploadFile, mode: str) -> tuple[str, int]:
    """边写边计字节数,超限立即中止并删除半截文件。

    前端也会拦一次,但不能只依赖前端:直接打接口、改前端 JS、旧页面缓存
    都能绕过去。
    """
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if mode == "quick":
        allowed, limit, kind = IMAGE_EXTS, MAX_IMAGE_BYTES, "图片"
    elif mode == "audio":
        allowed, limit, kind = AUDIO_EXTS, MAX_AUDIO_BYTES, "音频"
    else:
        allowed, limit, kind = VIDEO_EXTS, MAX_VIDEO_BYTES, "视频"

    if ext not in allowed:
        raise HTTPException(
            415, f"{kind}格式不支持: {ext or '(无扩展名)'};支持 {sorted(allowed)}")

    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    total = 0
    try:
        with open(path, "wb") as fh:
            while True:
                chunk = await upload.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise HTTPException(
                        413,
                        f"{kind}超过 {limit // 1024 // 1024}MB 上限"
                        f"(已收到 {total / 1024 / 1024:.1f}MB)。请裁剪后再上传。",
                    )
                fh.write(chunk)
    except BaseException:
        if os.path.exists(path):
            os.remove(path)
        raise
    finally:
        await upload.close()

    if total == 0:
        os.remove(path)
        raise HTTPException(400, "上传的文件是空的")
    return path, total


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


# ---------------------------------------------------------------------------
# 后台任务:把"跑判读"和"看结果"解耦
#
# 历史版本把整段判读挂在 POST /analyze 这一条 SSE 连接上,
# 手机切到别的 App / 锁屏时浏览器会挂起页面、掐断连接,结果就发不回来——而且
# 存记录的代码在连接后面,连库都写不成(模型照跑照花 token,结果全丢)。
#
# 现在改成:POST 只建任务、在后台独立把判读跑完并写库,立刻返回 job_id;
# 前端用 GET /analyze/stream/{job_id} 订阅进度。切屏、断网、甚至整页重载,
# 都能拿 job_id 重新连上取回结果——后台任务不依赖任何一条连接是否还开着。
#
# SQLite 保存任务与事件，_JOBS 只是当前进程活跃缓存。重启后的未完成任务
# 标为 interrupted_or_outcome_unknown，不自动重新请求模型。单 worker 锁
# 保证恢复过程不会错误地终止另一个进程的任务；多副本需另行设计队列/租约。
# ---------------------------------------------------------------------------

JOB_TTL_S = 1800          # 任务完成后在内存里保留 30 分钟,供重连 / 重载取回
MAX_ACTIVE_JOBS = max(1, int(os.environ.get("PETREAD_MAX_ACTIVE_JOBS", "2")))
MAX_RETAINED_JOBS = max(MAX_ACTIVE_JOBS, 200)
_JOBS: dict[str, dict] = {}
_TASKS: set[asyncio.Task] = set()


def _sweep_jobs() -> None:
    now = time.time()
    stale = [jid for jid, j in _JOBS.items()
             if j.get("finished_at") and now - j["finished_at"] > JOB_TTL_S]
    for jid in stale:
        _JOBS.pop(jid, None)


def _new_job(client_id: str) -> tuple[str, dict]:
    _sweep_jobs()
    active = [j for j in _JOBS.values() if not j["done"]]
    if len(active) >= MAX_ACTIVE_JOBS or any(j.get("client_id") == client_id for j in active):
        raise HTTPException(429, "当前已有分析任务运行,请稍后再试", headers={"Retry-After": "10"})
    while len(_JOBS) >= MAX_RETAINED_JOBS:
        finished = [jid for jid, j in _JOBS.items() if j["done"]]
        if not finished:
            raise HTTPException(429, "任务队列已满,请稍后再试")
        _JOBS.pop(min(finished, key=lambda jid: _JOBS[jid]["finished_at"]))
    jid = uuid.uuid4().hex
    job = {"id": jid, "events": [], "done": False, "created": time.time(), "finished_at": None,
           "client_id": client_id}
    _JOBS[jid] = job
    return jid, job


def _job_emit(job: dict, event: str, payload: dict) -> None:
    """追加一个事件。所有订阅者(含之后才连上的)都按下标从头补看到它。"""
    job["events"].append((event, payload))
    runtime.save_job(job)


def _job_finish(job: dict, event: str, payload: dict) -> None:
    job["done"] = True
    job["finished_at"] = time.time()
    _job_emit(job, event, payload)


def _cleanup_media(media_path: str) -> None:
    """判读结束后删掉上传的媒体原件。

    留下来的是库里的判读结果(result JSON)与用户反馈——校准只需要这些。
    原件既占磁盘又涉及隐私,判完就没有保留价值了。
    records.media_path 仍保留原值作为历史痕迹,只是文件本身不在了。

    清理是尽力而为:任何删除失败都只打日志,绝不能影响已经跑完的判读结果。
    """
    if not media_path:
        return                      # text 模式没有媒体文件
    # frames 模式会在同目录额外生成 <媒体名>_16k.wav 音轨(见 analyze._run_frames)
    for path in (media_path, os.path.splitext(media_path)[0] + "_16k.wav"):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass                    # 本来就没有(非 frames 模式的音轨)
        except OSError as exc:
            print(f"[petread] 清理上传文件失败(不影响判读): {path} {exc}")


async def _run_analysis_job(
    job: dict, *, media_path: str, size: int, species: str, context: str,
    mode: str, provider, knowledge: dict, max_frames: int, text: str, client_id: str,
) -> None:
    """后台判读:跑模型 -> 写库 -> 落最终事件。全程不依赖任何 HTTP 连接。"""
    loop = asyncio.get_running_loop()
    job_started = time.perf_counter()
    timings: dict[str, float | None] = {"preprocessing_s": None, "pipeline_s": None, "persistence_s": None}
    is_openai = getattr(getattr(provider, "cfg", None), "type", None) == "openai_native"
    if is_openai and job.get("durable"):
        provider.attempt_observer = runtime.attempt_observer(job["id"])
    execution_mode = "native" if is_openai and mode == "frames" else mode
    execution_path = (
        "openai_sampled_video" if is_openai and execution_mode == "native" else execution_mode
    )

    def report(stage: str, pct: int, message: str) -> None:
        if stage == "model" and pct < 90:
            message = "正在调用模型分析；耗时取决于素材与服务响应，请勿重复提交。"
        loop.call_soon_threadsafe(
            _job_emit, job, "progress",
            {"stage": stage, "percent": pct, "message": message})

    def work() -> dict:
        before = time.perf_counter()
        # Validate local media before the frozen analyzer can invoke motion
        # decoding. OpenAI video uses the adapter's bounded frames/audio path.
        if is_openai and media_path:
            try:
                provider.preflight(media_path, execution_mode)
            except ValueError as exc:
                raise MediaValidationError from exc
        timings["preprocessing_s"] = round(time.perf_counter() - before, 3)
        from pipeline.video import preprocessing_session
        with preprocessing_session():
            return analyze_mod.analyze(
                media_path, species, context, execution_mode, provider, knowledge,
                max_frames=max_frames, text_description=text, progress=report)

    _job_emit(job, "progress", {
        "stage": "upload", "percent": 3,
        "message": ("已收到描述,开始处理…" if mode == "text"
                    else f"已接收 {size / 1024 / 1024:.1f}MB,开始处理…")})
    if execution_mode != mode:
        _job_emit(job, "progress", {
            "stage": "prepare", "percent": 4,
            "message": "OpenAI 视频使用抽帧与音轨联合判读，已切换到兼容的视频链路",
        })

    # finally 保证成功 / 模型报错 / 写库失败三条出口都会清理媒体原件。
    try:
        try:
            before_pipeline = time.perf_counter()
            out = await loop.run_in_executor(None, work)
            timings["pipeline_s"] = round(time.perf_counter() - before_pipeline, 3)
        except MediaValidationError:
            message = ("视频未通过校验：请使用 15 秒内、10 MB 内、1080p / 30fps 以内的有效视频。未调用模型。"
                       if execution_mode == "native" else
                       "素材未通过本地校验，请检查格式、时长与分辨率后重新选择。未调用模型。")
            _job_finish(job, "error", {"message": message, "error_type": "media_validation_error"})
            return
        except Exception as exc:                   # 判读层不该抛,兜住以免任务卡在未完成
            print(f"[petread] analysis failed: {type(exc).__name__}")
            _job_finish(job, "error", {"message": "分析暂时失败,请稍后重试"})
            return

        result = out.get("result") or {}
        if "error" in result:
            kind = result.get("_error_kind") or "runner_or_internal_error"
            _job_finish(job, "error", {
                "message": "模型未返回有效判读,请稍后重试", "error_type": kind})
            return

        result.setdefault("_meta", {}).update({
            "requested_mode": mode,
            "execution_mode": execution_mode,
            "execution_path": execution_path,
            "release_id": RELEASE_ID,
            "timings": dict(timings),
        })

        actual_model = getattr(provider, "last_model", None) or (result.get("_meta") or {}).get("model") or out.get("model")

        label, confidence = store.summarize_result(result)
        # text 模式的观察描述并进 context 一起存,否则校准回顾时看不到用户输入了什么
        stored_context = context
        if mode == "text":
            desc = f"文字描述:{text}"
            stored_context = f"{context}｜{desc}" if context else desc
        try:
            before_persistence = time.perf_counter()
            record_id = store.insert_record(
                species=species, mode=mode, context=stored_context,
                media_path="",  # deleted media paths are not user history
                result=result, predicted_label=label, confidence=confidence,
                tokens_in=out.get("tokens_in"), tokens_out=out.get("tokens_out"),
                latency_s=out.get("latency_s"), provider=out.get("provider"),
                model=actual_model, client_id=client_id)
            timings["persistence_s"] = round(time.perf_counter() - before_persistence, 3)
        except Exception as exc:
            print(f"[petread] persistence failed: {type(exc).__name__}")
            _job_finish(job, "error", {"message": "结果保存失败,请稍后重试"})
            return

        _job_emit(job, "progress",
                  {"stage": "persist", "percent": 100, "message": "完成"})
        _job_finish(job, "done", {
            "record_id": record_id,
            "result": result,
            "meta": {
                "mode": mode,
                "execution_mode": execution_mode,
                "execution_path": execution_path,
                "species": species,
                "tokens_in": out.get("tokens_in"),
                "tokens_out": out.get("tokens_out"),
                "latency_s": out.get("latency_s"),
                "provider": out.get("provider"),
                "model": actual_model,
                "motion_features": out.get("motion_features"),
                "predicted_label": label,
                "confidence": confidence,
                "release_id": RELEASE_ID,
                "timings": {**timings, "server_total_s": round(time.perf_counter() - job_started, 3)},
            },
        })
    finally:
        # 判读已结束(无论结果如何),媒体原件的使命完成。
        _cleanup_media(media_path)
        if job.get('durable') and (not media_path or not os.path.exists(media_path)):
            runtime.clear_upload_reference(job['id'])


@app.post("/analyze")
async def analyze_endpoint(
    request: Request,
    file: UploadFile | None = File(None),   # text 模式无文件
    species: str = Form(...),
    context: str = Form(""),          # 空值走下面的中文校验,而不是框架的 missing
    mode: str = Form("native"),
    max_frames: int = Form(8),
    text: str = Form(""),             # text 模式的观察描述
    client_id: str = Form(""),        # 匿名设备 ID,用于隔离历史
) -> JSONResponse:
    """建任务并在后台启动判读,立刻返回 job_id。进度用 GET /analyze/stream 订阅。"""
    species = (species or "").strip().lower()
    mode = (mode or "native").strip().lower()
    context = (context or "").strip()
    text = (text or "").strip()
    client_id = _client_id(client_id)
    if (request.state.identity or {}).get("consent_version") != runtime.CONSENT_VERSION:
        raise HTTPException(403, "请先明确同意将本次素材和描述发送给 OpenAI 分析")
    request_key = request.headers.get("idempotency-key", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,80}", request_key):
        raise HTTPException(422, "需要有效的 Idempotency-Key，避免网络重试重复计费")
    existing = runtime.find_request(client_id, request_key)
    if existing:
        return JSONResponse({"job_id": existing["id"], "reused": True})

    if species not in ("dog", "cat"):
        raise HTTPException(422, "species 必须是 dog 或 cat")
    if mode not in analyze_mod.MODES:
        raise HTTPException(422, f"mode 必须是 {analyze_mod.MODES} 之一")
    if not 1 <= max_frames <= 16:
        raise HTTPException(422, "max_frames 必须在 1 到 16 之间")
    if len(context) > 2000:
        raise HTTPException(413, "场景描述请限制在 2000 字以内")

    provider = _provider()
    knowledge = load_knowledge(species)

    media_path = ""
    size = 0
    if mode == "text":
        if not text:
            raise HTTPException(422, "text 模式需要填写你观察到的描述")
        if len(text) > 4000:
            raise HTTPException(413, "描述过长,请精简到 4000 字以内")
    else:
        if file is None or not (file.filename or "").strip():
            raise HTTPException(422, "缺少上传文件")
        media_path, size = await _save_upload(file, mode)

    try:
        job_id, job = _new_job(client_id)
        job['upload_path'] = media_path or None
        runtime.create_job(job, request_key)
        job["durable"] = True
    except runtime.AdmissionError as exc:
        if "job_id" in locals():
            _JOBS.pop(job_id, None)
        _cleanup_media(media_path)
        raise HTTPException(429, str(exc), headers={"Retry-After": "60"})
    except BaseException:
        if "job_id" in locals():
            _JOBS.pop(job_id, None)
        _cleanup_media(media_path)
        raise
    # create_task 后台跑,不 await;请求这就返回。任务生命周期与本请求无关。
    task = asyncio.create_task(_run_analysis_job(
        job, media_path=media_path, size=size, species=species, context=context,
        mode=mode, provider=provider, knowledge=knowledge, max_frames=max_frames,
        text=text, client_id=client_id))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return JSONResponse({"job_id": job_id})


@app.get("/analyze/stream/{job_id}")
async def analyze_stream(job_id: str, client_id: str = "") -> StreamingResponse:
    """订阅某个任务的进度(SSE)。从头补发已缓冲的事件,因此断线重连 / 页面重载
    都能拿同一个 job_id 把进度和最终结果重新取回。EventSource(GET)即可用。"""

    client_id = _client_id(client_id)
    _sweep_jobs()
    job = _JOBS.get(job_id) or runtime.get_job(job_id, client_id)

    async def gen():
        if job is None or job.get("client_id") != client_id:
            # 任务不存在或已过期:发一个错误事件让前端停下,别让 EventSource 空转重连
            yield _sse("error", {"message": "任务不存在或已过期,请重新提交。"})
            return

        idx = 0
        while True:
            # 补发这条连接还没看到的事件(重连时会从 0 开始重放,天然幂等)
            while idx < len(job["events"]):
                event, payload = job["events"][idx]
                idx += 1
                yield _sse(event, payload)
                if event in ("done", "error"):
                    return
            if job["done"]:
                return
            # 空闲时每秒一次心跳,带上从任务创建起算的已用秒数(重连也连续)
            await asyncio.sleep(1.0)
            yield _sse("heartbeat",
                       {"elapsed_s": round(time.time() - job["created"], 1)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive", "Referrer-Policy": "no-referrer"},
    )


# ---------------------------------------------------------------------------
# 历史与反馈
# ---------------------------------------------------------------------------

@app.get("/history")
def history(client_id: str = "", limit: int = 30, offset: int = 0) -> JSONResponse:
    limit = max(1, min(int(limit), 200))
    # 只返回该设备自己的记录;client_id 为空则空列表(不泄漏他人/老数据)
    rows = store.list_records(_client_id(client_id),
                              limit=limit, offset=max(0, int(offset)))
    items = []
    for r in rows:
        result = r.get("result") or {}
        items.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "species": r["species"],
            "mode": r["mode"],
            "context": r["context"],
            "predicted_label": r["predicted_label"],
            "confidence": r["confidence"],
            "insufficient_evidence": bool(result.get("insufficient_evidence")),
            "translation": result.get("translation"),
            "tokens_in": r["tokens_in"],
            "tokens_out": r["tokens_out"],
            "latency_s": r["latency_s"],
            "model": r["model"],
            "feedback_count": r.get("feedback_count", 0),
            "last_actual_label": r.get("last_actual_label"),
        })
    return JSONResponse({"items": items, "limit": limit, "offset": offset})


# 导出列顺序与中文表头(给校准回顾用,直接能丢进表格软件)
_EXPORT_COLUMNS = [
    ("id", "记录ID"),
    ("created_at", "时间"),
    ("client_id", "设备ID"),
    ("species", "物种"),
    ("mode", "模式"),
    ("context", "场景"),
    ("predicted_label", "模型判断"),
    ("confidence", "置信度"),
    ("insufficient_evidence", "是否信息不足"),
    ("translation", "模型翻译"),
    ("actual_label", "用户反馈-实际"),
    ("user_note", "用户反馈-备注"),
    ("feedback_count", "反馈条数"),
    ("tokens_in", "输入token"),
    ("tokens_out", "输出token"),
    ("latency_s", "耗时秒"),
    ("model", "模型"),
]


@app.get("/export")
def export_csv(x_export_key: str = Header("")) -> Response:
    """导出**全部** records + feedback 为 CSV,供跨设备校准回顾。

    不按 client_id 过滤(要看到所有人的数据),但用 EXPORT_KEY 口令保护:
    请求头 X-Export-Key 与环境变量 EXPORT_KEY 比对,不在 URL 暴露口令。
    环境变量未设置时直接禁用导出(不留后门)。
    """
    expected = os.environ.get("EXPORT_KEY", "").strip()
    if not expected:
        raise HTTPException(503, "导出未启用:服务端未配置 EXPORT_KEY 环境变量")
    if not hmac.compare_digest((x_export_key or "").strip(), expected):
        raise HTTPException(403, "导出口令错误")

    rows = store.iter_export_rows()
    buf = io.StringIO()
    buf.write("﻿")          # UTF-8 BOM,让 Excel 正确识别中文
    writer = csv.writer(buf)
    writer.writerow([zh for _, zh in _EXPORT_COLUMNS])
    for r in rows:
        line = []
        for key, _ in _EXPORT_COLUMNS:
            v = r.get(key)
            if isinstance(v, bool):
                v = "是" if v else "否"
            if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")):
                v = "'" + v
            line.append("" if v is None else v)
        writer.writerow(line)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="petread_export_{stamp}.csv"',
            "Cache-Control": "no-store",
        },
    )


@app.get("/record/{record_id}")
def record_detail(record_id: int, client_id: str = "") -> JSONResponse:
    rec = store.get_record(record_id, client_id=_client_id(client_id))
    if not rec:
        raise HTTPException(404, f"记录 {record_id} 不存在")
    rec["feedback"] = store.list_feedback(record_id)
    rec.pop("media_path", None)          # 不把服务器路径暴露给前端
    return JSONResponse(rec)


@app.post("/feedback/{record_id}")
async def feedback(record_id: int, payload: dict) -> JSONResponse:
    client_id = _client_id(str(payload.get("client_id") or ""))
    actual_label = str(payload.get("actual_label") or "").strip()
    user_note = str(payload.get("user_note") or "").strip()
    if not actual_label and not user_note:
        raise HTTPException(422, "actual_label 与 user_note 至少填一个")
    if len(actual_label) > 1000 or len(user_note) > 4000:
        raise HTTPException(413, "反馈过长")
    try:
        fid = store.insert_feedback(record_id, actual_label, user_note, client_id=client_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return JSONResponse({"ok": True, "feedback_id": fid, "record_id": record_id})


@app.get("/profiles")
def profiles(client_id: str = "") -> JSONResponse:
    return JSONResponse({"items": store.list_profiles(client_id=_client_id(client_id))})


@app.post("/profiles")
async def create_profile(payload: dict) -> JSONResponse:
    client_id = _client_id(str(payload.get("client_id") or ""))
    pet_name = str(payload.get("pet_name") or "").strip()
    species = str(payload.get("species") or "").strip().lower()
    if not pet_name:
        raise HTTPException(422, "pet_name 必填")
    if species not in ("dog", "cat"):
        raise HTTPException(422, "species 必须是 dog 或 cat")
    notes = str(payload.get("calibration_notes") or "").strip()
    if len(pet_name) > 100 or len(notes) > 4000:
        raise HTTPException(413, "档案内容过长")
    pid = store.upsert_profile(
        pet_name, species, notes, client_id=client_id)
    return JSONResponse({"ok": True, "profile_id": pid})


if __name__ == "__main__":
    import uvicorn

    # 端口优先级:PORT(HF Spaces / 通用) > PETREAD_PORT(本地习惯) > 7860。
    # HF Docker Spaces 固定路由到 7860,host 必须 0.0.0.0。
    port = int(os.environ.get("PORT") or os.environ.get("PETREAD_PORT") or "7860")
    uvicorn.run(
        "app:app",
        host=os.environ.get("PETREAD_HOST", "0.0.0.0"),
        port=port,
        reload=bool(os.environ.get("PETREAD_RELOAD")),
    )
