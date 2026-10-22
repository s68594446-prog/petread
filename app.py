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
"""

from __future__ import annotations

import asyncio
import csv
import hmac
import io
import json
import os
import time
import uuid
from datetime import datetime
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 启动时自动加载项目根目录的 .env,这样 MIMO_API_KEY 等不必每次在命令行传。
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
from pipeline.prompts import load_knowledge

INDEX_HTML = os.path.join(PROJECT_ROOT, "index.html")
# 上传目录跟随 store.DATA_DIR,由 DATA_DIR 环境变量统一控制。
UPLOAD_DIR = os.path.join(store.DATA_DIR, "uploads")

MAX_VIDEO_BYTES = 20 * 1024 * 1024      # 20MB,与前端提示保持一致
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_AUDIO_BYTES = 15 * 1024 * 1024
CHUNK = 1024 * 1024

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".caf", ".amr"}

app = FastAPI(title="petread", docs_url="/docs")

_STATE: dict[str, Any] = {}


@app.on_event("startup")
def _startup() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    store.init_db()

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
    prov = (_STATE.get("providers") or {}).get("omni")
    if prov is None:
        raise HTTPException(500, "config.yaml 中没有 omni provider")
    return prov


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
    return {
        "ok": True,
        "mock": bool((_STATE.get("config") or {}).get("mock")),
        "provider": getattr(prov, "name", None),
        "model": getattr(getattr(prov, "cfg", None), "model", None),
        "key_present": bool(os.environ.get(
            getattr(getattr(prov, "cfg", None), "api_key_env", "") or "", "").strip()),
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
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise

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
# 一次判读要 20-30s。旧实现把整段判读挂在 POST /analyze 这一条 SSE 连接上,
# 手机切到别的 App / 锁屏时浏览器会挂起页面、掐断连接,结果就发不回来——而且
# 存记录的代码在连接后面,连库都写不成(模型照跑照花 token,结果全丢)。
#
# 现在改成:POST 只建任务、在后台独立把判读跑完并写库,立刻返回 job_id;
# 前端用 GET /analyze/stream/{job_id} 订阅进度。切屏、断网、甚至整页重载,
# 都能拿 job_id 重新连上取回结果——后台任务不依赖任何一条连接是否还开着。
#
# 任务存在进程内存(_JOBS)。当前是单进程 uvicorn,够用;将来多副本部署时
# 需换成 Redis 之类的外部存储。
# ---------------------------------------------------------------------------

JOB_TTL_S = 1800          # 任务完成后在内存里保留 30 分钟,供重连 / 重载取回
_JOBS: dict[str, dict] = {}


def _sweep_jobs() -> None:
    now = time.time()
    stale = [jid for jid, j in _JOBS.items()
             if j.get("finished_at") and now - j["finished_at"] > JOB_TTL_S]
    for jid in stale:
        _JOBS.pop(jid, None)


def _new_job() -> tuple[str, dict]:
    _sweep_jobs()
    jid = uuid.uuid4().hex
    job = {"events": [], "done": False, "created": time.time(), "finished_at": None}
    _JOBS[jid] = job
    return jid, job


def _job_emit(job: dict, event: str, payload: dict) -> None:
    """追加一个事件。所有订阅者(含之后才连上的)都按下标从头补看到它。"""
    job["events"].append((event, payload))


def _job_finish(job: dict, event: str, payload: dict) -> None:
    _job_emit(job, event, payload)
    job["done"] = True
    job["finished_at"] = time.time()


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

    def report(stage: str, pct: int, message: str) -> None:
        loop.call_soon_threadsafe(
            _job_emit, job, "progress",
            {"stage": stage, "percent": pct, "message": message})

    def work() -> dict:
        return analyze_mod.analyze(
            media_path, species, context, mode, provider, knowledge,
            max_frames=max_frames, text_description=text, progress=report)

    _job_emit(job, "progress", {
        "stage": "upload", "percent": 3,
        "message": ("已收到描述,开始处理…" if mode == "text"
                    else f"已接收 {size / 1024 / 1024:.1f}MB,开始处理…")})

    # finally 保证成功 / 模型报错 / 写库失败三条出口都会清理媒体原件。
    try:
        try:
            out = await loop.run_in_executor(None, work)
        except Exception as exc:                   # 判读层不该抛,兜住以免任务卡在未完成
            _job_finish(job, "error", {"message": f"{type(exc).__name__}: {exc}"})
            return

        result = out.get("result") or {}
        if "error" in result:
            _job_finish(job, "error", {
                "message": result["error"], "raw": str(result.get("raw", ""))[:2000]})
            return

        label, confidence = store.summarize_result(result)
        # text 模式的观察描述并进 context 一起存,否则校准回顾时看不到用户输入了什么
        stored_context = context
        if mode == "text":
            desc = f"文字描述:{text}"
            stored_context = f"{context}｜{desc}" if context else desc
        try:
            record_id = store.insert_record(
                species=species, mode=mode, context=stored_context,
                media_path=media_path,
                result=result, predicted_label=label, confidence=confidence,
                tokens_in=out.get("tokens_in"), tokens_out=out.get("tokens_out"),
                latency_s=out.get("latency_s"), provider=out.get("provider"),
                model=out.get("model"), client_id=client_id)
        except Exception as exc:
            _job_finish(job, "error",
                        {"message": f"写库失败 {type(exc).__name__}: {exc}"})
            return

        _job_emit(job, "progress",
                  {"stage": "persist", "percent": 100, "message": "完成"})
        _job_finish(job, "done", {
            "record_id": record_id,
            "result": result,
            "meta": {
                "mode": mode,
                "species": species,
                "tokens_in": out.get("tokens_in"),
                "tokens_out": out.get("tokens_out"),
                "latency_s": out.get("latency_s"),
                "provider": out.get("provider"),
                "model": out.get("model"),
                "motion_features": out.get("motion_features"),
                "predicted_label": label,
                "confidence": confidence,
            },
        })
    finally:
        # 判读已结束(无论结果如何),媒体原件的使命完成。
        _cleanup_media(media_path)


@app.post("/analyze")
async def analyze_endpoint(
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
    client_id = (client_id or "").strip()[:64]

    if species not in ("dog", "cat"):
        raise HTTPException(422, "species 必须是 dog 或 cat")
    if mode not in analyze_mod.MODES:
        raise HTTPException(422, f"mode 必须是 {analyze_mod.MODES} 之一")

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
        if not context:
            raise HTTPException(422, "context 必填:请描述拍摄场景")
        media_path, size = await _save_upload(file, mode)

    provider = _provider()
    knowledge = load_knowledge(species)

    job_id, job = _new_job()
    # create_task 后台跑,不 await;请求这就返回。任务生命周期与本请求无关。
    asyncio.create_task(_run_analysis_job(
        job, media_path=media_path, size=size, species=species, context=context,
        mode=mode, provider=provider, knowledge=knowledge, max_frames=max_frames,
        text=text, client_id=client_id))
    return JSONResponse({"job_id": job_id})


@app.get("/analyze/stream/{job_id}")
async def analyze_stream(job_id: str) -> StreamingResponse:
    """订阅某个任务的进度(SSE)。从头补发已缓冲的事件,因此断线重连 / 页面重载
    都能拿同一个 job_id 把进度和最终结果重新取回。EventSource(GET)即可用。"""

    async def gen():
        job = _JOBS.get(job_id)
        if job is None:
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
                 "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# 历史与反馈
# ---------------------------------------------------------------------------

@app.get("/history")
def history(client_id: str = "", limit: int = 30, offset: int = 0) -> JSONResponse:
    limit = max(1, min(int(limit), 200))
    # 只返回该设备自己的记录;client_id 为空则空列表(不泄漏他人/老数据)
    rows = store.list_records((client_id or "").strip()[:64],
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
def export_csv(key: str = "") -> Response:
    """导出**全部** records + feedback 为 CSV,供跨设备校准回顾。

    不按 client_id 过滤(要看到所有人的数据),但用 EXPORT_KEY 口令保护:
    URL 带 ?key=<EXPORT_KEY>,与环境变量 EXPORT_KEY 比对。
    环境变量未设置时直接禁用导出(不留后门)。
    """
    expected = os.environ.get("EXPORT_KEY", "").strip()
    if not expected:
        raise HTTPException(503, "导出未启用:服务端未配置 EXPORT_KEY 环境变量")
    if not hmac.compare_digest((key or "").strip(), expected):
        raise HTTPException(403, "导出口令错误:URL 需带正确的 ?key=<EXPORT_KEY>")

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
            line.append("" if v is None else v)
        writer.writerow(line)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="petread_export_{stamp}.csv"',
        },
    )


@app.get("/record/{record_id}")
def record_detail(record_id: int) -> JSONResponse:
    rec = store.get_record(record_id)
    if not rec:
        raise HTTPException(404, f"记录 {record_id} 不存在")
    rec["feedback"] = store.list_feedback(record_id)
    rec.pop("media_path", None)          # 不把服务器路径暴露给前端
    return JSONResponse(rec)


@app.post("/feedback/{record_id}")
async def feedback(record_id: int, payload: dict) -> JSONResponse:
    actual_label = str(payload.get("actual_label") or "").strip()
    user_note = str(payload.get("user_note") or "").strip()
    if not actual_label and not user_note:
        raise HTTPException(422, "actual_label 与 user_note 至少填一个")
    try:
        fid = store.insert_feedback(record_id, actual_label, user_note)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return JSONResponse({"ok": True, "feedback_id": fid, "record_id": record_id})


@app.get("/profiles")
def profiles() -> JSONResponse:
    return JSONResponse({"items": store.list_profiles()})


@app.post("/profiles")
async def create_profile(payload: dict) -> JSONResponse:
    pet_name = str(payload.get("pet_name") or "").strip()
    species = str(payload.get("species") or "").strip().lower()
    if not pet_name:
        raise HTTPException(422, "pet_name 必填")
    if species not in ("dog", "cat"):
        raise HTTPException(422, "species 必须是 dog 或 cat")
    pid = store.upsert_profile(
        pet_name, species, str(payload.get("calibration_notes") or "").strip())
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
