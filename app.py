"""petread 服务端。

启动:
    uvicorn app:app --host 0.0.0.0 --port 8000
或:
    python app.py

接口
----
GET  /                      前端页面
POST /analyze               multipart 上传 + SSE 推送进度与最终结果
GET  /history               历史记录
GET  /record/{id}           单条记录详情
POST /feedback/{record_id}  用户自评
GET  /profiles, POST /profiles   宠物档案

关于 SSE:EventSource 只能发 GET,没法带 multipart。这里让 POST /analyze 直接
返回 ``text/event-stream``,前端用 fetch + ReadableStream 读。代价是没有
EventSource 自带的断线重连——一次判读 20-30 秒,断了重传即可,不值得为此
拆成"先 POST 拿 job_id 再 GET 订阅"两个接口。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

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
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")

MAX_VIDEO_BYTES = 20 * 1024 * 1024      # 20MB,与前端提示保持一致
MAX_IMAGE_BYTES = 10 * 1024 * 1024
CHUNK = 1024 * 1024

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

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


@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    species: str = Form(...),
    context: str = Form(""),          # 空值走下面的中文校验,而不是框架的 missing
    mode: str = Form("native"),
    max_frames: int = Form(8),
):
    species = (species or "").strip().lower()
    mode = (mode or "native").strip().lower()
    context = (context or "").strip()

    if species not in ("dog", "cat"):
        raise HTTPException(422, "species 必须是 dog 或 cat")
    if mode not in analyze_mod.MODES:
        raise HTTPException(422, f"mode 必须是 {analyze_mod.MODES} 之一")
    if not context:
        raise HTTPException(422, "context 必填:请描述拍摄场景")

    media_path, size = await _save_upload(file, mode)
    provider = _provider()
    knowledge = load_knowledge(species)

    async def stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def report(stage: str, pct: int, message: str) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("progress", {"stage": stage, "percent": pct, "message": message}),
            )

        def work() -> dict:
            return analyze_mod.analyze(
                media_path, species, context, mode, provider, knowledge,
                max_frames=max_frames, progress=report,
            )

        yield _sse("progress", {
            "stage": "upload", "percent": 3,
            "message": f"已接收 {size / 1024 / 1024:.1f}MB,开始处理…",
        })

        task = loop.run_in_executor(None, work)
        started = time.time()

        # 模型那一段是 20-30 秒的阻塞调用,期间没有真实进度可推。
        # 每秒发一次心跳带上已用秒数,前端才不会看起来卡死。
        while True:
            try:
                event, payload = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield _sse(event, payload)
            except asyncio.TimeoutError:
                if task.done():
                    break
                yield _sse("heartbeat", {"elapsed_s": round(time.time() - started, 1)})
                continue
            if task.done() and queue.empty():
                break

        try:
            out = await task
        except Exception as exc:                       # 判读层不该抛,兜住以免连接挂死
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
            return

        while not queue.empty():
            event, payload = queue.get_nowait()
            yield _sse(event, payload)

        result = out.get("result") or {}
        if "error" in result:
            yield _sse("error", {
                "message": result["error"],
                "raw": str(result.get("raw", ""))[:2000],
            })
            return

        label, confidence = store.summarize_result(result)
        record_id = store.insert_record(
            species=species, mode=mode, context=context, media_path=media_path,
            result=result, predicted_label=label, confidence=confidence,
            tokens_in=out.get("tokens_in"), tokens_out=out.get("tokens_out"),
            latency_s=out.get("latency_s"), provider=out.get("provider"),
            model=out.get("model"),
        )

        yield _sse("progress", {"stage": "persist", "percent": 100, "message": "完成"})
        yield _sse("done", {
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

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# 历史与反馈
# ---------------------------------------------------------------------------

@app.get("/history")
def history(limit: int = 30, offset: int = 0) -> JSONResponse:
    limit = max(1, min(int(limit), 200))
    rows = store.list_records(limit=limit, offset=max(0, int(offset)))
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

    uvicorn.run(
        "app:app",
        host=os.environ.get("PETREAD_HOST", "0.0.0.0"),
        port=int(os.environ.get("PETREAD_PORT", "8000")),
        reload=bool(os.environ.get("PETREAD_RELOAD")),
    )
