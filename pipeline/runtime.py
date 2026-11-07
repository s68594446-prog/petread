"""Single-worker runtime: hashed device credentials, durable jobs and admission limits.

This is not cross-device account authentication. Possession of a secret device
token grants access to that device's data. Old self-assigned IDs cannot be claimed.
No model calls are made here; restarted jobs are never automatically replayed.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import uuid
import fcntl

from . import store

CONSENT_VERSION = "openai-media-2026-11-06"
SESSION_SECONDS = 365 * 86400

SCHEMA = """
CREATE TABLE IF NOT EXISTS device_sessions (
 token_hash TEXT PRIMARY KEY, client_id TEXT UNIQUE NOT NULL,
 created REAL NOT NULL, expires REAL NOT NULL, consent_version TEXT
);
CREATE TABLE IF NOT EXISTS analysis_jobs (
 id TEXT PRIMARY KEY, client_id TEXT NOT NULL, request_key TEXT NOT NULL,
 created REAL NOT NULL, finished REAL, events TEXT NOT NULL DEFAULT '[]',
 done INTEGER NOT NULL DEFAULT 0, upload_path TEXT, UNIQUE(client_id, request_key)
);
CREATE INDEX IF NOT EXISTS idx_jobs_client ON analysis_jobs(client_id,created);
CREATE TABLE IF NOT EXISTS admission_counters (
 bucket TEXT PRIMARY KEY, count INTEGER NOT NULL, expires REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS production_attempts (
 id TEXT PRIMARY KEY, job_id TEXT NOT NULL, model TEXT NOT NULL,
 started REAL NOT NULL, finished REAL, status TEXT NOT NULL,
 issued INTEGER, http_body TEXT, raw TEXT, usage TEXT, latency_s REAL, reason TEXT
);
"""


class AdmissionError(Exception):
    pass


def init(*, recover: bool = False) -> None:
    with store.connect() as conn:
        conn.executescript(SCHEMA)
        if 'upload_path' not in {r['name'] for r in conn.execute('PRAGMA table_info(analysis_jobs)')}:
            conn.execute('ALTER TABLE analysis_jobs ADD COLUMN upload_path TEXT')
        if recover:
            conn.execute("UPDATE production_attempts SET status='interrupted_or_outcome_unknown' WHERE finished IS NULL")
            # A provider may have processed an in-flight request. Never call it
            # again automatically or imply that it merely timed out.
            for row in conn.execute("SELECT * FROM analysis_jobs WHERE done=0").fetchall():
                events = json.loads(row["events"])
                events.append(["error", {
                    "message": "服务已重启，上次任务结果未知。未自动重试；可能已产生模型费用。",
                    "error_type": "interrupted_or_outcome_unknown",
                }])
                conn.execute("UPDATE analysis_jobs SET done=1, finished=?, events=? WHERE id=?",
                             (time.time(), json.dumps(events, ensure_ascii=False), row["id"]))
        conn.execute("DELETE FROM admission_counters WHERE expires < ?", (time.time(),))


def create_session() -> tuple[str, dict]:
    token, client_id = secrets.token_urlsafe(32), uuid.uuid4().hex
    now = time.time()
    with store.connect() as conn:
        conn.execute("INSERT INTO device_sessions VALUES (?,?,?,?,NULL)",
                     (hashlib.sha256(token.encode()).hexdigest(), client_id, now, now + SESSION_SECONDS))
    return token, {"client_id": client_id, "consent_version": None}


def authenticate(token: str) -> dict | None:
    if not token or len(token) > 128:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    with store.connect() as conn:
        row = conn.execute("SELECT client_id,consent_version FROM device_sessions WHERE token_hash=? AND expires>?",
                           (digest, time.time())).fetchone()
        return dict(row) if row else None


def consent(client_id: str, accepted: bool) -> None:
    with store.connect() as conn:
        conn.execute("UPDATE device_sessions SET consent_version=? WHERE client_id=?",
                     (CONSENT_VERSION if accepted else None, client_id))


def rate_limit(bucket: str, maximum: int, window: int = 86400) -> None:
    now = time.time()
    key = f"{bucket}:{int(now // window)}"
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM admission_counters WHERE expires < ?", (now,))
        row = conn.execute("SELECT count FROM admission_counters WHERE bucket=?", (key,)).fetchone()
        if row and row[0] >= maximum:
            raise AdmissionError("使用额度已达上限，请稍后再试")
        conn.execute("INSERT INTO admission_counters VALUES (?,1,?) ON CONFLICT(bucket) DO UPDATE SET count=count+1",
                     (key, now + window * 2))


def get_job(job_id: str, client_id: str) -> dict | None:
    with store.connect() as conn:
        row = conn.execute("SELECT * FROM analysis_jobs WHERE id=? AND client_id=?", (job_id, client_id)).fetchone()
    return _job(row) if row else None


def find_request(client_id: str, request_key: str) -> dict | None:
    with store.connect() as conn:
        row = conn.execute("SELECT * FROM analysis_jobs WHERE client_id=? AND request_key=?", (client_id, request_key)).fetchone()
    return _job(row) if row else None


def _job(row) -> dict:
    return {"id": row["id"], "client_id": row["client_id"], "created": row["created"],
            "done": bool(row["done"]), "finished_at": row["finished"], "events": json.loads(row["events"])}


def create_job(job: dict, request_key: str) -> None:
    now = time.time()
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        active = conn.execute("SELECT client_id FROM analysis_jobs WHERE done=0").fetchall()
        if len(active) >= int(os.environ.get("PETREAD_MAX_ACTIVE_JOBS", "2")) or any(r[0] == job["client_id"] for r in active):
            raise AdmissionError("已有任务运行，请等待完成")
        # Counters are independent of deletable history: deleting a device's
        # records must not reset the global daily spending guard.
        day = int(now // 86400)
        limits = [(f"jobs-global:{day}", int(os.environ.get("PETREAD_DAILY_JOBS", "60"))),
                  (f"jobs-device:{job['client_id']}:{day}", int(os.environ.get("PETREAD_DEVICE_DAILY_JOBS", "10")))]
        for bucket, limit in limits:
            row = conn.execute("SELECT count FROM admission_counters WHERE bucket=?", (bucket,)).fetchone()
            if row and row[0] >= limit or limit <= 0:
                raise AdmissionError("今日分析额度已用完，请明天再试")
        conn.execute("INSERT INTO analysis_jobs (id,client_id,request_key,created,upload_path) VALUES (?,?,?,?,?)",
                     (job["id"], job["client_id"], request_key, job["created"],job.get('upload_path')))
        for bucket, _ in limits:
            conn.execute("INSERT INTO admission_counters VALUES (?,1,?) ON CONFLICT(bucket) DO UPDATE SET count=count+1",
                         (bucket, now + 2 * 86400))


def save_job(job: dict) -> None:
    if not job.get("durable"):
        return
    with store.connect() as conn:
        conn.execute("UPDATE analysis_jobs SET events=?,done=?,finished=? WHERE id=?",
                     (json.dumps(job["events"], ensure_ascii=False), int(job["done"]),
                      job.get("finished_at"), job["id"]))


def delete_device(client_id: str) -> dict:
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM analysis_jobs WHERE client_id=? AND done=0", (client_id,)).fetchone():
            raise AdmissionError("请等待当前任务结束后再删除，避免删除后又产生新记录")
        uploads = conn.execute("SELECT upload_path FROM analysis_jobs WHERE client_id=? AND upload_path IS NOT NULL",(client_id,)).fetchall()
        if any(os.path.lexists(row[0]) for row in uploads):
            raise AdmissionError("原始素材尚未清理完成，请稍后重试或联系支持；尚未删除数据")
        counts = {}
        counts["feedback"] = conn.execute("DELETE FROM feedback WHERE record_id IN (SELECT id FROM records WHERE client_id=?)", (client_id,)).rowcount
        counts["production_attempts"] = conn.execute("DELETE FROM production_attempts WHERE job_id IN (SELECT id FROM analysis_jobs WHERE client_id=?)", (client_id,)).rowcount
        for table in ("records", "profiles", "analysis_jobs", "device_sessions"):
            counts[table] = conn.execute(f"DELETE FROM {table} WHERE client_id=?", (client_id,)).rowcount
        # No IP addresses/media are stored here. Retain only global counters.
        conn.execute("DELETE FROM admission_counters WHERE bucket LIKE ?", (f"jobs-device:{client_id}:%",))
    return counts


def attempt_observer(job_id: str):
    current = None
    def observe(event: str, data: dict):
        nonlocal current
        with store.connect() as conn:
            if event == 'started':
                current = uuid.uuid4().hex
                # This durable reservation precedes dispatch. If the process
                # dies in the gap, actual issuance is UNKNOWN, not invented 0/1.
                conn.execute("INSERT INTO production_attempts (id,job_id,model,started,status) VALUES (?,?,?,?,?)",
                             (current, job_id, data['model'], time.time(), 'dispatch_reserved_outcome_unknown'))
            elif event == 'received':
                conn.execute("UPDATE production_attempts SET issued=1,http_body=? WHERE id=?", (data['http_body'],current))
            elif event == 'finished':
                conn.execute("UPDATE production_attempts SET issued=1,finished=?,status=?,raw=?,usage=?,latency_s=?,reason=? WHERE id=?",
                             (time.time(),data['status'],data.get('raw'),json.dumps(data['usage']) if data.get('usage') is not None else None,
                              data.get('latency_s'),data.get('reason'),current))
    return observe


def attempts_for_device(client_id: str) -> list[dict]:
    with store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT a.* FROM production_attempts a JOIN analysis_jobs j ON a.job_id=j.id WHERE j.client_id=? ORDER BY a.started", (client_id,))]


def acquire_worker_lock():
    path = os.path.join(os.path.dirname(store.DEFAULT_DB_PATH), 'petread-worker.lock')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    handle = os.fdopen(fd, 'a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('PetRead SQLite 模式只允许一个 worker；另一进程已经持有任务数据库')
    return handle


def interrupted_uploads(client_id: str | None = None) -> list[tuple[str, str]]:
    with store.connect() as conn:
        sql="SELECT id,upload_path FROM analysis_jobs WHERE done=1 AND upload_path IS NOT NULL"
        rows=conn.execute(sql+' AND client_id=?',(client_id,)) if client_id else conn.execute(sql)
        return [(r['id'],r['upload_path']) for r in rows]


def clear_upload_reference(job_id: str) -> None:
    with store.connect() as conn:
        conn.execute('UPDATE analysis_jobs SET upload_path=NULL WHERE id=?',(job_id,))
