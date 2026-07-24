"""SQLite 持久层。用标准库 sqlite3,不引入 ORM。

约定
----
* 每次操作开一个短连接,避免跨线程复用带来的问题(FastAPI 会在线程池里跑同步代码)。
* 时间统一存 ISO 8601 字符串(本地时区),便于直接展示与排序。
* ``result_json`` 存模型返回的完整 dict,不做裁剪,便于事后重新分析。
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据根目录:SQLite 与上传文件都放这里。由 DATA_DIR 控制,默认项目内的 ./data。
# HF Spaces 上可设成 /data(需开启持久化存储,否则重启即丢)。
DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(PROJECT_ROOT, "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "petread.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    species         TEXT    NOT NULL,
    mode            TEXT    NOT NULL,
    context         TEXT    NOT NULL DEFAULT '',
    media_path      TEXT    NOT NULL DEFAULT '',
    result_json     TEXT    NOT NULL DEFAULT '{}',
    predicted_label TEXT,
    confidence      REAL,
    tokens_in       INTEGER,
    tokens_out      INTEGER,
    latency_s       REAL,
    provider        TEXT,
    model           TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id    INTEGER NOT NULL REFERENCES records(id),
    actual_label TEXT    NOT NULL DEFAULT '',
    user_note    TEXT    NOT NULL DEFAULT '',
    created_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    pet_name          TEXT NOT NULL,
    species           TEXT NOT NULL,
    calibration_notes TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_created  ON records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_record  ON feedback(record_id);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def insert_record(
    *,
    species: str,
    mode: str,
    context: str,
    media_path: str,
    result: dict,
    predicted_label: str | None,
    confidence: float | None,
    tokens_in: int | None,
    tokens_out: int | None,
    latency_s: float | None,
    provider: str | None,
    model: str | None,
    db_path: str | None = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO records
               (created_at, species, mode, context, media_path, result_json,
                predicted_label, confidence, tokens_in, tokens_out, latency_s,
                provider, model)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (_now(), species, mode, context, media_path,
             json.dumps(result, ensure_ascii=False),
             predicted_label, confidence, tokens_in, tokens_out, latency_s,
             provider, model),
        )
        return int(cur.lastrowid)


def get_record(record_id: int, db_path: str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
    return _row_to_record(row) if row else None


def list_records(limit: int = 50, offset: int = 0,
                 db_path: str | None = None) -> list[dict]:
    """按时间倒序列出记录,附带该记录的反馈条数与最近一条反馈。"""
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT r.*,
                      (SELECT COUNT(*) FROM feedback f WHERE f.record_id = r.id)
                        AS feedback_count,
                      (SELECT f.actual_label FROM feedback f
                        WHERE f.record_id = r.id ORDER BY f.id DESC LIMIT 1)
                        AS last_actual_label
                 FROM records r
                ORDER BY r.id DESC
                LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def _row_to_record(row: sqlite3.Row) -> dict:
    item = dict(row)
    try:
        item["result"] = json.loads(item.pop("result_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        item["result"] = {}
    return item


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------

def insert_feedback(record_id: int, actual_label: str, user_note: str = "",
                    db_path: str | None = None) -> int:
    with connect(db_path) as conn:
        exists = conn.execute("SELECT 1 FROM records WHERE id = ?",
                              (record_id,)).fetchone()
        if not exists:
            raise ValueError(f"record {record_id} 不存在")
        cur = conn.execute(
            """INSERT INTO feedback (record_id, actual_label, user_note, created_at)
               VALUES (?,?,?,?)""",
            (record_id, actual_label, user_note, _now()),
        )
        return int(cur.lastrowid)


def list_feedback(record_id: int, db_path: str | None = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM feedback WHERE record_id = ? ORDER BY id DESC",
            (record_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

def upsert_profile(pet_name: str, species: str, calibration_notes: str = "",
                   db_path: str | None = None) -> int:
    """同名同物种的档案覆盖校准备注,否则新建。"""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM profiles WHERE pet_name = ? AND species = ?",
            (pet_name, species),
        ).fetchone()
        if row:
            conn.execute("UPDATE profiles SET calibration_notes = ? WHERE id = ?",
                         (calibration_notes, row["id"]))
            return int(row["id"])
        cur = conn.execute(
            """INSERT INTO profiles (pet_name, species, calibration_notes, created_at)
               VALUES (?,?,?,?)""",
            (pet_name, species, calibration_notes, _now()),
        )
        return int(cur.lastrowid)


def list_profiles(db_path: str | None = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM profiles ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 结果摘要:从模型输出里抽出要单独存列的字段
# ---------------------------------------------------------------------------

def summarize_result(result: dict) -> tuple[str | None, float | None]:
    """取置信度最高的候选作为 predicted_label / confidence。

    ``insufficient_evidence`` 为真或没有候选时返回 ``(None, None)``,
    不要伪造一个标签。
    """
    if not isinstance(result, dict) or result.get("insufficient_evidence"):
        return None, None
    candidates = result.get("candidates") or []
    best_label, best_conf = None, None
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        try:
            conf = float(cand.get("confidence"))
        except (TypeError, ValueError):
            continue
        if best_conf is None or conf > best_conf:
            best_label, best_conf = str(cand.get("label") or ""), conf
    return best_label, best_conf


if __name__ == "__main__":
    init_db()
    print(f"已初始化: {DEFAULT_DB_PATH}")
    with connect() as c:
        tables = [r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    print("表:", tables)
