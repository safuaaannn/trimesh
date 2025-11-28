import json
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class SQLiteSessionStore:
    """Lightweight persistent session store backed by sqlite."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    filepath TEXT NOT NULL,
                    session_dir TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    num_persons INTEGER DEFAULT 0,
                    rig_data TEXT,
                    error TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                ON sessions(status);
                """
            )

    def register_session(
        self,
        session_id: str,
        filepath: str,
        session_dir: str,
        original_filename: str,
    ) -> Dict[str, Any]:
        now = time.time()
        payload = {
            "session_id": session_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "filepath": filepath,
            "session_dir": session_dir,
            "original_filename": original_filename,
            "num_persons": 0,
            "rig_data": None,
            "error": None,
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, status, created_at, updated_at,
                    filepath, session_dir, original_filename,
                    num_persons, rig_data, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    payload["session_id"],
                    payload["status"],
                    payload["created_at"],
                    payload["updated_at"],
                    payload["filepath"],
                    payload["session_dir"],
                    payload["original_filename"],
                    payload["num_persons"],
                    payload["rig_data"],
                    payload["error"],
                ),
            )
        return payload

    def update_session(self, session_id: str, **fields) -> Optional[Dict[str, Any]]:
        if not fields:
            return self.get_session(session_id)

        fields["updated_at"] = time.time()
        set_clause = ", ".join(f"{column}=?" for column in fields.keys())
        values = [self._prepare_value(column, value) for column, value in fields.items()]
        values.append(session_id)

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"UPDATE sessions SET {set_clause} WHERE session_id=?;",
                values,
            )
            if cur.rowcount == 0:
                return None
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?;",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def _prepare_value(self, column: str, value: Any):
        if column == "rig_data":
            if value is None:
                return None
            return json.dumps(value)
        return value

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        if data.get("rig_data"):
            data["rig_data"] = json.loads(data["rig_data"])
        else:
            data["rig_data"] = None
        return data

