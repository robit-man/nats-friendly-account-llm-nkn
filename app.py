#!/usr/bin/env python3
"""Standalone NKN ⇄ Ollama relay (no embedded web server)."""

from __future__ import annotations

import atexit
import contextlib
import csv
import datetime
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# === Virtualenv bootstrap ====================================================

BASE_DIR: Path = Path(__file__).resolve().parent
VENV_DIR: Path = BASE_DIR / ".venv"
VENV_BIN_DIR: Path = VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
VENV_PYTHON: Path = VENV_BIN_DIR / ("python.exe" if os.name == "nt" else "python")

REQUIRED_PACKAGES = ["requests"]


def _in_managed_venv() -> bool:
    try:
        return VENV_DIR.resolve() == Path(sys.prefix).resolve()
    except Exception:  # pragma: no cover - safety fallthrough
        return False


def _bootstrap_venv_if_needed() -> None:
    if __name__ != "__main__":
        return
    if _in_managed_venv():
        return

    if not VENV_PYTHON.exists():
        print("[bootstrap] Creating .venv …")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
        if REQUIRED_PACKAGES:
            subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", *REQUIRED_PACKAGES])

    env = os.environ.copy()
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]], env)


_bootstrap_venv_if_needed()

# === Runtime imports ========================================================

import codecs
import requests
import base64

def infer_capabilities_from_name(name: str) -> Dict[str, bool]:
    """Heuristic capability detection for models based on name."""
    lowered = (name or "").lower()
    supports_images = any(
        key in lowered
        for key in [
            "vision",
            "vl",
            "clip",
            "llava",
            "vmodel",
            "gpt-4o",
            "vision-",
            "mm",
            "multimodal",
            "moondream",
            "idefics",
            "qwen-vl",
        ]
    )
    supports_thinking = any(
        key in lowered
        for key in [
            "think",
            "deepseek",
            "reason",
            "qwen2.5",
            "qwen2.5-thinking",
            "thinking",
        ]
    )
    supports_tools = any(
        key in lowered
        for key in [
            "tool",
            "tools",
            "function",
            "functions",
            "function_call",
            "function-call",
        ]
    )
    return {"supports_images": supports_images, "supports_thinking": supports_thinking, "supports_tools": supports_tools}


def infer_capabilities(name: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """Merge name-based and metadata-based capability hints."""
    caps = infer_capabilities_from_name(name)
    meta = details or {}
    meta_caps = meta.get("capabilities") or meta.get("capability") or []
    if isinstance(meta_caps, str):
        meta_caps = [meta_caps]
    family_fields: List[str] = []
    for key in ("family", "model_type"):
        if meta.get(key):
            family_fields.append(str(meta[key]))
    if isinstance(meta.get("families"), list):
        family_fields.extend([str(f) for f in meta.get("families") if f])
    meta_tokens = [str(x).lower() for x in meta_caps] + [f.lower() for f in family_fields]
    if any(tok in ("vision", "image", "images", "multimodal", "vl", "mm") or "vision" in tok for tok in meta_tokens):
        caps["supports_images"] = True
    if any(tok in ("thinking", "think", "reason", "reasoning") for tok in meta_tokens):
        caps["supports_thinking"] = True
    if any(tok in ("tool", "tools", "function", "functions", "function_call", "function-calling", "function-calls") or "tool" in tok for tok in meta_tokens):
        caps["supports_tools"] = True
    return caps


def assemble_image_chunks(chunks: List[Any], default_mime: str = "") -> Tuple[Optional[str], Optional[str]]:
    """Reassemble chunked base64 image payloads; returns (b64, error)."""
    if not chunks:
        return None, None
    b64_data = ""
    if isinstance(chunks[0], dict):
        shards = []
        for sh in chunks:
            if not isinstance(sh, dict):
                return None, "Invalid image chunk format"
            try:
                idx = int(sh.get("i", 0))
                total = int(sh.get("t", 0))
            except Exception:
                return None, "Invalid image chunk indices"
            data = sh.get("data") or sh.get("chunk") or sh.get("c")
            if not isinstance(data, str) or not data:
                return None, "Missing image chunk data"
            if len(data) > IMAGE_CHUNK_MAX_BYTES * 2:  # guard per-chunk envelope
                return None, "Image chunk exceeds size limit"
            shards.append((idx, total, data))
        shards.sort(key=lambda s: s[0])
        if shards and shards[0][0] != 0:
            return None, "Image chunks missing start chunk"
        declared_total = shards[0][1] if shards else 0
        if declared_total and declared_total != len(shards):
            return None, f"Image chunks incomplete ({len(shards)}/{declared_total})"
        # check contiguity
        for expected_idx, (idx, _, _) in enumerate(shards):
            if idx != expected_idx:
                return None, f"Image chunks missing index {expected_idx}"
        total_base64_len = sum(len(data) for _, _, data in shards)
        if total_base64_len > IMAGE_TOTAL_MAX_BYTES * 1.5:
            return None, "Image too large"
        b64_data = "".join(data for _, _, data in shards)
    else:
        parts = [c for c in chunks if isinstance(c, str)]
        if not parts:
            return None, "Invalid image chunk data"
        total_base64_len = sum(len(p) for p in parts)
        if total_base64_len > IMAGE_TOTAL_MAX_BYTES * 1.5:
            return None, "Image too large"
        b64_data = "".join(parts)

    if not b64_data:
        return None, "Empty image payload"
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except Exception:
        return None, "Invalid base64 image data"
    if len(raw) > IMAGE_TOTAL_MAX_BYTES:
        return None, "Image exceeds size limit"
    return b64_data, None

# === App configuration ======================================================

APP_TITLE = "Inference Relay"
SYSTEM_PROMPT_FILE = BASE_DIR / "system.md"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_CHAT_ENDPOINT = "/api/chat"
OLLAMA_TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT", "300"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "")
OLLAMA_KEEPALIVE_INTERVAL_S = int(os.environ.get("OLLAMA_KEEPALIVE_INTERVAL", "240"))
OLLAMA_STREAM_CHUNK_B = int(os.environ.get("OLLAMA_STREAM_CHUNK_B", str(16 * 1024)))
OLLAMA_HEARTBEAT_S = int(os.environ.get("OLLAMA_HEARTBEAT_S", "10"))
MAX_CONTEXT_MESSAGES = int(os.environ.get("MAX_CONTEXT", "30"))
MAX_MESSAGE_LENGTH = 4000
IMAGE_CHUNK_MAX_BYTES = 250 * 1024  # keep well under 1MB per packet
IMAGE_TOTAL_MAX_BYTES = 8 * 1024 * 1024  # safety cap per message
PENDING_IMAGE_TTL_S = 10 * 60  # expire uploads after 10 minutes

DATABASE_PATH = BASE_DIR / "chat.db"
SQLITE_FOREIGN_KEYS_ON = "PRAGMA foreign_keys = ON;"

PASSWORD_HASH_ALGORITHM = "sha256"
PASSWORD_PBKDF2_ITERATIONS = 150_000
PASSWORD_SALT_BYTES = 16
PASSWORD_HASH_SEPARATOR = "$"

NKN_IDENTIFIER = os.environ.get("NKN_IDENTIFIER", "inference")
NKN_NUM_SUBCLIENTS = int(os.environ.get("NKN_NUM_SUBCLIENTS", "8"))
LEGACY_BRIDGE_DIR = BASE_DIR / ".nkn_bridge"
DEFAULT_BRIDGE_DIR = BASE_DIR / "nkn_bridge"
_bridge_dir_env = os.environ.get("NKN_BRIDGE_DIR")
if _bridge_dir_env:
    NKN_BRIDGE_DIR = Path(_bridge_dir_env).expanduser()
else:
    if LEGACY_BRIDGE_DIR.exists() and not DEFAULT_BRIDGE_DIR.exists():
        NKN_BRIDGE_DIR = LEGACY_BRIDGE_DIR
    else:
        NKN_BRIDGE_DIR = DEFAULT_BRIDGE_DIR
NKN_JS_NAME = "bridge.js"
NKN_PACKAGE_VERSION = "1.3.6"


# === SQLite helpers =========================================================


def hash_password(plaintext: str) -> str:
    salt = secrets.token_bytes(PASSWORD_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        PASSWORD_HASH_ALGORITHM,
        plaintext.encode("utf-8"),
        salt,
        PASSWORD_PBKDF2_ITERATIONS,
    )
    return PASSWORD_HASH_SEPARATOR.join(
        [
            PASSWORD_HASH_ALGORITHM,
            str(PASSWORD_PBKDF2_ITERATIONS),
            salt.hex(),
            derived.hex(),
        ]
    )


def verify_password(plaintext: str, stored: str) -> bool:
    try:
        algo, iterations_str, salt_hex, hash_hex = stored.split(PASSWORD_HASH_SEPARATOR)
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        return False

    candidate = hashlib.pbkdf2_hmac(
        algo,
        plaintext.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate, expected)


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._messages_has_model: bool = False
        self._users_has_prefs: bool = False
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        default_model_sql = OLLAMA_MODEL_NAME.replace("'", "''")
        with self._lock:
            cur = self._connection.cursor()
            cur.execute(SQLITE_FOREIGN_KEYS_ON)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_nkn_addr TEXT DEFAULT '',
                    preferences TEXT DEFAULT '{}'
                );
                """
            )
            try:
                cur.execute(
                    "ALTER TABLE users ADD COLUMN last_nkn_addr TEXT DEFAULT ''"
                )
                self._connection.commit()
            except sqlite3.OperationalError:
                pass
            try:
                cur.execute("ALTER TABLE users ADD COLUMN preferences TEXT DEFAULT '{}'")
                self._connection.commit()
            except sqlite3.OperationalError:
                pass
            cur.execute("PRAGMA table_info(users);")
            user_cols = {row[1] for row in cur.fetchall()}
            self._users_has_prefs = "preferences" in user_cols

            # Chat sessions table
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL DEFAULT 'New Chat',
                    model TEXT NOT NULL DEFAULT '{default_model_sql}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id INTEGER,
                    model TEXT NOT NULL DEFAULT '{default_model_sql}',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_thumb TEXT,
                    uuid TEXT,
                    is_complete INTEGER DEFAULT 1,
                    last_seq INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );
                """
            )

            # Add session_id column to existing messages table if it doesn't exist
            try:
                cur.execute("ALTER TABLE messages ADD COLUMN session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE")
                self._connection.commit()
            except sqlite3.OperationalError:
                pass

            # Add uuid column to existing messages table if it doesn't exist
            try:
                cur.execute("ALTER TABLE messages ADD COLUMN uuid TEXT")
                self._connection.commit()
            except sqlite3.OperationalError:
                pass

            # Add model column to existing messages table if it doesn't exist
            cur.execute("PRAGMA table_info(messages);")
            message_cols = {row[1] for row in cur.fetchall()}
            if "model" not in message_cols:
                try:
                    cur.execute(f"ALTER TABLE messages ADD COLUMN model TEXT DEFAULT '{default_model_sql}'")
                    self._connection.commit()
                    message_cols.add("model")
                    print("[db] Added model column to messages")
                except sqlite3.OperationalError:
                    print("[db] Could not add model column to messages; falling back to default model in queries")
            self._messages_has_model = "model" in message_cols

            # Create unique index on uuid to prevent duplicates
            try:
                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_uuid ON messages(uuid) WHERE uuid IS NOT NULL")
                self._connection.commit()
            except sqlite3.OperationalError:
                pass

            # Add is_complete column for progressive persistence
            try:
                cur.execute("ALTER TABLE messages ADD COLUMN is_complete INTEGER DEFAULT 1")
                self._connection.commit()
            except sqlite3.OperationalError:
                pass

            # Add last_seq column for resumable streaming
            try:
                cur.execute("ALTER TABLE messages ADD COLUMN last_seq INTEGER DEFAULT 0")
                self._connection.commit()
            except sqlite3.OperationalError:
                pass

            # Add image_thumb column for storing small previews
            try:
                cur.execute("ALTER TABLE messages ADD COLUMN image_thumb TEXT")
                self._connection.commit()
                message_cols.add("image_thumb")
            except sqlite3.OperationalError:
                pass

            self._messages_has_thumb = "image_thumb" in message_cols

            # Add model column to chat_sessions if it doesn't exist
            try:
                cur.execute(f"ALTER TABLE chat_sessions ADD COLUMN model TEXT NOT NULL DEFAULT '{default_model_sql}'")
                self._connection.commit()
            except sqlite3.OperationalError:
                pass

            self._connection.commit()

    def create_user(self, username: str, password: str) -> Optional[int]:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        password_hash = hash_password(password)
        with self._lock:
            try:
                if self._users_has_prefs:
                    cur = self._connection.execute(
                        "INSERT INTO users (username, password_hash, created_at, preferences) VALUES (?, ?, ?, '{}');",
                        (username, password_hash, now),
                    )
                else:
                    cur = self._connection.execute(
                        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?);",
                        (username, password_hash, now),
                    )
                self._connection.commit()
                return int(cur.lastrowid)
            except sqlite3.IntegrityError:
                return None

    def authenticate_user(self, username: str, password: str) -> Optional[Tuple[int, str, Dict[str, Any]]]:
        with self._lock:
            cur = self._connection.execute(
                "SELECT id, username, password_hash, preferences FROM users WHERE username = ?;",
                (username,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        prefs = {}
        if self._users_has_prefs and row["preferences"]:
            try:
                prefs = json.loads(row["preferences"])
            except Exception:
                prefs = {}
        return row["id"], row["username"], prefs

    def create_session(self, user_id: int, title: str = "New Chat", model: Optional[str] = None) -> int:
        """Create a new chat session and return its ID."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        chosen_model = model or OLLAMA_MODEL_NAME
        with self._lock:
            cur = self._connection.execute(
                "INSERT INTO chat_sessions (user_id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?);",
                (user_id, title, chosen_model, now, now),
            )
            self._connection.commit()
            return int(cur.lastrowid)

    def get_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all chat sessions for a user, ordered by most recent."""
        with self._lock:
            cur = self._connection.execute(
                """
                SELECT id, title, model, created_at, updated_at,
                       (SELECT COUNT(*) FROM messages WHERE session_id = chat_sessions.id) as message_count
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC;
                """,
                (user_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "model": row["model"] or OLLAMA_MODEL_NAME,
                "message_count": row["message_count"],
            }
            for row in rows
        ]

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._connection.execute(
                "SELECT id, title, model, created_at, updated_at FROM chat_sessions WHERE id = ?;",
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "title": row["title"],
            "model": row["model"] or OLLAMA_MODEL_NAME,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_session_model(self, session_id: int) -> str:
        session = self.get_session(session_id)
        return (session.get("model") if session else None) or OLLAMA_MODEL_NAME

    def update_session_model(self, session_id: int, model: str) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            self._connection.execute(
                "UPDATE chat_sessions SET model = ?, updated_at = ? WHERE id = ?;",
                (model, now, session_id),
            )
            self._connection.commit()

    def get_or_create_default_session(self, user_id: int, default_model: Optional[str] = None) -> int:
        """Get the most recent session or create a new one."""
        sessions = self.get_sessions(user_id)
        if sessions:
            return sessions[0]["id"]
        return self.create_session(user_id, "New Chat", model=default_model or OLLAMA_MODEL_NAME)

    def update_session_title(self, session_id: int, title: str) -> None:
        """Update session title."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            self._connection.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?;",
                (title, now, session_id),
            )
            self._connection.commit()

    def update_session_timestamp(self, session_id: int) -> None:
        """Update session's updated_at timestamp."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            self._connection.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?;",
                (now, session_id),
            )
            self._connection.commit()

    def delete_session(self, session_id: int, user_id: int) -> bool:
        """Delete a session (messages will cascade delete)."""
        with self._lock:
            cur = self._connection.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?;",
                (session_id, user_id),
            )
            self._connection.commit()
            return cur.rowcount > 0

    def append_message(self, user_id: int, session_id: int, role: str, content: str, uuid: str = None, model: Optional[str] = None, image_thumb: Optional[str] = None) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        model_name = model or OLLAMA_MODEL_NAME
        with self._lock:
            try:
                columns = ["user_id", "session_id", "role", "content", "uuid", "created_at"]
                values = [user_id, session_id, role, content, uuid, now]
                if self._messages_has_model:
                    columns.insert(5, "model")
                    values.insert(5, model_name)
                if getattr(self, "_messages_has_thumb", False):
                    columns.insert(-1, "image_thumb")
                    values.insert(-1, image_thumb)
                placeholders = ", ".join(["?"] * len(values))
                col_sql = ", ".join(columns)
                self._connection.execute(
                    f"INSERT INTO messages ({col_sql}) VALUES ({placeholders});",
                    tuple(values),
                )
                self._connection.commit()
            except sqlite3.IntegrityError as e:
                # UUID already exists, skip duplicate insert
                if "uuid" in str(e).lower():
                    print(f"[db] Skipped duplicate message with uuid={uuid}")
                else:
                    raise
        # Update session timestamp
        self.update_session_timestamp(session_id)

    def get_recent_messages(self, user_id: int, session_id: int, limit: int) -> List[Dict[str, str]]:
        with self._lock:
            select_cols = ["role", "content", "uuid", "is_complete", "last_seq"]
            if self._messages_has_model:
                select_cols.append("model")
            if getattr(self, "_messages_has_thumb", False):
                select_cols.append("image_thumb")
            col_sql = ", ".join(select_cols)
            cur = self._connection.execute(
                f"""
                SELECT {col_sql}
                FROM messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY id ASC
                LIMIT ?;
                """,
                (user_id, session_id, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "id": row["uuid"],
                "is_complete": bool(row["is_complete"] if row["is_complete"] is not None else True),
                "last_seq": row["last_seq"] if row["last_seq"] is not None else 0,
                "model": row["model"] if "model" in row.keys() else OLLAMA_MODEL_NAME,
                "image_thumb": row["image_thumb"] if "image_thumb" in row.keys() else None,
            }
            for row in rows
        ]

    def get_message_by_uuid(self, uuid: Optional[str]) -> Optional[Dict[str, Any]]:
        if not uuid:
            return None
        with self._lock:
            select_cols = ["role", "content", "is_complete", "last_seq"]
            if self._messages_has_model:
                select_cols.append("model")
            if getattr(self, "_messages_has_thumb", False):
                select_cols.append("image_thumb")
            col_sql = ", ".join(select_cols)
            cur = self._connection.execute(
                f"""
                SELECT {col_sql}
                FROM messages
                WHERE uuid = ?
                ORDER BY id DESC
                LIMIT 1;
                """,
                (uuid,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "role": row["role"],
            "content": row["content"],
            "is_complete": bool(row["is_complete"] if row["is_complete"] is not None else True),
            "last_seq": row["last_seq"] if row["last_seq"] is not None else 0,
            "model": row["model"] if "model" in row.keys() else OLLAMA_MODEL_NAME,
            "image_thumb": row["image_thumb"] if "image_thumb" in row.keys() else None,
        }

    def update_last_nkn_addr(self, user_id: int, address: str) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE users SET last_nkn_addr = ? WHERE id = ?;",
                (address, user_id),
            )
            self._connection.commit()

    def set_user_preferences(self, user_id: int, prefs: Dict[str, Any]) -> None:
        if not self._users_has_prefs:
            return
        try:
            encoded = json.dumps(prefs or {})
        except Exception:
            encoded = "{}"
        with self._lock:
            self._connection.execute(
            "UPDATE users SET preferences = ? WHERE id = ?;",
            (encoded, user_id),
        )
            self._connection.commit()

    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        if not self._users_has_prefs:
            return {}
        with self._lock:
            cur = self._connection.execute(
                "SELECT preferences FROM users WHERE id = ? LIMIT 1;",
                (user_id,),
            )
            row = cur.fetchone()
        if not row or not row["preferences"]:
            return {}
        try:
            return json.loads(row["preferences"])
        except Exception:
            return {}

    def upsert_partial_message(self, user_id: int, session_id: int, role: str, content: str, uuid: str, last_seq: int, model: Optional[str] = None) -> None:
        """Create or update a partial (streaming) message."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        model_name = model or OLLAMA_MODEL_NAME
        with self._lock:
            # Check if message with this UUID already exists
            cur = self._connection.execute(
                "SELECT id FROM messages WHERE uuid = ?;",
                (uuid,),
            )
            existing = cur.fetchone()

            if existing:
                # Update existing partial message
                self._connection.execute(
                    "UPDATE messages SET content = ?, last_seq = ?, is_complete = 0" + (" , model = ?" if self._messages_has_model else "") + " WHERE uuid = ?;",
                    (content, last_seq, *( [model_name] if self._messages_has_model else [] ), uuid),
                )
            else:
                # Create new partial message
                try:
                    if self._messages_has_model:
                        self._connection.execute(
                            "INSERT INTO messages (user_id, session_id, role, content, uuid, model, is_complete, last_seq, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?);",
                            (user_id, session_id, role, content, uuid, model_name, last_seq, now),
                        )
                    else:
                        self._connection.execute(
                            "INSERT INTO messages (user_id, session_id, role, content, uuid, is_complete, last_seq, created_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?);",
                            (user_id, session_id, role, content, uuid, last_seq, now),
                        )
                except sqlite3.IntegrityError as e:
                    if "uuid" in str(e).lower():
                        print(f"[db] UUID collision during upsert for uuid={uuid}")
                    else:
                        raise

            self._connection.commit()
        # Update session timestamp
        self.update_session_timestamp(session_id)

    def mark_message_complete(self, uuid: str, final_content: str, final_seq: int, model: Optional[str] = None) -> None:
        """Mark a partial message as complete."""
        with self._lock:
            if self._messages_has_model:
                self._connection.execute(
                    "UPDATE messages SET content = ?, last_seq = ?, is_complete = 1, model = COALESCE(model, ?) WHERE uuid = ?;",
                    (final_content, final_seq, model or OLLAMA_MODEL_NAME, uuid),
                )
            else:
                self._connection.execute(
                    "UPDATE messages SET content = ?, last_seq = ?, is_complete = 1 WHERE uuid = ?;",
                    (final_content, final_seq, uuid),
                )
            self._connection.commit()

    def get_partial_message(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Get partial message state for resumable streaming."""
        with self._lock:
            if self._messages_has_model:
                cur = self._connection.execute(
                    "SELECT content, last_seq, is_complete, model FROM messages WHERE uuid = ?;",
                    (uuid,),
                )
            else:
                cur = self._connection.execute(
                    "SELECT content, last_seq, is_complete FROM messages WHERE uuid = ?;",
                    (uuid,),
                )
            row = cur.fetchone()
        if row:
            return {
                "content": row["content"],
                "last_seq": row["last_seq"],
                "is_complete": bool(row["is_complete"]),
                "model": row["model"] if "model" in row.keys() else OLLAMA_MODEL_NAME,
            }
        return None


class SessionManager:
    """Manages user sessions bound to NKN addresses with expiry."""
    def __init__(self) -> None:
        self._sessions: Dict[str, Tuple[int, str, int, float]] = {}  # src -> (user_id, username, current_session_id, timestamp)
        self._lock = threading.Lock()
        self._session_timeout_s = 3600 * 24  # 24 hours

    def login(self, src: str, user_id: int, username: str, session_id: int) -> None:
        """Create a new session bound to the NKN address."""
        with self._lock:
            self._sessions[src] = (user_id, username, session_id, time.time())
            print(f"[session] Created session for user_id={user_id} username={username} session_id={session_id} addr={src}")

    def logout(self, src: str) -> None:
        """Remove session for the given NKN address."""
        with self._lock:
            if src in self._sessions:
                user_id, username, session_id, _ = self._sessions[src]
                print(f"[session] Logout user_id={user_id} username={username} addr={src}")
            self._sessions.pop(src, None)

    def get(self, src: str) -> Optional[Tuple[int, str, int]]:
        """Get session for the given NKN address, validating it hasn't expired."""
        with self._lock:
            session = self._sessions.get(src)
            if session is None:
                return None
            user_id, username, session_id, timestamp = session
            # Check if session expired
            if time.time() - timestamp > self._session_timeout_s:
                print(f"[session] Expired session for user_id={user_id} addr={src}")
                self._sessions.pop(src, None)
                return None
            return (user_id, username, session_id)

    def refresh(self, src: str) -> None:
        """Refresh session timestamp on activity."""
        with self._lock:
            session = self._sessions.get(src)
            if session:
                user_id, username, session_id, _ = session
                self._sessions[src] = (user_id, username, session_id, time.time())

    def set_current_session(self, src: str, session_id: int) -> None:
        """Switch the current session for a user."""
        with self._lock:
            session = self._sessions.get(src)
            if session:
                user_id, username, _, timestamp = session
                self._sessions[src] = (user_id, username, session_id, timestamp)

    def validate_user_address(self, db: Database, src: str, user_id: int) -> bool:
        """Verify that the user's last known NKN address matches the current one."""
        # This adds an extra layer of security by verifying the stored address
        # matches the current connection address
        return True  # For now, rely on session binding; can be enhanced later

# === Helpers: system prompt and Ollama ======================================


class SystemPromptLoader:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._cached_text: Optional[str] = None
        self._cached_mtime: Optional[float] = None

    def get_prompt(self) -> str:
        if not self._path.exists():
            return "You are a helpful assistant."
        mtime = self._path.stat().st_mtime
        if self._cached_text is None or self._cached_mtime != mtime:
            self._cached_text = self._path.read_text(encoding="utf-8")
            self._cached_mtime = mtime
        return self._cached_text


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        endpoint: str,
        model_name: str,
        timeout_s: int,
        keep_alive: str,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._endpoint = endpoint
        self._model = model_name
        self._timeout = timeout_s
        self._keep_alive = keep_alive

    def stream(self, messages: List[Dict[str, str]], model_name: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None) -> Iterable[Dict[str, Any]]:
        url = f"{self._base}{self._endpoint}"
        target_model = model_name or self._model
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if self._keep_alive:
            payload["keep_alive"] = self._keep_alive
        response = requests.post(url, json=payload, timeout=self._timeout, stream=True)
        response.raise_for_status()

        decoder = codecs.getincrementaldecoder("utf-8")()
        buf = ""
        seq = 0
        last_event = time.time()

        for chunk in response.iter_content(chunk_size=OLLAMA_STREAM_CHUNK_B):
            if not chunk:
                if time.time() - last_event >= OLLAMA_HEARTBEAT_S:
                    yield {"keepalive": True, "timestamp": time.time()}
                    last_event = time.time()
                continue
            decoded = decoder.decode(chunk)
            if not decoded:
                continue
            buf += decoded
            while True:
                nl = buf.find("\n")
                if nl < 0:
                    break
                line = buf[:nl]
                buf = buf[nl + 1:]
                if not line.strip():
                    continue
                seq += 1
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                packet = {
                    "seq": seq,
                    "timestamp": time.time(),
                    "detail": data.get("detail"),
                    "model": data.get("model"),
                }
                message = data.get("message") or {}
                content = ""
                if isinstance(message, dict):
                    content = message.get("content") or ""
                    if message.get("tool_calls"):
                        packet["tool_calls"] = message.get("tool_calls")
                elif isinstance(message, str):
                    content = message
                if content:
                    packet["content"] = content
                if data.get("done"):
                    packet["done"] = True
                yield packet
                last_event = time.time()
        tail = decoder.decode(b"", final=True)
        if tail:
            buf += tail
        if buf.strip():
            seq += 1
            try:
                data = json.loads(buf)
            except json.JSONDecodeError:
                return
            packet = {
                "seq": seq,
                "timestamp": time.time(),
                "detail": data.get("detail"),
                "model": data.get("model"),
            }
            message = data.get("message") or {}
            if isinstance(message, dict):
                packet["content"] = message.get("content") or ""
                if message.get("tool_calls"):
                    packet["tool_calls"] = message.get("tool_calls")
            elif isinstance(message, str):
                packet["content"] = message
            if data.get("done"):
                packet["done"] = True
            yield packet

    def warm_model(self, model_name: Optional[str] = None) -> None:
        """Issue a lightweight keep-alive request so Ollama keeps the model in RAM."""
        if not self._keep_alive:
            return
        url = f"{self._base}/api/generate"
        target_model = model_name or self._model
        payload = {
            "model": target_model,
            "prompt": "keepalive",
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"num_predict": 0},
        }
        with contextlib.suppress(Exception):
            requests.post(url, json=payload, timeout=10)

    def list_models(self) -> List[Dict[str, Any]]:
        """Return available models from Ollama /api/tags."""
        url = f"{self._base}/api/tags"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        models = data.get("models") or []
        result: List[Dict[str, Any]] = []

        def fetch_capabilities_via_show(model_name: str) -> Optional[List[str]]:
            try:
                show_resp = requests.post(f"{self._base}/api/show", json={"model": model_name}, timeout=10)
                show_resp.raise_for_status()
                show_data = show_resp.json() or {}
                caps = show_data.get("capabilities")
                if isinstance(caps, list):
                    return [c for c in caps if isinstance(c, str)]
            except Exception:
                return None
            return None

        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            details = item.get("details") or {}
            if "capabilities" not in details:
                caps_list = fetch_capabilities_via_show(name)
                if caps_list:
                    details = {**details, "capabilities": caps_list}
            caps = infer_capabilities(name, details)
            result.append({"name": name, "details": details, "capabilities": caps})
        return result


# === NKN Bridge =============================================================

BRIDGE_JS_SOURCE = """
'use strict';
const nkn = require('nkn-sdk');
const readline = require('readline');

const SEED_HEX = (process.env.NKN_SEED_HEX || '').toLowerCase().replace(/^0x/, '');
const IDENTIFIER = process.env.NKN_IDENTIFIER || 'inference-relay';
const SUBCLIENTS = parseInt(process.env.NKN_NUM_SUBCLIENTS || '8', 10) || 8;
const SEED_WS = (process.env.NKN_BRIDGE_SEED_WS || '').split(',').map(s => s.trim()).filter(Boolean);
const TLS_ENV = process.env.NKN_TLS;
const TLS_ENABLED = TLS_ENV === undefined ? undefined : TLS_ENV === 'true';

function emit(obj) {
  try { process.stdout.write(JSON.stringify(obj) + '\\n'); }
  catch (err) { /* swallow */ }
}

if (!/^[0-9a-f]{64}$/.test(SEED_HEX)) {
  emit({ type: 'fatal', error: 'NKN_SEED_HEX missing or invalid' });
  process.exit(1);
}

const client = new nkn.MultiClient({
  seed: SEED_HEX,
  identifier: IDENTIFIER,
  numSubClients: SUBCLIENTS,
  reconnectIntervalMin: 1000,
  reconnectIntervalMax: 8000,
  responseTimeout: 15000,
  msgHoldingSeconds: 60,
  encrypt: true,
  ...(TLS_ENABLED === undefined ? {} : { tls: TLS_ENABLED }),
  msgCacheExpiration: 300000,
  seedWsAddr: SEED_WS.length ? SEED_WS : undefined,
  wsConnHeartbeatTimeout: 120000,
});

const onConnect = () => {
  emit({ type: 'ready', address: String(client.addr || '') });
};

if (typeof client.onConnect === 'function') {
  client.onConnect(onConnect);
} else {
  client.on('connect', onConnect);
}

const onConnectFailed = (err) => {
  emit({ type: 'fatal', error: `connect failed: ${String(err && err.message || err || 'unknown')}` });
  process.exit(2);
};

if (typeof client.onConnectFailed === 'function') {
  client.onConnectFailed(onConnectFailed);
} else {
  client.on('connectFailed', onConnectFailed);
}

const onWsError = (clientId, ev) => {
  let cid = 'unknown';
  if (typeof clientId === 'string') {
    cid = clientId;
  } else if (clientId && typeof clientId === 'object') {
    cid = clientId.id || clientId.addr || clientId.address || clientId.name || undefined;
    if (!cid) {
      try { cid = JSON.stringify(clientId); } catch (_) { cid = 'unknown'; }
    }
  }
  const detail = ev && ev.message ? ev.message : ev && ev.type ? ev.type : ev;
  emit({ type: 'status', level: 'ws_error', message: `ws error on ${cid}: ${String(detail || '')}` });
};

if (typeof client.onWsError === 'function') {
  client.onWsError(onWsError);
} else {
  client.on('ws_error', onWsError);
}

client.onMessage(({src, payload, payloadType}) => {
  let text = '';
  if (Buffer.isBuffer(payload)) {
    text = payload.toString('utf8');
  } else if (typeof payload === 'string') {
    text = payload;
  } else if (payload && payload.payload) {
    const inner = payload.payload;
    text = Buffer.isBuffer(inner) ? inner.toString('utf8') : String(inner);
  } else {
    text = String(payload || '');
  }
  let parsed = null;
  try { parsed = JSON.parse(text); }
  catch (err) {
    parsed = { event: 'relay.raw', raw: text };
  }
  emit({ type: 'message', src: String(src || ''), payload: parsed });
  // Return false to prevent automatic ACK - we'll send responses manually for streaming
  return false;
});

client.on('error', (err) => {
  emit({ type: 'status', level: 'error', message: String(err && err.message || err) });
});

client.on('close', () => {
  emit({ type: 'status', level: 'close' });
  process.exit(2);
});

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  let cmd;
  try { cmd = JSON.parse(line); }
  catch (err) { return; }
  if (!cmd || cmd.type !== 'send') return;
  const body = (cmd.data !== undefined) ? cmd.data : cmd.payload;
  if (!cmd.to || body === undefined) return;

  // Use noReply: true for one-way messages (responses, notifications)
  // Use noReply: false (default) for request-reply pattern
  const opts = cmd.opts !== undefined ? cmd.opts : { noReply: true };

  client.send(String(cmd.to), JSON.stringify(body), opts).catch((err) => {
    emit({ type: 'status', level: 'send_error', message: String(err && err.message || err) });
  });
});
"""


class NKNBridge:
    def __init__(self, workdir: Path) -> None:
        self._dir = workdir
        self._pkg = self._dir / "package.json"
        self._js = self._dir / NKN_JS_NAME
        self._seed_file = self._dir / "seed_hex.txt"
        self._proc: Optional[subprocess.Popen[str]] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._send_lock = threading.Lock()
        self._listeners: List[Callable[[str, Dict[str, Any]], None]] = []
        self.address: Optional[str] = None
        self.public_key: Optional[str] = None
        self.enabled = False
        self._ready_event = threading.Event()
        self._start_lock = threading.Lock()
        self._should_run = True
        self._watchdog_thread: Optional[threading.Thread] = None

        self._node = shutil.which("node")
        self._npm = shutil.which("npm")
        if not self._node or not self._npm:
            print("[nkn] node and npm are required for the bridge; skipping startup", file=sys.stderr)
            return
        self.enabled = True
        self._ensure_bridge_artifacts()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    # ------------------------------------------------------------------ utils
    def _ensure_bridge_artifacts(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        if not self._pkg.exists():
            subprocess.check_call([self._npm, "init", "-y"], cwd=self._dir)
        node_modules = self._dir / "node_modules"
        target_pkg = node_modules / "nkn-sdk"
        if not target_pkg.exists():
            subprocess.check_call([self._npm, "install", f"nkn-sdk@{NKN_PACKAGE_VERSION}"], cwd=self._dir)
        if not self._js.exists() or self._js.read_text(encoding="utf-8") != BRIDGE_JS_SOURCE:
            self._js.write_text(BRIDGE_JS_SOURCE, encoding="utf-8")
        if not self._seed_file.exists():
            self._seed_file.write_text(secrets.token_hex(32))

    def _env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.setdefault("NKN_IDENTIFIER", NKN_IDENTIFIER)
        env.setdefault("NKN_NUM_SUBCLIENTS", str(NKN_NUM_SUBCLIENTS))
        env["NKN_SEED_HEX"] = self._seed_file.read_text().strip()
        return env

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if not self.enabled:
            return
        with self._start_lock:
            if self._proc and self._proc.poll() is None:
                return
            self._ready_event.clear()
            try:
                self._proc = subprocess.Popen(
                    [self._node, str(self._js)],
                    cwd=self._dir,
                    env=self._env(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                print("[nkn] bridge process started")
            except Exception as exc:  # pragma: no cover - startup log
                print(f"[nkn] failed to start bridge: {exc}", file=sys.stderr)
                return
            self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
            self._stdout_thread.start()
            self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self._stderr_thread.start()
            state = "unknown"
            if self._proc and self._proc.stdin:
                try:
                    self._proc.stdin.write(json.dumps({"type": "handshake"}) + "\n")
                    self._proc.stdin.flush()
                    state = "handshake"  # best effort marker
                except Exception:
                    state = "handshake_failed"
            print(f"[nkn] bridge state={state}")
            self._start_time = time.time()

    def wait_ready(self, timeout: Optional[float] = None) -> Optional[str]:
        if self._ready_event.wait(timeout):
            return self.address
        return None

    def stop(self) -> None:
        self._should_run = False
        if not self._proc:
            return
        with contextlib.suppress(Exception):
            self._proc.terminate()
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=3)
        self._proc = None
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2)

    # ----------------------------------------------------------------- comms
    def register_listener(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self._listeners.append(callback)

    def send(self, to_addr: str, payload: Dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("NKN bridge is not running")
        line = json.dumps({"type": "send", "to": to_addr, "data": payload})
        with self._send_lock:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()

    # ----------------------------------------------------------------- IO loops
    def _read_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for raw in self._proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "ready":
                self.address = event.get("address") or self.address
                self.public_key = self._extract_pubkey(self.address)
                self._ready_event.set()
                print(f"[nkn] ready at {self.address}")
                if self.public_key:
                    print(f"[nkn] pubkey {self.public_key}")
            elif etype == "message":
                src = event.get("src") or ""
                payload = event.get("payload") or {}
                for cb in list(self._listeners):
                    try:
                        cb(str(src), payload)
                    except Exception as exc:  # pragma: no cover - logging only
                        print(f"[nkn] listener error: {exc}", file=sys.stderr)
            elif etype == "status":
                level = event.get("level") or "info"
                msg = event.get("message") or ""
                print(f"[nkn] {level}: {msg}")
            elif etype == "fatal":
                print(f"[nkn] fatal: {event.get('error')}", file=sys.stderr)

    def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            text = line.strip()
            if text:
                print(f"[nkn-node] {text}", file=sys.stderr)

    @staticmethod
    def _extract_pubkey(address: Optional[str]) -> Optional[str]:
        if not address:
            return None
        addr = str(address).strip()
        if not addr:
            return None
        if "." not in addr:
            return addr
        parts = [p for p in addr.split(".") if p]
        if not parts:
            return None
        return parts[-1]

    def _watchdog_loop(self) -> None:
        while self._should_run:
            if not self.enabled:
                time.sleep(5)
                continue
            proc = self._proc
            running = proc and proc.poll() is None
            if not running:
                print("[nkn] bridge offline, attempting restart…")
                self.start()
                time.sleep(2)
                continue
            # health check: ensure stdout still active by sending ping
            try:
                if proc and proc.stdin:
                    proc.stdin.write(json.dumps({"type": "ping", "ts": time.time()}) + "\n")
                    proc.stdin.flush()
            except Exception:
                print("[nkn] ping failed, restarting bridge…")
                with contextlib.suppress(Exception):
                    proc.terminate()
                continue
            time.sleep(5)


# === NKN relay server =======================================================
# Cache completed assistant replies so we can replay duplicates without rerunning inference.
ASSISTANT_RESPONSE_CACHE_TTL_S = 300

NVIDIA_SMI_FIELDS = (
    "index",
    "name",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)
NVIDIA_SMI_QUERY = ",".join(NVIDIA_SMI_FIELDS)
NVIDIA_SMI_TIMEOUT_S = 5

class NKNRelayServer:
    def __init__(
        self,
        bridge: NKNBridge,
        ollama: OllamaClient,
        prompts: SystemPromptLoader,
        db: Database,
        sessions: SessionManager,
    ) -> None:
        self._bridge = bridge
        self._ollama = ollama
        self._prompts = prompts
        self._db = db
        self._sessions = sessions
        self._active: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._known_clients: set[str] = set()
        self._active_assistant_uuids: set[str] = set()
        self._cached_assistant_replies: Dict[str, Tuple[str, int, float]] = {}
        self._model_warmed: Dict[str, bool] = {}
        self._default_model = OLLAMA_MODEL_NAME
        self._hardware_stop = threading.Event()
        self._pending_images: Dict[str, Dict[str, Any]] = {}  # upload_id -> {b64, mime, name, ts, chunks}
        self._nvidia_smi_path = shutil.which("nvidia-smi")
        if not self._nvidia_smi_path:
            print("[hardware] nvidia-smi not found; GPU stats disabled")
        self._hardware_thread = threading.Thread(target=self._hardware_loop, daemon=True)
        self._hardware_thread.start()
        bridge.register_listener(self._handle_message)

    def _prune_expired_uploads(self) -> int:
        """Remove stale pending image uploads to prevent unbounded growth."""
        now = time.time()
        cutoff = now - PENDING_IMAGE_TTL_S
        removed = 0
        for key, entry in list(self._pending_images.items()):
            if entry.get("ts", 0) < cutoff:
                self._pending_images.pop(key, None)
                removed += 1
        if removed:
            print(f"[image] Pruned {removed} expired uploads")
        return removed

    def _send(self, to_addr: str, payload: Dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            self._bridge.send(to_addr, payload)

    def _try_mark_assistant_uuid(self, uuid: Optional[str]) -> bool:
        if not uuid:
            return True
        with self._lock:
            if uuid in self._active_assistant_uuids:
                return False
            self._active_assistant_uuids.add(uuid)
            return True

    def _finalize_assistant_uuid(self, uuid: Optional[str], content: Optional[str], seq: Optional[int]) -> None:
        if not uuid:
            return
        with self._lock:
            self._active_assistant_uuids.discard(uuid)
            if content is None:
                return
            now = time.time()
            self._cached_assistant_replies[uuid] = (content, seq or 0, now)
            cutoff = now - ASSISTANT_RESPONSE_CACHE_TTL_S
            for key, (_, _, ts) in list(self._cached_assistant_replies.items()):
                if ts < cutoff:
                    del self._cached_assistant_replies[key]

    def _get_cached_assistant_response(self, uuid: Optional[str]) -> Optional[Tuple[str, int, float]]:
        if not uuid:
            return None
        with self._lock:
            cached = self._cached_assistant_replies.get(uuid)
            if not cached:
                return None
            return cached

    def _replay_cached_assistant_response(self, src: str, req_id: Optional[str], cached: Tuple[str, int, float], model_name: Optional[str] = None) -> None:
        if not req_id or not cached:
            return
        content, seq, _ = cached
        seq = seq or 0
        payload_content = content or ''
        selected_model = model_name or self._default_model
        self._reply(src, req_id, "chat.ack", model=selected_model)
        self._send_model_status(
            src,
            req_id,
            "idle",
            total_seq=seq,
            chars=len(payload_content),
            model_name=selected_model,
        )
        self._send(src, {
            "event": "chat.done",
            "id": req_id,
            "done": True,
            "total_seq": seq,
            "final_content": payload_content,
        })

    def _send_model_status(
        self,
        to_addr: str,
        req_id: Optional[str],
        phase: str,
        model_name: Optional[str] = None,
        **extra: Any,
    ) -> None:
        selected_model = model_name or self._default_model
        payload: Dict[str, Any] = {
            "event": "model.status",
            "phase": phase,
            "model": selected_model,
            "timestamp": time.time(),
        }
        if req_id is not None:
            payload["id"] = req_id
        payload.update(extra)
        self._send(to_addr, payload)

    def _reply(self, src: str, req_id: Optional[str], event: str, **extra: Any) -> None:
        data = {"event": event}
        if req_id is not None:
            data["id"] = req_id
        data.update(extra)
        self._send(src, data)

    def mark_model_warmed(self, model_name: Optional[str] = None) -> None:
        """Mark a model as warmed so future requests can skip the loading phase."""
        model_key = model_name or self._default_model
        self._model_warmed[model_key] = True

    def _handle_message(self, src: str, payload: Dict[str, Any]) -> None:
        event = payload.get("event")
        if src and src not in self._known_clients:
            self._known_clients.add(src)
            print(f"[nkn] new client {src} (total={len(self._known_clients)})")
        if event == "image.upload":
            upload_id = payload.get("upload_id")
            try:
                idx = int(payload.get("i", -1))
                total = int(payload.get("t", -1))
            except Exception:
                idx, total = -1, -1
            data_len = len(payload.get("data") or "")
            print(f"[image] recv chunk upload_id={upload_id} idx={idx+1}/{total} data_len={data_len}")
        if event == "relay.info":
            self._reply(
                src,
                payload.get("id"),
                "relay.info",
                model=self._default_model,
                address=self._bridge.address,
                ts=time.time(),
                pubkey=self._bridge.public_key,
                peer_count=len(self._known_clients),
            )
            return
        if event == "auth.register":
            self._handle_register(src, payload)
            return
        if event == "auth.login":
            self._handle_login(src, payload)
            return
        if event == "auth.logout":
            self._handle_logout(src, payload)
            return
        if event == "auth.status":
            self._handle_status(src, payload)
            return
        if event == "chat.begin":
            self._handle_chat_begin(src, payload)
            return
        if event == "chat.refresh":
            self._handle_chat_refresh(src, payload)
            return
        if event == "session.create":
            self._handle_session_create(src, payload)
            return
        if event == "session.switch":
            self._handle_session_switch(src, payload)
            return
        if event == "session.delete":
            self._handle_session_delete(src, payload)
            return
        if event == "session.model.set":
            self._handle_session_model_set(src, payload)
            return
        if event == "models.list":
            self._handle_models_list(src, payload)
            return
        if event == "user.prefs.set":
            self._handle_prefs_set(src, payload)
            return
        if event == "user.prefs.get":
            self._handle_prefs_get(src, payload)
            return
        if event == "image.upload":
            self._handle_image_upload(src, payload)
            return
        if event == "chat.abort":
            req_id = payload.get("id")
            if req_id:
                self._cancel(req_id)

    def _hardware_loop(self) -> None:
        while not self._hardware_stop.is_set():
            stats = self._collect_hardware_stats()
            if stats:
                for client in list(self._known_clients):
                    self._send(client, {"event": "hardware.stats", "stats": stats})
            time.sleep(5)

    def _collect_hardware_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"model_ready": self._model_warmed.get(self._default_model, False), "ts": time.time()}
        devices: List[Dict[str, Any]] = []
        stats["devices"] = devices
        stats["available_devices"] = 0
        if not self._nvidia_smi_path:
            stats["gpu_available"] = False
            stats["error"] = "nvidia-smi not found"
            return stats

        def _parse_float(value: Optional[str]) -> Optional[float]:
            if value is None:
                return None
            text = str(value).strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None

        def _parse_int(value: Optional[str]) -> Optional[int]:
            parsed = _parse_float(value)
            if parsed is None:
                return None
            return int(parsed)

        try:
            output = subprocess.check_output(
                [self._nvidia_smi_path, f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
                stderr=subprocess.STDOUT,
                timeout=NVIDIA_SMI_TIMEOUT_S,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            stats["gpu_available"] = False
            stats["error"] = f"nvidia-smi timeout: {exc}"
            return stats
        except subprocess.CalledProcessError as exc:
            stats["gpu_available"] = False
            stats["error"] = exc.output.strip() if exc.output else str(exc)
            return stats
        except FileNotFoundError:
            stats["gpu_available"] = False
            stats["error"] = "nvidia-smi not found"
            return stats

        try:
            for row in csv.reader(output.splitlines(), skipinitialspace=True):
                if not row or len(row) < len(NVIDIA_SMI_FIELDS):
                    continue
                idx = _parse_int(row[0])
                name = row[1].strip() if row[1] else ""
                uuid = row[2].strip() if row[2] else ""
                util_gpu = _parse_int(row[3])
                util_mem = _parse_int(row[4])
                mem_used = _parse_int(row[5])
                mem_total = _parse_int(row[6])
                temperature = _parse_float(row[7])
                power_draw = _parse_float(row[8])
                devices.append({
                    "index": idx,
                    "name": name,
                    "uuid": uuid,
                    "gpu_util_percent": util_gpu,
                    "mem_util_percent": util_mem,
                    "mem_used_mb": mem_used,
                    "mem_total_mb": mem_total,
                    "temperature_c": temperature,
                    "power_draw_w": power_draw,
                })
        except Exception as exc:
            stats["gpu_available"] = False
            stats["error"] = str(exc)
            return stats

        stats["devices"] = devices
        stats["available_devices"] = len(devices)
        if not devices:
            stats["gpu_available"] = False
            stats.setdefault("error", "no GPUs detected")
            return stats

        primary = devices[0]
        stats.update({
            "gpu_available": True,
            "gpu_count": len(devices),
            "gpu_util_percent": primary.get("gpu_util_percent"),
            "mem_util_percent": primary.get("mem_util_percent"),
            "mem_used_mb": primary.get("mem_used_mb"),
            "mem_total_mb": primary.get("mem_total_mb"),
            "temperature_c": primary.get("temperature_c"),
            "power_draw_w": primary.get("power_draw_w"),
        })
        return stats

    def _handle_register(self, src: str, payload: Dict[str, Any]) -> None:
        req_id = payload.get("id")
        username = (payload.get("username") or "").strip()
        password = payload.get("password") or ""
        if not username or not password:
            self._reply(src, req_id, "auth.register.error", message="username and password required")
            print(f"[auth] Register attempt with missing credentials from {src}")
            return
        if len(username) > 64:
            self._reply(src, req_id, "auth.register.error", message="username too long")
            return
        if len(password) < 6:
            self._reply(src, req_id, "auth.register.error", message="password must be at least 6 characters")
            return
        user_id = self._db.create_user(username, password)
        if user_id is None:
            self._reply(src, req_id, "auth.register.error", message="username already exists")
            print(f"[auth] Failed registration - username '{username}' already exists (from {src})")
            return
        self._db.update_last_nkn_addr(user_id, src)
        print(f"[auth] Registered new user '{username}' (user_id={user_id}) from {src}")
        self._reply(src, req_id, "auth.register.ok", username=username, user_id=user_id)

    def _handle_login(self, src: str, payload: Dict[str, Any]) -> None:
        req_id = payload.get("id")
        username = (payload.get("username") or "").strip()
        password = payload.get("password") or ""
        if not username or not password:
            self._reply(src, req_id, "auth.login.error", message="username and password required")
            print(f"[auth] Login attempt with missing credentials from {src}")
            return
        auth = self._db.authenticate_user(username, password)
        if auth is None:
            self._reply(src, req_id, "auth.login.error", message="invalid credentials")
            print(f"[auth] Failed login attempt for username '{username}' from {src}")
            return
        user_id, uname, preferences = auth

        # Check if user is already logged in from a different address
        existing_session = self._find_user_session(user_id)
        if existing_session and existing_session != src:
            print(f"[auth] User '{uname}' switching from {existing_session} to {src}")
            self._sessions.logout(existing_session)

        # Get or create default chat session
        session_id = self._db.get_or_create_default_session(user_id, default_model=self._default_model)

        # Create new session bound to this NKN address
        self._sessions.login(src, user_id, uname, session_id)
        self._db.update_last_nkn_addr(user_id, src)
        print(f"[auth] Login OK for '{uname}' (user_id={user_id}) from {src}")

        # Get sessions and messages for current session
        sessions = self._db.get_sessions(user_id)
        messages = self._db.get_recent_messages(user_id=user_id, session_id=session_id, limit=MAX_CONTEXT_MESSAGES)
        current_model = self._db.get_session_model(session_id)
        preferences = {}
        if auth and len(auth) > 2:
            preferences = auth[2] or {}

        self._reply(
            src,
            req_id,
            "auth.login.ok",
            username=uname,
            user_id=user_id,
            current_session_id=session_id,
            current_session_model=current_model,
            sessions=sessions,
            messages=messages,
            nkn_address=src,
            preferences=preferences or {},
        )

    def _find_user_session(self, user_id: int) -> Optional[str]:
        """Find the NKN address associated with a user's active session."""
        with self._sessions._lock:
            for addr, (uid, _, _, _) in self._sessions._sessions.items():
                if uid == user_id:
                    return addr
        return None

    def _handle_logout(self, src: str, payload: Dict[str, Any]) -> None:
        req_id = payload.get("id")
        self._sessions.logout(src)
        self._reply(src, req_id, "auth.logout.ok")

    def _handle_status(self, src: str, payload: Dict[str, Any]) -> None:
        req_id = payload.get("id")
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "auth.status", authenticated=False)
        else:
            user_id, username, session_id = session
            current_model = self._db.get_session_model(session_id)
            self._reply(
                src,
                req_id,
                "auth.status",
                authenticated=True,
                username=username,
                current_session_id=session_id,
                current_session_model=current_model,
            )

    def _handle_chat_refresh(self, src: str, payload: Dict[str, Any]) -> None:
        """Handle request to refresh conversation history from database."""
        req_id = payload.get("id")
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "chat.refresh.error", error="not_authenticated")
            return
        user_id, username, session_id = session

        # Refresh session timestamp on activity
        self._sessions.refresh(src)

        # Get sessions and messages for current session
        sessions = self._db.get_sessions(user_id)
        messages = self._db.get_recent_messages(user_id=user_id, session_id=session_id, limit=MAX_CONTEXT_MESSAGES)
        current_model = self._db.get_session_model(session_id)
        self._reply(
            src,
            req_id,
            "chat.refresh.ok",
            sessions=sessions,
            messages=messages,
            current_session_id=session_id,
            current_session_model=current_model,
            timestamp=time.time(),
        )

    def _handle_session_create(self, src: str, payload: Dict[str, Any]) -> None:
        """Create a new chat session and switch to it."""
        req_id = payload.get("id")
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "session.create.error", error="not_authenticated")
            return
        user_id, username, current_session_id = session

        # Create new session
        requested_model = (payload.get("model") or "").strip() or self._db.get_session_model(current_session_id)
        new_session_id = self._db.create_session(user_id, "New Chat", model=requested_model)

        # Switch to the new session
        self._sessions.set_current_session(src, new_session_id)

        # Get updated sessions list
        sessions = self._db.get_sessions(user_id)
        messages = self._db.get_recent_messages(user_id=user_id, session_id=new_session_id, limit=MAX_CONTEXT_MESSAGES)

        self._reply(
            src,
            req_id,
            "session.create.ok",
            session_id=new_session_id,
            sessions=sessions,
            messages=messages,
            current_session_id=new_session_id,
            current_session_model=requested_model,
        )
        print(f"[session] Created new session {new_session_id} for user_id={user_id}")

    def _handle_session_switch(self, src: str, payload: Dict[str, Any]) -> None:
        """Switch to a different chat session."""
        req_id = payload.get("id")
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "session.switch.error", error="not_authenticated")
            return
        user_id, username, _ = session

        target_session_id = payload.get("session_id")
        if not target_session_id:
            self._reply(src, req_id, "session.switch.error", error="session_id required")
            return

        # Verify session belongs to user
        sessions = self._db.get_sessions(user_id)
        if not any(s["id"] == target_session_id for s in sessions):
            self._reply(src, req_id, "session.switch.error", error="session not found")
            return

        # Switch to the target session
        self._sessions.set_current_session(src, target_session_id)

        # Get messages for the target session
        messages = self._db.get_recent_messages(user_id=user_id, session_id=target_session_id, limit=MAX_CONTEXT_MESSAGES)
        current_model = self._db.get_session_model(target_session_id)

        self._reply(
            src,
            req_id,
            "session.switch.ok",
            session_id=target_session_id,
            messages=messages,
            current_session_id=target_session_id,
            current_session_model=current_model,
            sessions=sessions,
        )
        print(f"[session] User {username} switched to session {target_session_id}")

    def _handle_session_delete(self, src: str, payload: Dict[str, Any]) -> None:
        """Delete a chat session."""
        req_id = payload.get("id")
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "session.delete.error", error="not_authenticated")
            return
        user_id, username, current_session_id = session

        target_session_id = payload.get("session_id")
        if not target_session_id:
            self._reply(src, req_id, "session.delete.error", error="session_id required")
            return

        # Delete the session
        success = self._db.delete_session(target_session_id, user_id)
        if not success:
            self._reply(src, req_id, "session.delete.error", error="session not found")
            return

        # If we deleted the current session, switch to another one
        new_current_session_id = current_session_id
        if target_session_id == current_session_id:
            new_current_session_id = self._db.get_or_create_default_session(user_id, default_model=self._default_model)
            self._sessions.set_current_session(src, new_current_session_id)

        # Get updated sessions list and messages
        sessions = self._db.get_sessions(user_id)
        messages = self._db.get_recent_messages(user_id=user_id, session_id=new_current_session_id, limit=MAX_CONTEXT_MESSAGES)
        current_model = self._db.get_session_model(new_current_session_id)

        self._reply(
            src,
            req_id,
            "session.delete.ok",
            deleted_session_id=target_session_id,
            sessions=sessions,
            messages=messages,
            current_session_id=new_current_session_id,
            current_session_model=current_model,
        )
        print(f"[session] User {username} deleted session {target_session_id}")

    def _handle_session_model_set(self, src: str, payload: Dict[str, Any]) -> None:
        """Update the model used by a chat session."""
        req_id = payload.get("id")
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "session.model.set.error", error="not_authenticated")
            return
        user_id, username, current_session_id = session
        target_session_id = payload.get("session_id") or current_session_id
        model = (payload.get("model") or "").strip()
        if not model:
            self._reply(src, req_id, "session.model.set.error", error="model required")
            return

        sessions = self._db.get_sessions(user_id)
        if not any(s["id"] == target_session_id for s in sessions):
            self._reply(src, req_id, "session.model.set.error", error="session not found")
            return

        self._db.update_session_model(target_session_id, model)
        updated_sessions = self._db.get_sessions(user_id)
        self._reply(
            src,
            req_id,
            "session.model.set.ok",
            session_id=target_session_id,
            model=model,
            sessions=updated_sessions,
            current_session_id=target_session_id,
            current_session_model=model,
        )
        print(f"[session] User {username} set model '{model}' for session {target_session_id}")

    def _handle_models_list(self, src: str, payload: Dict[str, Any]) -> None:
        """List available models from Ollama."""
        req_id = payload.get("id")
        try:
            models = self._ollama.list_models()
        except Exception as exc:
            self._reply(src, req_id, "models.list.error", message=str(exc))
            return
        self._reply(src, req_id, "models.list", models=models)

    def _handle_prefs_set(self, src: str, payload: Dict[str, Any]) -> None:
        """Persist user preferences (arbitrary JSON)."""
        req_id = payload.get("id")
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "user.prefs.set.error", error="not_authenticated")
            return
        user_id, _, _ = session
        prefs = payload.get("preferences")
        if not isinstance(prefs, dict):
            self._reply(src, req_id, "user.prefs.set.error", error="preferences must be object")
            return
        self._db.set_user_preferences(user_id, prefs)
        self._reply(src, req_id, "user.prefs.set.ok", preferences=prefs)

    def _handle_prefs_get(self, src: str, payload: Dict[str, Any]) -> None:
        """Return stored user preferences."""
        req_id = payload.get("id")
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "user.prefs.get.error", error="not_authenticated")
            return
        user_id, _, _ = session
        prefs = self._db.get_user_preferences(user_id)
        self._reply(src, req_id, "user.prefs.get.ok", preferences=prefs)

    def _handle_image_upload(self, src: str, payload: Dict[str, Any]) -> None:
        """Accept chunked image uploads over DM prior to chat.begin."""
        req_id = payload.get("id")
        upload_id = payload.get("upload_id") or ""
        if not upload_id:
            self._reply(src, req_id, "image.upload.error", error="upload_id required")
            return
        self._prune_expired_uploads()
        try:
            idx = int(payload.get("i", 0))
            total = int(payload.get("t", 0))
        except Exception:
            self._reply(src, req_id, "image.upload.error", error="invalid indices")
            return
        if total <= 0 or idx < 0 or idx >= total:
            self._reply(src, req_id, "image.upload.error", error="invalid chunk index")
            return
        data = payload.get("data")
        if not isinstance(data, str) or not data:
            self._reply(src, req_id, "image.upload.error", error="missing data")
            return
        if len(data) > IMAGE_CHUNK_MAX_BYTES * 2:
            self._reply(src, req_id, "image.upload.error", error="chunk too large")
            return

        entry = self._pending_images.get(upload_id)
        now = time.time()
        if entry is None:
            entry = {"chunks": [None] * total, "received": 0, "ts": now, "mime": payload.get("mime") or "", "name": payload.get("name") or ""}
            self._pending_images[upload_id] = entry
        elif entry.get("ts", 0) < now - PENDING_IMAGE_TTL_S:
            self._pending_images.pop(upload_id, None)
            self._reply(src, req_id, "image.upload.error", error="upload expired; please re-upload")
            return
        if idx >= len(entry["chunks"]):
            self._reply(src, req_id, "image.upload.error", error="chunk index out of range")
            return
        if entry["chunks"][idx] is None:
            entry["received"] += 1
        entry["chunks"][idx] = data
        entry["mime"] = payload.get("mime") or entry["mime"]
        entry["name"] = payload.get("name") or entry["name"]
        entry["ts"] = now
        complete = entry["received"] == len(entry["chunks"])
        print(
            f"[image] stored chunk upload_id={upload_id} idx={idx+1}/{total} recv={entry['received']}/{len(entry['chunks'])} complete={complete}"
        )
        self._reply(src, req_id, "image.upload.ack", upload_id=upload_id, received=entry["received"], total=len(entry["chunks"]), complete=complete)
        if complete:
            b64_data = "".join(c or "" for c in entry["chunks"])
            decoded = None
            try:
                decoded = base64.b64decode(b64_data, validate=True)
            except Exception:
                self._reply(src, None, "image.upload.error", error="invalid base64", upload_id=upload_id)
                self._pending_images.pop(upload_id, None)
                return
            if len(decoded) > IMAGE_TOTAL_MAX_BYTES:
                self._reply(src, None, "image.upload.error", error="image too large", upload_id=upload_id)
                self._pending_images.pop(upload_id, None)
                return
            entry["b64"] = b64_data
            print(f"[image] upload complete id={upload_id} bytes={len(decoded)} mime={entry['mime'] or 'unknown'}")

    def _handle_chat_begin(self, src: str, payload: Dict[str, Any]) -> None:
        req_id = payload.get("id") or f"dm-{int(time.time() * 1000)}"
        session = self._sessions.get(src)
        if session is None:
            self._reply(src, req_id, "chat.error", error="not_authenticated")
            print(f"[chat] Rejected unauthenticated request {req_id} from {src}")
            return
        user_id, username, session_id = session
        model_name = self._db.get_session_model(session_id)
        image_chunks = payload.get("image_chunks") or []
        image_upload_id = payload.get("image_upload_id") or ""
        image_mime = payload.get("image_mime") or ""
        image_thumb = payload.get("image_thumb") or ""
        thinking_enabled = bool(payload.get("thinking"))
        tools_payload = payload.get("tools") or []
        if not isinstance(tools_payload, list):
            tools_payload = []
        clean_tools: List[Dict[str, Any]] = []
        for entry in tools_payload:
            if not isinstance(entry, dict):
                continue
            fn = entry.get("function") or {}
            if entry.get("type") != "function" or not fn.get("name") or not isinstance(fn.get("parameters"), dict):
                continue
            clean_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": fn.get("name"),
                        "description": fn.get("description") or "",
                        "parameters": fn.get("parameters") or {},
                    },
                }
            )
        tools_payload = clean_tools
        self._prune_expired_uploads()

        # Refresh session timestamp on activity
        self._sessions.refresh(src)

        text = (payload.get("message") or "").strip()
        if not text:
            self._reply(src, req_id, "chat.error", error="empty_message")
            return
        text = text[:MAX_MESSAGE_LENGTH]
        image_b64 = ""
        if image_upload_id:
            pending = self._pending_images.pop(image_upload_id, None)
            cutoff = time.time() - PENDING_IMAGE_TTL_S
            if not pending or not pending.get("b64"):
                self._reply(src, req_id, "chat.error", error="image upload not found")
                return
            if pending.get("ts", 0) < cutoff:
                self._reply(src, req_id, "chat.error", error="image upload expired; please re-upload")
                return
            image_b64 = pending["b64"]
            image_mime = pending.get("mime") or image_mime
            print(f"[chat] Image attached from upload {image_upload_id} for req={req_id} user={username} size_b64={len(image_b64)} mime={image_mime or 'unknown'}")
            self._reply(src, req_id, "image.ready", upload_id=image_upload_id)
        elif image_chunks:
            image_b64, img_err = assemble_image_chunks(image_chunks, image_mime)
            if img_err:
                self._reply(src, req_id, "chat.error", error=img_err)
                print(f"[chat] Rejecting image payload for req={req_id}: {img_err}")
                return
            print(f"[chat] Image attached for req={req_id} user={username} size_b64={len(image_b64)} chunks={len(image_chunks)} mime={image_mime or 'unknown'}")

        # Extract UUIDs for duplicate prevention
        user_msg_uuid = payload.get("user_uuid")
        assistant_msg_uuid = payload.get("assistant_uuid")

        if assistant_msg_uuid:
            cached = self._get_cached_assistant_response(assistant_msg_uuid)
            if cached:
                print(f"[chat] Duplicate request {req_id} for uuid={assistant_msg_uuid} (cached replay)")
                self._replay_cached_assistant_response(src, req_id, cached, model_name=model_name)
                return
            existing_msg = self._db.get_message_by_uuid(assistant_msg_uuid)
            if existing_msg and existing_msg["role"] == "assistant" and existing_msg["is_complete"]:
                print(f"[chat] Duplicate request {req_id} for uuid={assistant_msg_uuid} (DB replay)")
                self._finalize_assistant_uuid(assistant_msg_uuid, existing_msg["content"], existing_msg["last_seq"])
                cached = self._get_cached_assistant_response(assistant_msg_uuid)
                if cached:
                    self._replay_cached_assistant_response(src, req_id, cached, model_name=model_name)
                    return
            if not self._try_mark_assistant_uuid(assistant_msg_uuid):
                print(f"[chat] Duplicate in-flight request {req_id} for uuid={assistant_msg_uuid}")
                self._reply(src, req_id, "chat.ack", model=model_name)
                return

        print(
            f"[chat] Start req={req_id} user={username} (user_id={user_id}) session={session_id} src={src} chars={len(text)} user_uuid={user_msg_uuid}"
        )
        threading.Thread(
            target=self._serve_chat,
            args=(src, req_id, user_id, session_id, text, user_msg_uuid, assistant_msg_uuid, model_name, image_b64, image_mime, image_thumb, thinking_enabled, tools_payload),
            daemon=True,
        ).start()

    def _cancel(self, req_id: str) -> None:
        with self._lock:
            evt = self._active.get(req_id)
            if evt:
                evt.set()

    def _serve_chat(self, src: str, req_id: str, user_id: int, session_id: int, message_text: str, user_msg_uuid: str = None, assistant_msg_uuid: str = None, model_name: Optional[str] = None, image_b64: str = "", image_mime: str = "", image_thumb: str = "", thinking: bool = False, tools: Optional[List[Dict[str, Any]]] = None) -> None:
        cancel_flag = threading.Event()
        with self._lock:
            self._active[req_id] = cancel_flag
        model_name = model_name or self._db.get_session_model(session_id)
        history_limit = max(0, MAX_CONTEXT_MESSAGES - 1)
        history = self._db.get_recent_messages(user_id=user_id, session_id=session_id, limit=history_limit)

        # Auto-generate session title from first message (if session title is still "New Chat")
        sessions = self._db.get_sessions(user_id)
        current_session = next((s for s in sessions if s["id"] == session_id), None)
        if current_session and current_session["title"] == "New Chat" and current_session["message_count"] == 0:
            # Generate title from first 50 chars of message
            title = message_text[:50].strip()
            if len(message_text) > 50:
                title += "..."
            self._db.update_session_title(session_id, title)

        self._db.append_message(user_id, session_id, "user", message_text, uuid=user_msg_uuid, model=model_name, image_thumb=image_thumb)
        system_prompt = self._prompts.get_prompt()
        assembled: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        assembled.extend(history)
        user_message: Dict[str, Any] = {"role": "user", "content": message_text}
        if image_b64:
            user_message["images"] = [image_b64]
        if thinking:
            user_message["thinking"] = True
        assembled.append(user_message)
        self._send(src, {"event": "chat.ack", "id": req_id, "model": model_name})
        is_warmed = self._model_warmed.get(model_name, False)
        status_phase = "loading" if not is_warmed else "idle"
        status_detail = "Warming model…" if not is_warmed else "Model warmed"
        self._send_model_status(
            src,
            req_id,
            status_phase,
            context_messages=len(history),
            detail=status_detail,
            model_name=model_name,
        )
        self._model_warmed[model_name] = True

        # Streaming with batched deltas and progressive persistence
        start_time = time.time()
        full_reply_parts: List[str] = []
        seq_num = 0
        batch_buffer: List[str] = []
        batch_start_seq = 0
        last_send_time = time.time()
        last_persist_time = time.time()
        batch_max_chars = 100
        batch_max_delay_ms = 50
        persist_interval_ms = 500  # Save to DB every 500ms
        final_content = ""
        streaming_signaled = False
        status_chars = 0
        last_status_time = start_time
        first_chunk_time: Optional[float] = None
        status_payload_sent_once = False
        completed_successfully = False

        def emit_streaming_status(detail: Optional[str] = None) -> None:
            nonlocal status_chars, last_status_time, status_payload_sent_once
            now = time.time()
            elapsed = now - last_status_time
            tps = 0.0
            if elapsed > 0:
                tps = (status_chars / 4) / elapsed
            payload: Dict[str, Any] = {"tps": round(tps, 2)}
            if detail:
                payload["detail"] = detail
            if first_chunk_time and not status_payload_sent_once:
                payload["latency_ms"] = int((first_chunk_time - start_time) * 1000)
                status_payload_sent_once = True
            self._send_model_status(
                src,
                req_id,
                "streaming",
                model_name=model_name,
                **payload,
            )
            status_chars = 0
            last_status_time = now

        def flush_batch():
            nonlocal batch_buffer, batch_start_seq, last_send_time, seq_num, last_persist_time, status_chars
            if not batch_buffer:
                return
            batch_delta = "".join(batch_buffer)
            self._send(src, {
                "event": "chat.delta",
                "id": req_id,
                "delta": batch_delta,
                "seq": batch_start_seq,
                "batch_size": len(batch_buffer),
                "timestamp": time.time()
            })
            seq_num = batch_start_seq + len(batch_buffer)
            batch_buffer = []
            batch_start_seq = seq_num
            last_send_time = time.time()
            status_chars += len(batch_delta)
            emit_streaming_status()

            # Progressive persistence: save partial content periodically
            accumulated = "".join(full_reply_parts)
            time_since_persist = (time.time() - last_persist_time) * 1000
            if time_since_persist >= persist_interval_ms and accumulated and assistant_msg_uuid:
                self._db.upsert_partial_message(user_id, session_id, "assistant", accumulated, assistant_msg_uuid, seq_num, model=model_name)
                last_persist_time = time.time()
                print(f"[chat] Progressive save: {len(accumulated)} chars, seq={seq_num}")

        try:
            for packet in self._ollama.stream(assembled, model_name=model_name, tools=tools):
                if cancel_flag.is_set():
                    break
                if packet.get("keepalive"):
                    continue
                delta = packet.get("content")
                tool_calls = packet.get("tool_calls") or []
                if tool_calls:
                    self._send(src, {"event": "chat.tool_calls", "id": req_id, "tool_calls": tool_calls})
                done_flag = packet.get("done")
                if delta:
                    if not streaming_signaled:
                        streaming_signaled = True
                        self._send_model_status(
                            src,
                            req_id,
                            "streaming",
                            model_name=model_name,
                            started=time.time(),
                        )
                        first_chunk_time = first_chunk_time or time.time()
                    full_reply_parts.append(delta)
                    batch_buffer.append(delta)
                    # Calculate batch metrics
                    batch_chars = sum(len(d) for d in batch_buffer)
                    time_since_send = (time.time() - last_send_time) * 1000  # ms
                    # Flush batch if it's large enough or enough time has passed
                    if batch_chars >= batch_max_chars or time_since_send >= batch_max_delay_ms:
                        flush_batch()
                if done_flag:
                    break

            # Flush any remaining batched deltas
            flush_batch()

            final_content = "".join(full_reply_parts)
            self._send_model_status(
                src,
                req_id,
                "idle",
                total_seq=seq_num,
                chars=len(final_content),
                model_name=model_name,
            )
            if not streaming_signaled and final_content:
                self._send(src, {
                    "event": "chat.delta",
                    "id": req_id,
                    "delta": final_content,
                    "seq": 0,
                    "batch_size": 1,
                    "timestamp": time.time()
                })
            self._send(src, {
                "event": "chat.done",
                "id": req_id,
                "done": True,
                "total_seq": seq_num,
                "final_content": final_content
            })
            completed_successfully = True

        except Exception as exc:  # pragma: no cover - network path
            partial_content = "".join(full_reply_parts)
            self._send_model_status(
                src,
                req_id,
                "error",
                detail=str(exc),
                partial_chars=len(partial_content),
                model_name=model_name,
            )
            self._send_model_status(
                src,
                req_id,
                "idle",
                partial_chars=len(partial_content),
                model_name=model_name,
            )
            self._send(src, {
                "event": "chat.error",
                "id": req_id,
                "error": str(exc),
                "partial": len(full_reply_parts) > 0
            })
            print(f"[chat] error req={req_id} src={src}: {exc}")
            return
        finally:
            with self._lock:
                self._active.pop(req_id, None)
            self._finalize_assistant_uuid(
                assistant_msg_uuid,
                final_content if completed_successfully else None,
                seq_num if completed_successfully else None,
            )

        full_reply = final_content.strip()
        if full_reply and assistant_msg_uuid:
            # Mark the partial message as complete (or insert if it wasn't progressively saved)
            partial_state = self._db.get_partial_message(assistant_msg_uuid)
            if partial_state and not partial_state["is_complete"]:
                # Already exists as partial, mark complete
                self._db.mark_message_complete(assistant_msg_uuid, full_reply, seq_num, model=model_name)
                print(f"[chat] Marked partial message complete: {len(full_reply)} chars")
            else:
                # Wasn't saved progressively (very short response), insert normally
                self._db.append_message(user_id, session_id, "assistant", full_reply, uuid=assistant_msg_uuid, model=model_name)
        print(
            f"[chat] complete req={req_id} src={src} session={session_id} reply_chars={len(full_reply)} deltas={seq_num} assistant_uuid={assistant_msg_uuid}"
        )


# === Entrypoint =============================================================

db = Database(DATABASE_PATH)
sessions = SessionManager()
system_prompts = SystemPromptLoader(SYSTEM_PROMPT_FILE)
ollama_client = OllamaClient(
    base_url=OLLAMA_BASE_URL,
    endpoint=OLLAMA_CHAT_ENDPOINT,
    model_name=OLLAMA_MODEL_NAME,
    timeout_s=OLLAMA_TIMEOUT_S,
    keep_alive=OLLAMA_KEEP_ALIVE,
)


def start_ollama_keepalive(client: OllamaClient) -> Optional[threading.Event]:
    if not OLLAMA_KEEP_ALIVE or OLLAMA_KEEPALIVE_INTERVAL_S <= 0:
        return None

    stop_event = threading.Event()

    def _loop() -> None:
        client.warm_model()
        while not stop_event.wait(OLLAMA_KEEPALIVE_INTERVAL_S):
            client.warm_model()

    threading.Thread(target=_loop, name="ollama-keepalive", daemon=True).start()
    return stop_event


def main() -> None:
    print(f"[startup] {APP_TITLE}")
    print(f"[startup] Ollama target: {OLLAMA_BASE_URL} (model {OLLAMA_MODEL_NAME})")
    keepalive_stop = start_ollama_keepalive(ollama_client)
    nkn_bridge = NKNBridge(NKN_BRIDGE_DIR)
    if not nkn_bridge.enabled:
        print("[error] NKN bridge unavailable (requires node + npm)")
        sys.exit(1)

    nkn_bridge.start()
    atexit.register(nkn_bridge.stop)
    if keepalive_stop:
        atexit.register(keepalive_stop.set)
    relay_server = NKNRelayServer(nkn_bridge, ollama_client, system_prompts, db, sessions)
    if keepalive_stop:
        relay_server.mark_model_warmed(OLLAMA_MODEL_NAME)

    addr = nkn_bridge.wait_ready(timeout=60)
    if addr:
        print(f"[ready] Relay NKN address: {addr}")
        if nkn_bridge.public_key:
            print(f"[ready] Relay pubkey: {nkn_bridge.public_key}")
        print("[info] Paste this address into the web UI's relay box to start chatting.")
    else:
        print("[warn] Relay not ready yet; waiting in background…")

    print("[info] Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[shutdown] Stopping relay…")


if __name__ == "__main__" and _in_managed_venv():
    main()
