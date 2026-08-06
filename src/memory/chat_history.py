from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Сообщение из истории чата."""
    message_id: int
    sender_id: int
    sender_name: str
    text: str
    timestamp: datetime
    is_outgoing: bool


class ChatHistory:
    """Локальная история чатов в SQLite.

    Хранит все входящие и исходящие сообщения.
    Нужна для Bot API (бот не видит свои исходящие в Telegram).
    """

    def __init__(
        self,
        db_path: str | Path = "data/history.db",
        max_per_chat: int | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_per_chat = max_per_chat or int(os.getenv("HISTORY_MAX_PER_CHAT", "1000"))

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                sender_id INTEGER NOT NULL,
                sender_name TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp REAL NOT NULL,
                is_outgoing INTEGER NOT NULL DEFAULT 0,
                UNIQUE(chat_id, message_id, sender_id)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_id ON messages(chat_id, timestamp)
        """)
        self._conn.commit()

        count = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        logger.info("Chat history: %s (%d messages)", self._db_path, count)

    def add_message(
        self,
        chat_id: int,
        message_id: int,
        sender_id: int,
        sender_name: str,
        text: str,
        *,
        is_outgoing: bool = False,
        timestamp: datetime | None = None,
    ) -> None:
        """Записать сообщение в историю."""
        ts = timestamp or datetime.now(timezone.utc)
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO messages (chat_id, message_id, sender_id, sender_name, text, timestamp, is_outgoing) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chat_id, message_id, sender_id, sender_name, text, ts.timestamp(), int(is_outgoing)),
            )
            self._conn.commit()
        except Exception:
            logger.exception("Failed to add message to history")

    def get_messages(self, chat_id: int, limit: int = 20) -> list[ChatMessage]:
        """Получить последние N сообщений из чата."""
        rows = self._conn.execute(
            "SELECT message_id, sender_id, sender_name, text, timestamp, is_outgoing "
            "FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()

        messages = []
        for row in reversed(rows):
            messages.append(ChatMessage(
                message_id=row[0],
                sender_id=row[1],
                sender_name=row[2],
                text=row[3],
                timestamp=datetime.fromtimestamp(row[4], tz=timezone.utc),
                is_outgoing=bool(row[5]),
            ))
        return messages

    def get_last_activity(self, chat_id: int) -> datetime | None:
        """Получить время последнего сообщения в чате."""
        row = self._conn.execute(
            "SELECT MAX(timestamp) FROM messages WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        if row and row[0]:
            return datetime.fromtimestamp(row[0], tz=timezone.utc)
        return None

    def get_all_chats(self) -> list[dict]:
        """Получить список всех чатов с статистикой."""
        rows = self._conn.execute(
            "SELECT chat_id, COUNT(*) as msg_count, MAX(timestamp) as last_ts "
            "FROM messages GROUP BY chat_id ORDER BY last_ts DESC"
        ).fetchall()

        chats = []
        for row in rows:
            chats.append({
                "chat_id": row[0],
                "message_count": row[1],
                "last_activity": datetime.fromtimestamp(row[2], tz=timezone.utc).isoformat(),
            })
        return chats

    def trim_old_messages(self) -> int:
        """Удалить старые сообщения, оставив только max_per_chat на чат."""
        chat_ids = [r[0] for r in self._conn.execute(
            "SELECT DISTINCT chat_id FROM messages"
        ).fetchall()]

        total_deleted = 0
        for chat_id in chat_ids:
            self._conn.execute("""
                DELETE FROM messages WHERE chat_id = ? AND id NOT IN (
                    SELECT id FROM messages WHERE chat_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                )
            """, (chat_id, chat_id, self._max_per_chat))
            total_deleted += self._conn.execute("SELECT changes()").fetchone()[0]

        self._conn.commit()
        if total_deleted > 0:
            logger.info("Trimmed %d old messages", total_deleted)
        return total_deleted
