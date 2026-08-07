from __future__ import annotations

import json
import logging
from datetime import timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _format_for_embedding(msg) -> str:
    """Форматировать сообщение для эмбеддинга с временем и отправителем."""
    ts = msg.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    direction = "→" if msg.is_outgoing else "←"
    return f"[{ts}] [{direction} {msg.sender_name}] {msg.text}"


class ChatIndexer:
    """Индексирует историю чата в векторную БД (коллекция chat_history).

    Не векторизует каждое сообщение — накапливает счётчик и батчит
    раз в N сообщений (chat_index_threshold). Индексирует и входящие,
    и исходящие сообщения. Только новые сообщения (backfill не делает).
    """

    def __init__(
        self,
        *,
        chat_history,
        vector_store,
        llm,
        threshold: int = 100,
        state_file: str = "data/chat_index_state.json",
    ) -> None:
        self._chat_history = chat_history
        self._store = vector_store
        self._llm = llm
        self._threshold = threshold
        self._state_path = Path(state_file)
        self._unindexed = 0
        self._last_indexed_id = self._load_last_id()

    def _load_last_id(self) -> int:
        try:
            if self._state_path.exists():
                data = json.loads(self._state_path.read_text(encoding="utf-8"))
                return int(data.get("last_indexed_id", 0))
        except Exception:
            logger.exception("Failed to load chat index state")
        return 0

    def _save_last_id(self, last_id: int) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps({"last_indexed_id": last_id}),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to save chat index state")

    @property
    def last_indexed_id(self) -> int:
        return self._last_indexed_id

    async def on_message(self) -> None:
        """Вызывается после каждого сохранённого сообщения."""
        self._unindexed += 1
        if self._unindexed >= self._threshold:
            await self.flush()

    async def flush(self) -> None:
        """Проиндексировать сообщения с last_indexed_id, обновить state."""
        if self._unindexed == 0:
            return

        messages = self._chat_history.get_messages_since(
            self._last_indexed_id,
            limit=self._unindexed,
        )
        if not messages:
            self._unindexed = 0
            return

        texts = [_format_for_embedding(m) for m in messages]
        embeddings = await self._llm.embed_batch(texts)
        if embeddings is None:
            logger.warning("Chat indexing skipped: embeddings unavailable")
            self._unindexed = 0
            return

        count = 0
        for text, emb, msg in zip(texts, embeddings, messages):
            self._store.add_chat(
                text,
                emb,
                metadata={
                    "chat_id": msg.chat_id,
                    "sender": msg.sender_name,
                    "is_outgoing": int(msg.is_outgoing),
                    "timestamp": msg.timestamp.isoformat(),
                },
            )
            count += 1

        self._last_indexed_id = messages[-1].row_id
        self._save_last_id(self._last_indexed_id)
        self._unindexed = 0
        logger.info("Chat indexed: %d messages (last_id=%d)", count, self._last_indexed_id)
