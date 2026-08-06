from __future__ import annotations

import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Косинусное сходство между двумя векторами."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity между двумя текстами (fallback без эмбеддингов)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class AntiRepeat:
    """Проверка на повторяющиеся ответы.

    Хранит последние N ответов бота на чат.
    При отправке проверяет новый текст на similarity с предыдущими.
    Если similarity > порога → ошибка.
    """

    def __init__(
        self,
        llm=None,
        *,
        threshold: float = 0.75,
        max_history: int = 20,
    ) -> None:
        self._llm = llm
        self._threshold = threshold
        self._max_history = max_history
        self._history: dict[int, list[str]] = defaultdict(list)
        self._embeddings: dict[int, list[list[float]]] = defaultdict(list)

    async def check(self, chat_id: int, new_text: str) -> str | None:
        """Проверить текст на повтор. Возвращает ошибку если повтор, None если ок."""
        recent = self._history.get(chat_id, [])
        if not recent:
            return None

        # Cosine similarity если есть LLM
        if self._llm:
            try:
                new_emb = await self._llm.embed(new_text)
                if new_emb is not None:
                    return await self._check_embeddings(chat_id, new_text, new_emb)
            except Exception:
                logger.debug("Embeddings failed, falling back to Jaccard")

        # Fallback: Jaccard similarity
        return self._check_jaccard(chat_id, new_text)

    async def _check_embeddings(self, chat_id: int, new_text: str, new_emb: list[float]) -> str | None:
        """Проверка через cosine similarity эмбеддингов."""
        history_embs = self._embeddings.get(chat_id, [])
        for i, old_emb in enumerate(history_embs):
            sim = _cosine_similarity(new_emb, old_emb)
            if sim > self._threshold:
                logger.warning("Anti-repeat: similarity=%.3f with '%s'", sim, self._history[chat_id][i][:50])
                return (
                    f"You are repeating yourself (similarity={sim:.2f}). "
                    "Rephrase your message, say something different, or use wait() to end the conversation."
                )
        return None

    def _check_jaccard(self, chat_id: int, new_text: str) -> str | None:
        """Проверка через Jaccard similarity (fallback)."""
        recent = self._history.get(chat_id, [])
        for old_text in recent:
            sim = _jaccard_similarity(new_text, old_text)
            if sim > self._threshold:
                logger.warning("Anti-repeat (Jaccard): similarity=%.3f with '%s'", sim, old_text[:50])
                return (
                    f"You are repeating yourself (similarity={sim:.2f}). "
                    "Rephrase your message, say something different, or use wait() to end the conversation."
                )
        return None

    async def record(self, chat_id: int, text: str) -> None:
        """Запомнить отправленное сообщение."""
        self._history[chat_id].append(text)

        # Сохраняем эмбеддинг если есть LLM
        if self._llm:
            try:
                emb = await self._llm.embed(text)
                if emb is not None:
                    self._embeddings[chat_id].append(emb)
            except Exception:
                pass

        # Обрезаем историю
        if len(self._history[chat_id]) > self._max_history:
            self._history[chat_id] = self._history[chat_id][-self._max_history:]
            if self._llm:
                self._embeddings[chat_id] = self._embeddings[chat_id][-self._max_history:]
