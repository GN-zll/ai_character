from __future__ import annotations

import asyncio
import logging
import random

from src.client.base import IncomingMessage
from src.core.notification import Notification, NotificationManager

logger = logging.getLogger(__name__)


class MessageBatcher:
    """Батчер сообщений с debounce, thinking delay и miss chance.

    Логика (kuni-style):
    1. Сообщение приходит → miss_chance (5% пропустить)
    2. Thinking delay (1-3 сек) — "думает"
    3. Batch window (2-5 сек) — собирает сообщения
    4. Notification с ВСЕМИ сообщениями → Worker
    """

    def __init__(
        self,
        notification_manager: NotificationManager,
        *,
        miss_chance: float = 0.05,
        thinking_delay_min: float = 1.0,
        thinking_delay_max: float = 3.0,
        batch_window_min: float = 2.0,
        batch_window_max: float = 5.0,
        preview_length: int = 10,
    ) -> None:
        self._nm = notification_manager
        self._miss_chance = miss_chance
        self._thinking_delay_min = thinking_delay_min
        self._thinking_delay_max = thinking_delay_max
        self._batch_window_min = batch_window_min
        self._batch_window_max = batch_window_max
        self._preview_length = preview_length

        self._pending: set[int] = set()
        self._buffers: dict[int, list[IncomingMessage]] = {}

    async def add(self, msg: IncomingMessage) -> None:
        """Обработать входящее сообщение."""
        chat_id = msg.chat_id

        # Шанс пропустить уведомление (не заметила)
        if random.random() < self._miss_chance:
            logger.info(
                "Missed notification from %s (chat_id=%s) — %.0f%% chance",
                msg.sender_name,
                chat_id,
                self._miss_chance * 100,
            )
            return

        # Уже есть pending notification для этого чата — просто буферизируем
        if chat_id in self._pending:
            self._buffers.setdefault(chat_id, []).append(msg)
            logger.debug("Buffered message from %s (chat_id=%s)", msg.sender_name, chat_id)
            return

        # Первое сообщение — запускаем обработку
        self._pending.add(chat_id)
        self._buffers.setdefault(chat_id, []).append(msg)

        # Запускаем обработку в фоне (не блокируя другие чаты)
        asyncio.create_task(self._process_chat(chat_id))

    async def _process_chat(self, chat_id: int) -> None:
        """Обработать накопленные сообщения из чата."""
        try:
            # Thinking delay (1-3 сек) — "думает"
            thinking_delay = random.uniform(
                self._thinking_delay_min, self._thinking_delay_max
            )
            logger.debug("Thinking delay: %.1fs for chat %s", thinking_delay, chat_id)
            await asyncio.sleep(thinking_delay)

            # Batch window (2-5 сек) — собираем ещё сообщения
            batch_window = random.uniform(
                self._batch_window_min, self._batch_window_max
            )
            logger.debug("Batch window: %.1fs for chat %s", batch_window, chat_id)
            await asyncio.sleep(batch_window)

            # Собираем всё из буфера
            messages = self._buffers.pop(chat_id, [])
            self._pending.discard(chat_id)

            if not messages:
                return

            logger.info(
                "Batching %d message(s) from chat %s", len(messages), chat_id
            )

            # Формируем notification
            combined = self._format_batch(messages)
            await self._nm.push(Notification(
                priority=10,
                message=combined,
                pin=f"<chat_id={chat_id} />",
                metadata={"chat_id": chat_id, "batch_size": len(messages)},
            ))

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error processing chat %s", chat_id)
            self._pending.discard(chat_id)
            self._buffers.pop(chat_id, None)

    def _format_batch(self, messages: list[IncomingMessage]) -> str:
        """Форматировать batch сообщений для notification."""
        count = len(messages)
        chat_id = messages[0].chat_id
        sender = messages[0].sender_name

        # Preview — последние 10 символов последнего сообщения
        last_text = messages[-1].text or ""
        preview = last_text[:self._preview_length]

        if count == 1:
            return (
                f"You have 1 unread message in chat {chat_id} ({sender}):\n"
                f'"{preview}..."\n'
                f"Use open_chat({chat_id}) to read and reply."
            )
        else:
            return (
                f"You have {count} unread messages in chat {chat_id} ({sender}):\n"
                f'Latest: "{preview}..."\n'
                f"Use open_chat({chat_id}) to read and reply."
            )
