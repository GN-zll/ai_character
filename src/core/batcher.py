from __future__ import annotations

import asyncio
import logging
import random

from src.client.base import IncomingMessage
from src.core.notification import Notification, NotificationManager

logger = logging.getLogger(__name__)


class MessageBatcher:
    """Собирает сообщения из одного чата, обрабатывает пачкой.

    Логика debounce:
    - Сообщение приходит → кладётся в буфер, запускается таймер (random 0-5 сек)
    - Если до истечения таймера приходит ещё сообщение → буфер + сброс таймера
    - Таймер сработал → все сообщения из буфера → одна нотификация → LLM
    """

    def __init__(
        self,
        notification_manager: NotificationManager,
        *,
        min_delay: float = 0.0,
        max_delay: float = 5.0,
    ) -> None:
        self._nm = notification_manager
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._buffers: dict[int, list[IncomingMessage]] = {}
        self._tasks: dict[int, asyncio.Task] = {}

    async def add(self, msg: IncomingMessage) -> None:
        """Добавить сообщение в буфер (с debounce)."""
        chat_id = msg.chat_id

        if chat_id not in self._buffers:
            self._buffers[chat_id] = []
        self._buffers[chat_id].append(msg)

        # Сбрасываем предыдущий таймер
        if chat_id in self._tasks:
            self._tasks[chat_id].cancel()
            logger.debug("Batcher: timer reset for chat %s", chat_id)

        # Запускаем новый таймер
        self._tasks[chat_id] = asyncio.create_task(self._flush_after(chat_id))

    async def _flush_after(self, chat_id: int) -> None:
        """Ждём случайную задержку, потом отправляем пачку."""
        delay = random.uniform(self._min_delay, self._max_delay)
        logger.debug("Batcher: waiting %.1fs for chat %s", delay, chat_id)
        await asyncio.sleep(delay)

        messages = self._buffers.pop(chat_id, [])
        self._tasks.pop(chat_id, None)

        if not messages:
            return

        logger.info("Batcher: flushing %d messages from chat %s", len(messages), chat_id)

        # Формируем одну нотификацию из всех сообщений
        combined = self._format_messages(messages)
        await self._nm.push(Notification(
            priority=10,
            message=combined,
            pin=f"<chat_id={chat_id} />",
            metadata={"chat_id": chat_id, "batch_size": len(messages)},
        ))

    @staticmethod
    def _format_messages(messages: list[IncomingMessage]) -> str:
        """Форматировать пачку сообщений в один текст."""
        if len(messages) == 1:
            msg = messages[0]
            text = f"New message from {msg.sender_name} (chat_id={msg.chat_id}): {msg.text}"
            if msg.reply_to_text:
                text = f"New message from {msg.sender_name} (chat_id={msg.chat_id}, replying to '{msg.reply_to_text}'): {msg.text}"
            return text

        lines = [f"New messages from {messages[0].sender_name} (chat_id={messages[0].chat_id}):"]
        for msg in messages:
            if msg.reply_to_text:
                lines.append(f"- (replying to '{msg.reply_to_text}') {msg.text}")
            else:
                lines.append(f"- {msg.text}")
        return "\n".join(lines)
