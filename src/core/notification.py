from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass(order=True)
class Notification:
    """Нотификация — событие для обработки worker'ом."""
    priority: int = field(compare=True)
    message: str = field(compare=False)
    pin: str | None = field(default=None, compare=False)
    metadata: dict = field(default_factory=dict, compare=False)


class NotificationManager:
    """Очередь нотификаций с приоритетом (kuni-style).

    Входящие сообщения, proactive триггеры и прочие события
    попадают сюда как нотификации и обрабатываются worker'ами.
    """

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[Notification] = asyncio.PriorityQueue()
        self._workers: list[asyncio.Task] = []

    async def push(self, notification: Notification) -> None:
        """Добавить нотификацию в очередь."""
        await self._queue.put(notification)
        logger.debug("Notification pushed: %s (priority=%d)", notification.message[:50], notification.priority)

    async def next(self) -> Notification:
        """Получить следующую нотификацию (блокирует если очередь пуста)."""
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()
