from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class IncomingMessage:
    """Унифицированное входящее сообщение."""
    chat_id: int
    message_id: int
    sender_id: int
    sender_name: str
    text: str | None
    date: object  # datetime
    is_outgoing: bool = False
    reply_to: int | None = None
    raw: object = None  # оригинальный объект сообщения


@dataclass
class SentMessage:
    """Результат отправки сообщения."""
    chat_id: int
    message_id: int


MessageHandler = Callable[[IncomingMessage], Awaitable[None]]


class BaseTelegramClient(ABC):
    """Абстрактный интерфейс Telegram клиента."""

    def __init__(self) -> None:
        self._message_handlers: list[MessageHandler] = []

    # ── Lifecycle ──────────────────────────────────────────────

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def run_until_disconnected(self) -> None: ...

    # ── Отправка сообщений ─────────────────────────────────────

    @abstractmethod
    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        reply_to: int | None = None,
    ) -> SentMessage: ...

    @abstractmethod
    async def send_photo(
        self,
        chat_id: int | str,
        file: str | Path,
        *,
        caption: str | None = None,
    ) -> SentMessage: ...

    @abstractmethod
    async def send_voice(
        self,
        chat_id: int | str,
        file: str | Path,
    ) -> SentMessage: ...

    # ── Получение данных ───────────────────────────────────────

    @abstractmethod
    async def get_me(self) -> dict: ...

    # ── Действия ───────────────────────────────────────────────

    @abstractmethod
    async def send_chat_action(self, chat_id: int | str, action: str = "typing") -> None:
        """Отправить индикатор действия (typing, upload_photo, ...).

        Args:
            chat_id: ID чата
            action: тип действия — "typing", "upload_photo", "record_voice" и т.д.
        """
        ...

    # ── Обработка входящих ─────────────────────────────────────

    def on_new_message(self) -> Callable:
        """Декоратор для регистрации обработчика новых сообщений."""

        def decorator(func: MessageHandler) -> MessageHandler:
            self._message_handlers.append(func)
            return func

        return decorator

    async def _dispatch_message(self, msg: IncomingMessage) -> None:
        """Вызвать все зарегистрированные обработчики."""
        for handler in self._message_handlers:
            try:
                await handler(msg)
            except Exception:
                logger.exception("Error in message handler")

    @abstractmethod
    async def start_listening(self) -> None: ...
