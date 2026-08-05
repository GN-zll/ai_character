from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Awaitable

from telethon import TelegramClient as TelethonClient, events
from telethon.tl.types import (
    Dialog,
    Message,
    User,
    Chat,
    Channel,
    InputPeerUser,
    InputPeerChat,
    InputPeerChannel,
)
from telethon.tl.functions.messages import ReadHistoryRequest
from telethon.tl.functions.users import GetFullUserRequest

logger = logging.getLogger(__name__)

# Тип колбэка для обработки новых сообщений
MessageHandler = Callable[[Message], Awaitable[None]]


class TelegramClient:
    """Обёртка над Telethon для работы с личным аккаунтом Telegram."""

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_dir: Path = Path("data/session"),
    ) -> None:
        session_dir.mkdir(parents=True, exist_ok=True)
        session_path = session_dir / "userbot"

        self._client = TelethonClient(
            str(session_path),
            api_id=api_id,
            api_hash=api_hash,
            # system_version="4.16.30-vxCUSTOM",
            # lang_code="en",
        )
        self._message_handlers: list[MessageHandler] = []

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self, phone: str | None = None) -> None:
        """Подключение и авторизация. При первом запуске запросит номер и код."""
        await self._client.start(phone=phone)
        me = await self._client.get_me()
        logger.info("Logged in as %s (id=%s)", me.first_name, me.id)

    async def disconnect(self) -> None:
        """Отключение от Telegram."""
        await self._client.disconnect()
        logger.info("Disconnected from Telegram")

    async def run_until_disconnected(self) -> None:
        """Блокирующий запуск — ждёт пока клиент не отключится."""
        await self._client.run_until_disconnected()

    # ── Отправка сообщений ─────────────────────────────────────

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        reply_to: int | None = None,
    ) -> Message:
        """Отправить текстовое сообщение."""
        msg = await self._client.send_message(
            chat_id,
            text,
            reply_to=reply_to,
        )
        logger.debug("Sent message to %s: %s", chat_id, text[:80])
        return msg

    async def send_photo(
        self,
        chat_id: int | str,
        file: str | Path,
        *,
        caption: str | None = None,
    ) -> Message:
        """Отправить фотографию."""
        return await self._client.send_file(chat_id, file, caption=caption)

    async def send_voice(
        self,
        chat_id: int | str,
        file: str | Path,
    ) -> Message:
        """Отправить голосовое сообщение."""
        return await self._client.send_file(chat_id, file, voice_note=True)

    async def send_file(
        self,
        chat_id: int | str,
        file: str | Path,
        *,
        caption: str | None = None,
    ) -> Message:
        """Отправить произвольный файл."""
        return await self._client.send_file(chat_id, file, caption=caption)

    # ── Чтение / получение ─────────────────────────────────────

    async def get_dialogs(self, limit: int = 20) -> list[Dialog]:
        """Получить список диалогов (чатов)."""
        return await self._client.get_dialogs(limit=limit)

    async def get_messages(
        self,
        chat_id: int | str,
        limit: int = 20,
        *,
        offset_id: int = 0,
    ) -> list[Message]:
        """Получить историю сообщений из чата."""
        return await self._client.get_messages(
            chat_id,
            limit=limit,
            offset_id=offset_id,
        )

    async def get_unread_dialogs(self) -> list[Dialog]:
        """Получить диалоги с непрочитанными сообщениями."""
        dialogs = await self._client.get_dialogs(limit=None)
        return [d for d in dialogs if d.unread_count > 0]

    async def mark_read(self, chat_id: int | str) -> None:
        """Отметить сообщения в чате как прочитанные."""
        await self._client.send_read_acknowledge(chat_id)

    async def get_entity(self, chat_id: int | str) -> User | Chat | Channel:
        """Получить информацию о пользователе/чате/канале."""
        return await self._client.get_entity(chat_id)

    # ── Обработка входящих ─────────────────────────────────────

    def on_new_message(self, *, from_users: list[int] | None = None) -> Callable:
        """Декоратор для регистрации обработчика новых сообщений.

        Usage:
            @client.on_new_message()
            async def handler(message: Message):
                print(message.text)
        """

        def decorator(func: MessageHandler) -> MessageHandler:
            self._message_handlers.append(func)
            return func

        return decorator

    async def _setup_event_handler(self) -> None:
        """Внутренний метод — подключает обработчик событий Telethon."""

        @self._client.on(events.NewMessage)
        async def _on_new_message(event: events.NewMessage.Event) -> None:
            message: Message = event.message
            for handler in self._message_handlers:
                try:
                    await handler(message)
                except Exception:
                    logger.exception("Error in message handler")

    async def start_listening(self) -> None:
        """Запуск прослушивания входящих сообщений (после start())."""
        await self._setup_event_handler()
        logger.info("Listening for new messages...")

    # ── Утилиты ────────────────────────────────────────────────

    @property
    def me(self) -> User | None:
        """Возвращает объект текущего пользователя (после start())."""
        return self._client._self_input_peer  # noqa: SLF001
        # Лучше использовать get_me() асинхронно

    async def get_me(self) -> User:
        """Получить информацию о себе."""
        return await self._client.get_me()

    def is_connected(self) -> bool:
        """Проверить, подключён ли клиент."""
        return self._client.is_connected()
