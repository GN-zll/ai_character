from __future__ import annotations

import logging
from pathlib import Path

from telethon import TelegramClient as TelethonClient, events
from telethon.tl.types import User, Chat, Channel

from src.client.base import BaseTelegramClient, IncomingMessage, SentMessage

logger = logging.getLogger(__name__)


class UserbotClient(BaseTelegramClient):
    """Клиент для личного аккаунта Telegram через Telethon."""

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_dir: Path = Path("data/session"),
    ) -> None:
        super().__init__()
        session_dir.mkdir(parents=True, exist_ok=True)
        session_path = session_dir / "userbot"

        self._client = TelethonClient(
            str(session_path),
            api_id=api_id,
            api_hash=api_hash,
        )

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self, phone: str | None = None) -> None:
        await self._client.start(phone=phone)
        me = await self._client.get_me()
        logger.info("Userbot logged in as %s (id=%s)", me.first_name, me.id)

    async def disconnect(self) -> None:
        await self._client.disconnect()
        logger.info("Userbot disconnected")

    async def run_until_disconnected(self) -> None:
        await self._client.run_until_disconnected()

    # ── Отправка сообщений ─────────────────────────────────────

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        reply_to: int | None = None,
    ) -> SentMessage:
        msg = await self._client.send_message(chat_id, text, reply_to=reply_to)
        logger.debug("Sent message to %s: %s", chat_id, text[:80])
        return SentMessage(chat_id=msg.chat_id, message_id=msg.id)

    async def send_photo(
        self,
        chat_id: int | str,
        file: str | Path,
        *,
        caption: str | None = None,
    ) -> SentMessage:
        msg = await self._client.send_file(chat_id, file, caption=caption)
        return SentMessage(chat_id=msg.chat_id, message_id=msg.id)

    async def send_voice(
        self,
        chat_id: int | str,
        file: str | Path,
    ) -> SentMessage:
        msg = await self._client.send_file(chat_id, file, voice_note=True)
        return SentMessage(chat_id=msg.chat_id, message_id=msg.id)

    # ── Получение данных ───────────────────────────────────────

    async def get_me(self) -> dict:
        user: User = await self._client.get_me()
        return {"id": user.id, "first_name": user.first_name, "username": user.username}

    async def send_chat_action(self, chat_id: int | str, action: str = "typing") -> None:
        from telethon.tl.functions.messages import SetTypingRequest
        from telethon.tl.types import SendMessageTypingAction
        await self._client(SetTypingRequest(peer=chat_id, action=SendMessageTypingAction()))

    async def edit_message(self, chat_id: int | str, message_id: int, new_text: str) -> None:
        await self._client.edit_message(chat_id, message_id, new_text)

    async def mark_read(self, chat_id: int | str) -> None:
        await self._client.send_read_acknowledge(chat_id)

    # ── Обработка входящих ─────────────────────────────────────

    async def start_listening(self) -> None:
        @self._client.on(events.NewMessage)
        async def _on_new_message(event: events.NewMessage.Event) -> None:
            msg = event.message
            sender = await msg.get_sender()
            sender_name = getattr(sender, "first_name", "Unknown")

            incoming = IncomingMessage(
                chat_id=msg.chat_id,
                message_id=msg.id,
                sender_id=msg.sender_id or 0,
                sender_name=sender_name,
                text=msg.text,
                date=msg.date,
                is_outgoing=msg.out,
                reply_to=msg.reply_to_msg_id,
                raw=msg,
            )
            await self._dispatch_message(incoming)

        logger.info("Userbot listening for new messages...")

    # ── Userbot-специфичные методы ─────────────────────────────

    async def get_dialogs(self, limit: int = 20):
        return await self._client.get_dialogs(limit=limit)

    async def get_messages(self, chat_id: int | str, limit: int = 20, *, offset_id: int = 0):
        return await self._client.get_messages(chat_id, limit=limit, offset_id=offset_id)

    async def get_unread_dialogs(self):
        dialogs = await self._client.get_dialogs(limit=None)
        return [d for d in dialogs if d.unread_count > 0]

    async def mark_read(self, chat_id: int | str) -> None:
        await self._client.send_read_acknowledge(chat_id)

    async def get_entity(self, chat_id: int | str) -> User | Chat | Channel:
        return await self._client.get_entity(chat_id)
