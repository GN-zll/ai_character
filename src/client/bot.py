from __future__ import annotations

import logging
from pathlib import Path

from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, MessageHandler as TgMessageHandler, ContextTypes, filters

from src.client.base import BaseTelegramClient, IncomingMessage, SentMessage

logger = logging.getLogger(__name__)


class BotClient(BaseTelegramClient):
    """Клиент Telegram Bot API через python-telegram-bot."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token
        self._app = ApplicationBuilder().token(token).build()
        self._bot: Bot | None = None

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        self._bot = self._app.bot
        me = await self._bot.get_me()
        logger.info("Bot started as %s (@%s)", me.first_name, me.username)

    async def disconnect(self) -> None:
        if self._app.running:
            await self._app.stop()
        logger.info("Bot disconnected")

    async def run_until_disconnected(self) -> None:
        import asyncio as _asyncio

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Bot polling started")

        stop_event = _asyncio.Event()

        def _signal_handler() -> None:
            stop_event.set()

        import signal
        loop = _asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                pass

        await stop_event.wait()
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    # ── Отправка сообщений ─────────────────────────────────────

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        reply_to: int | None = None,
    ) -> SentMessage:
        assert self._bot is not None
        msg = await self._bot.send_message(
            chat_id=int(chat_id),
            text=text,
            reply_to_message_id=reply_to,
        )
        logger.debug("Sent message to %s: %s", chat_id, text[:80])
        return SentMessage(chat_id=msg.chat_id, message_id=msg.message_id)

    async def send_photo(
        self,
        chat_id: int | str,
        file: str | Path,
        *,
        caption: str | None = None,
    ) -> SentMessage:
        assert self._bot is not None
        with open(file, "rb") as f:
            msg = await self._bot.send_photo(chat_id=int(chat_id), photo=f, caption=caption)
        return SentMessage(chat_id=msg.chat_id, message_id=msg.message_id)

    async def send_voice(
        self,
        chat_id: int | str,
        file: str | Path,
    ) -> SentMessage:
        assert self._bot is not None
        with open(file, "rb") as f:
            msg = await self._bot.send_voice(chat_id=int(chat_id), voice=f)
        return SentMessage(chat_id=msg.chat_id, message_id=msg.message_id)

    # ── Получение данных ───────────────────────────────────────

    async def get_me(self) -> dict:
        assert self._bot is not None
        user = await self._bot.get_me()
        return {"id": user.id, "first_name": user.first_name, "username": user.username}

    async def send_chat_action(self, chat_id: int | str, action: str = "typing") -> None:
        assert self._bot is not None
        await self._bot.send_chat_action(chat_id=int(chat_id), action=action)

    async def edit_message(self, chat_id: int | str, message_id: int, new_text: str) -> None:
        assert self._bot is not None
        await self._bot.edit_message_text(chat_id=int(chat_id), message_id=message_id, text=new_text)

    # ── Обработка входящих ─────────────────────────────────────

    async def start_listening(self) -> None:
        async def _handle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            msg = update.message
            if msg is None:
                return

            sender = msg.from_user
            incoming = IncomingMessage(
                chat_id=msg.chat_id,
                message_id=msg.message_id,
                sender_id=sender.id if sender else 0,
                sender_name=sender.first_name if sender else "Unknown",
                text=msg.text,
                date=msg.date,
                is_outgoing=False,
                reply_to=msg.reply_to_message.message_id if msg.reply_to_message else None,
                raw=msg,
            )
            await self._dispatch_message(incoming)

        self._app.add_handler(TgMessageHandler(filters.TEXT & ~filters.COMMAND, _handle))
        logger.info("Bot listening for new messages...")
