from __future__ import annotations

import os
import sys
import logging

from src.client.base import BaseTelegramClient

logger = logging.getLogger(__name__)


def create_client(client_type: str = "bot") -> BaseTelegramClient:
    """Фабрика для создания Telegram клиента."""
    if client_type == "bot":
        from src.client.bot import BotClient

        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            logger.error("TELEGRAM_BOT_TOKEN не задан в .env файле")
            sys.exit(1)
        return BotClient(token=token)

    elif client_type == "userbot":
        from src.client.userbot import UserbotClient

        api_id = os.getenv("TELEGRAM_API_ID")
        api_hash = os.getenv("TELEGRAM_API_HASH")
        if not api_id or not api_hash:
            logger.error("TELEGRAM_API_ID и TELEGRAM_API_HASH должны быть заданы в .env")
            sys.exit(1)
        return UserbotClient(api_id=int(api_id), api_hash=api_hash)

    else:
        logger.error("Неизвестный тип клиента: %s. Допустимые: bot, userbot", client_type)
        sys.exit(1)
