from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.client.telegram import TelegramClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Загрузить конфигурацию из .env файла."""
    load_dotenv()

    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")

    if not api_id or not api_hash:
        logger.error(
            "TELEGRAM_API_ID и TELEGRAM_API_HASH должны быть заданы в .env файле. "
            "Получить можно на https://my.telegram.org"
        )
        sys.exit(1)

    return {
        "api_id": int(api_id),
        "api_hash": api_hash,
    }


async def main() -> None:
    config = load_config()

    client = TelegramClient(
        api_id=config["api_id"],
        api_hash=config["api_hash"],
    )

    # Регистрируем обработчик сообщений
    @client.on_new_message()
    async def echo_handler(message) -> None:
        """Простой эхо-обработчик для тестирования."""
        sender = await message.get_sender()
        sender_name = getattr(sender, "first_name", "Unknown")
        logger.info("Message from %s: %s", sender_name, message.text)

        # Пока просто эхо — позже здесь будет логика AI
        if message.text and not message.out:
            await client.send_message(
                message.chat_id,
                f"Echo: {message.text}",
                reply_to=message.id,
            )

    # Подключаемся
    logger.info("Starting Telegram client...")
    await client.start()
    await client.start_listening()

    me = await client.get_me()
    logger.info("Bot started as %s (id=%s)", me.first_name, me.id)
    logger.info("Press Ctrl+C to stop")

    # Запускаем до отключения
    try:
        await client.run_until_disconnected()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
