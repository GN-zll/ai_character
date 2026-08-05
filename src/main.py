from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from src.client import create_client
from src.client.base import IncomingMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    load_dotenv()

    client = create_client()

    @client.on_new_message()
    async def echo_handler(msg: IncomingMessage) -> None:
        logger.info("Message from %s: %s", msg.sender_name, msg.text)
        if msg.text and not msg.is_outgoing:
            await client.send_message(msg.chat_id, f"Echo: {msg.text}", reply_to=msg.message_id)

    logger.info("Starting Telegram client...")
    await client.start()
    await client.start_listening()

    me = await client.get_me()
    logger.info("Started as %s (id=%s)", me["first_name"], me["id"])
    logger.info("Press Ctrl+C to stop")

    try:
        await client.run_until_disconnected()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
