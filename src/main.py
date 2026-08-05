from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from src.client import create_client
from src.client.base import IncomingMessage
from src.llm.provider import LLMProvider
from src.memory.diary import Diary
from src.memory.rag import VectorStore
from src.memory.working_memory import WorkingMemory
from src.character.personality import Personality
from src.core.notification import Notification, NotificationManager
from src.core.worker import Worker
from src.core.proactive import ProactiveScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_whitelist() -> set[int]:
    raw = os.getenv("WHITELIST_CHAT_IDS", "")
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


async def main() -> None:
    load_dotenv()

    # ── Whitelist ──────────────────────────────────────────────
    whitelist = load_whitelist()
    if whitelist:
        logger.info("Whitelist active: %s", whitelist)
    else:
        logger.info("No whitelist — all messages will be processed")

    # ── Components ─────────────────────────────────────────────
    client = create_client()
    llm = LLMProvider()
    diary = Diary()
    vector_store = VectorStore()
    working_memory = WorkingMemory()
    personality = Personality()
    notification_manager = NotificationManager()

    worker = Worker(
        name="main",
        client=client,
        llm=llm,
        diary=diary,
        vector_store=vector_store,
        working_memory=working_memory,
        personality=personality,
        notification_manager=notification_manager,
    )

    proactive = ProactiveScheduler(
        notification_manager=notification_manager,
        diary=diary,
        llm=llm,
    )

    # ── Message handler → notification ─────────────────────────
    @client.on_new_message()
    async def on_message(msg: IncomingMessage) -> None:
        if whitelist and msg.chat_id not in whitelist:
            logger.debug("Ignored message from %s (chat_id=%s)", msg.sender_name, msg.chat_id)
            return

        logger.info("Message from %s (chat_id=%s): %s", msg.sender_name, msg.chat_id, msg.text)

        if msg.text and not msg.is_outgoing:
            notification = Notification(
                priority=10,  # высокий приоритет для входящих сообщений
                message=f"New message from {msg.sender_name} (chat_id={msg.chat_id}): {msg.text}",
                pin=f"<chat_id={msg.chat_id} />",
                metadata={"chat_id": msg.chat_id, "sender_name": msg.sender_name},
            )
            await notification_manager.push(notification)

    # ── Start ──────────────────────────────────────────────────
    logger.info("Starting...")
    await client.start()
    await client.start_listening()

    me = await client.get_me()
    logger.info("Bot started as %s (id=%s)", me["first_name"], me["id"])

    worker.start()
    proactive.start()

    logger.info("Press Ctrl+C to stop")

    try:
        await client.run_until_disconnected()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await worker.stop()
        await proactive.stop()
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
