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
from src.memory.contacts import Contacts
from src.character.personality import Personality
from src.core.notification import Notification, NotificationManager
from src.core.worker import Worker
from src.core.proactive import ProactiveScheduler
from src.core.batcher import MessageBatcher

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
    contacts = Contacts()
    personality = Personality()
    notification_manager = NotificationManager()

    # Добавляем owner'а в контакты
    owner_id = os.getenv("OWNER_CHAT_ID")
    if owner_id:
        contacts.update(int(owner_id), name=personality.name + "'s owner", tags=["owner"])

    worker = Worker(
        name="main",
        client=client,
        llm=llm,
        diary=diary,
        vector_store=vector_store,
        working_memory=working_memory,
        contacts=contacts,
        personality=personality,
        notification_manager=notification_manager,
    )

    proactive = ProactiveScheduler(
        notification_manager=notification_manager,
        diary=diary,
        llm=llm,
    )

    batcher = MessageBatcher(notification_manager, min_delay=0.0, max_delay=5.0)

    # ── Message handler → batcher ─────────────────────────────
    @client.on_new_message()
    async def on_message(msg: IncomingMessage) -> None:
        if whitelist and msg.chat_id not in whitelist:
            logger.debug("Ignored message from %s (chat_id=%s)", msg.sender_name, msg.chat_id)
            return

        logger.info("Message from %s (chat_id=%s): %s", msg.sender_name, msg.chat_id, msg.text)

        if msg.text and not msg.is_outgoing:
            await batcher.add(msg)

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
