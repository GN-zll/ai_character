from __future__ import annotations

import asyncio
import logging

from src.config import Config
from src.client import create_client
from src.client.base import IncomingMessage
from src.llm.provider import LLMProvider
from src.memory.diary import Diary
from src.memory.rag import VectorStore
from src.memory.working_memory import WorkingMemory
from src.memory.contacts import Contacts
from src.memory.chat_history import ChatHistory
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


async def main() -> None:
    # ── Config ─────────────────────────────────────────────────
    config = Config.load()

    if config.telegram.whitelist:
        logger.info("Whitelist active: %s", config.telegram.whitelist)
    else:
        logger.info("No whitelist — all messages will be processed")

    # ── Components ─────────────────────────────────────────────
    client = create_client(config.telegram.client_type)
    llm = LLMProvider(config.llm)
    diary = Diary(config.memory.diary_dir)
    vector_store = VectorStore(config.memory.vectors_dir)
    working_memory = WorkingMemory(config.memory.working_memory_file)
    contacts = Contacts(config.memory.contacts_file)
    chat_history = ChatHistory(config.memory.history_db, config.memory.history_max_per_chat)
    personality = Personality(config.character)
    notification_manager = NotificationManager()

    # Добавляем owner'а в контакты
    if config.character.owner_chat_id:
        contacts.update(
            config.character.owner_chat_id,
            name=config.character.owner_name,
            tags=["owner"],
        )

    worker = Worker(
        name="main",
        config=config,
        client=client,
        llm=llm,
        diary=diary,
        vector_store=vector_store,
        working_memory=working_memory,
        contacts=contacts,
        chat_history=chat_history,
        personality=personality,
        notification_manager=notification_manager,
    )

    proactive = ProactiveScheduler(
        config=config,
        notification_manager=notification_manager,
        diary=diary,
        llm=llm,
    )

    batcher = MessageBatcher(
        notification_manager,
        min_delay=config.behavior.batch_delay_min,
        max_delay=config.behavior.batch_delay_max,
    )

    # ── Message handler → history + batcher ───────────────────
    @client.on_new_message()
    async def on_message(msg: IncomingMessage) -> None:
        # Whitelist
        if config.telegram.whitelist and msg.chat_id not in config.telegram.whitelist:
            logger.debug("Ignored message from %s (chat_id=%s)", msg.sender_name, msg.chat_id)
            return

        logger.info("Message from %s (chat_id=%s): %s", msg.sender_name, msg.chat_id, msg.text)

        # Сохраняем в историю
        if msg.text:
            chat_history.add_message(
                chat_id=msg.chat_id,
                message_id=msg.message_id,
                sender_id=msg.sender_id,
                sender_name=msg.sender_name,
                text=msg.text,
                is_outgoing=msg.is_outgoing,
            )

        # Отправляем в batcher
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
