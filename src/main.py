from __future__ import annotations

import asyncio
import logging

from src.character.personality import Personality
from src.client import create_client
from src.client.base import IncomingMessage
from src.config import Config
from src.core.batcher import MessageBatcher
from src.core.notification import NotificationManager
from src.core.scheduler import Scheduler
from src.core.sleep import SleepManager
from src.core.worker import Worker
from src.llm.provider import LLMProvider
from src.memory.chat_history import ChatHistory
from src.memory.contacts import Contacts
from src.memory.diary import Diary
from src.memory.rag import VectorStore
from src.memory.todo import TodoList
from src.memory.working_memory import WorkingMemory

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
    llm = LLMProvider(config.llm, log_config=config.llm_log)
    diary = Diary(config.memory.diary_dir)
    vector_store = VectorStore(config.memory.vectors_dir)
    working_memory = WorkingMemory(config.memory.working_memory_file)
    todo_list = TodoList(config.memory.todo_file)

    # Contacts с статами отношений
    stat_names = [s.name for s in config.relationship_stats]
    contacts = Contacts(
        config.memory.contacts_file,
        stat_names=stat_names,
        stat_levels=config.stat_levels,
    )

    chat_history = ChatHistory(config.memory.history_db, config.memory.history_max_per_chat)
    personality = Personality(config.character)
    notification_manager = NotificationManager()

    # Scheduler — единый планерщик для alarms, reminders, wait, proactive
    scheduler = Scheduler.create(
        notification_manager,
        config=config,
        data_file=config.memory.reminders_file,
    )

    # Добавляем owner'а в контакты
    if config.character.owner_chat_id:
        contacts.update(
            config.character.owner_chat_id,
            name=config.character.owner_name,
            tags=["owner"],
        )

    sleep_manager = SleepManager(
        config=config,
        notification_manager=notification_manager,
        scheduler=scheduler,
        diary=diary,
        llm=llm,
        working_memory=working_memory,
        contacts=contacts,
    )
    scheduler.set_sleep_manager(sleep_manager)
    scheduler.configure_dependencies(diary=diary, contacts=contacts, llm=llm)

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
        todo_list=todo_list,
        personality=personality,
        notification_manager=notification_manager,
        sleep_manager=sleep_manager,
        scheduler=scheduler,
    )

    # Callback: когда alarm срабатывает → сбрасываем mode worker'а в IDLE
    from src.core.tools import WorkerMode
    sleep_manager._on_wake_callback = lambda: setattr(worker, '_mode', WorkerMode.IDLE)

    batcher = MessageBatcher(
        notification_manager=notification_manager,
        miss_chance=config.behavior.miss_notification_chance,
        thinking_delay_min=config.behavior.thinking_delay_min,
        thinking_delay_max=config.behavior.thinking_delay_max,
        batch_window_min=config.behavior.batch_window_min,
        batch_window_max=config.behavior.batch_window_max,
        preview_length=config.behavior.notification_preview_length,
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

        # Reply context: ищем оригинал сообщения
        if msg.reply_to and not msg.reply_to_text:
            original = chat_history.get_message_by_id(msg.chat_id, msg.reply_to)
            if original:
                msg.reply_to_text = original.text

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
    scheduler.start()
    sleep_manager.start()

    logger.info("Press Ctrl+C to stop")

    try:
        await client.run_until_disconnected()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await worker.stop()
        await sleep_manager.stop()
        await scheduler.stop()
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
