from __future__ import annotations

import asyncio
import logging
import random

from src.config import Config
from src.core.notification import Notification, NotificationManager
from src.memory.diary import Diary
from src.memory.contacts import Contacts
from src.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class ProactiveScheduler:
    """Планировщик proactive действий (kuni-style).

    Каждые N минут с шансом X% пушит нотификацию для рефлексии.
    LLM сама решает что делать — рефлексировать, написать кому-то и т.д.
    """

    def __init__(
        self,
        config: Config,
        notification_manager: NotificationManager,
        diary: Diary,
        llm: LLMProvider,
        contacts: Contacts | None = None,
    ) -> None:
        self._config = config
        self._nm = notification_manager
        self._diary = diary
        self._llm = llm
        self._contacts = contacts

        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Proactive scheduler started (interval=%d min, chance=%.0f%%)",
                     self._config.behavior.proactive_interval_min,
                     self._config.behavior.proactive_chance * 100)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._config.behavior.proactive_interval_min * 60)

                if random.random() > self._config.behavior.proactive_chance:
                    logger.debug("Proactive: skipped (chance)")
                    continue

                await self._act_proactively()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Proactive scheduler error")
                await asyncio.sleep(60)

    async def _act_proactively(self) -> None:
        """Создать proactive нотификацию (kuni-style).

        1. Проверяем есть ли непрочитанные сообщения
        2. Если нет — берём запись из дневника для рефлексии
        """
        # Проверяем контакты с ненулевыми статами для outreach
        if self._contacts:
            candidates = self._contacts.get_changed_contacts()
            if candidates:
                # Выбираем контакт с наибольшей давностью общения
                import random as _random
                contact = _random.choice(candidates)
                await self._nm.push(Notification(
                    priority=5,
                    message=(
                        f"You haven't talked to {contact.name} in a while.\n"
                        f"Use get_chats() to see who you can write to, "
                        f"and get_chat_context() to read recent messages.\n"
                        f"Think about writing to {contact.name}!"
                    ),
                    pin="<act_proactively />",
                    metadata={"source": "proactive_outreach", "target_chat_id": contact.chat_id},
                ))
                logger.info("Proactive outreach to %s", contact.name)
                return

        # Обычная proactive логика — diary entry
        entry = self._diary.random_entry()
        if not entry:
            logger.debug("Proactive: no diary entries, skipping")
            return

        prompt = (
            f"<diary_entry>\n{entry.text}\n</diary_entry>\n\n"
            "It's time to reflect on your thoughts!\n"
            "- Maybe make some reasoning?\n"
            "- Maybe do some reflection?\n"
            "- Maybe write to a person and initiate a dialogue? "
            "You can use get_chats to see who you can write to, "
            "and get_chat_context to read recent messages.\n"
            "Act proactively!"
        )

        await self._nm.push(Notification(
            priority=0,
            message=prompt,
            pin="<act_proactively />",
            metadata={"source": "proactive_reflection"},
        ))
        logger.info("Proactive reflection: %s", entry.text[:80])
