from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timezone

from src.core.notification import Notification, NotificationManager
from src.memory.diary import Diary
from src.memory.contacts import Contacts
from src.memory.chat_history import ChatHistory
from src.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class ProactiveScheduler:
    """Планировщик proactive действий (kuni-style).

    Каждые N минут с шансом X% пушит нотификацию для рефлексии
    или proactive outreach. Выбирает контакт с которым давно не общались.
    """

    def __init__(
        self,
        notification_manager: NotificationManager,
        diary: Diary,
        llm: LLMProvider,
        contacts: Contacts | None = None,
        chat_history: ChatHistory | None = None,
    ) -> None:
        self._nm = notification_manager
        self._diary = diary
        self._llm = llm
        self._contacts = contacts
        self._chat_history = chat_history

        self._interval_min = int(os.getenv("PROACTIVE_INTERVAL_MIN", "27"))
        self._chance = float(os.getenv("PROACTIVE_CHANCE", "0.5"))

        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Proactive scheduler started (interval=%d min, chance=%.0f%%)",
                     self._interval_min, self._chance * 100)

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
                await asyncio.sleep(self._interval_min * 60)

                if random.random() > self._chance:
                    logger.debug("Proactive: skipped (chance)")
                    continue

                await self._act_proactively()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Proactive scheduler error")
                await asyncio.sleep(60)

    def _pick_contact_for_outreach(self) -> tuple[int, str] | None:
        """Выбрать контакт для proactive outreach.

        Приоритет: контакты с которыми давно не общались.
        """
        if not self._contacts or not self._chat_history:
            return None

        contacts = self._contacts.list_all()
        if not contacts:
            return None

        # Исключаем owner'а (ему не пишем proactively обычно)
        candidates = [c for c in contacts if "owner" not in c.tags]
        if not candidates:
            candidates = contacts  # fallback

        # Сортируем по последней активности (давно не общались = выше приоритет)
        scored = []
        for c in candidates:
            last = self._chat_history.get_last_activity(c.chat_id)
            if last:
                age_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
            else:
                age_hours = 999  # никогда не общались
            scored.append((age_hours, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Берём из топ-3 с наибольшей давностью
        top = scored[:3]
        _, contact = random.choice(top)
        return contact.chat_id, contact.name

    async def _act_proactively(self) -> None:
        """Создать proactive нотификацию."""
        # Пытаемся выбрать контакт для outreach
        target = self._pick_contact_for_outreach()

        if target:
            chat_id, name = target
            # Получаем контекст чата
            context = ""
            if self._chat_history:
                messages = self._chat_history.get_messages(chat_id, limit=10)
                if messages:
                    lines = []
                    for m in messages:
                        role = "me" if m.is_outgoing else m.sender_name
                        lines.append(f"[{role}]: {m.text}")
                    context = "\n".join(lines)

            prompt = (
                f"You haven't talked to {name} (chat_id={chat_id}) in a while.\n"
            )
            if context:
                prompt += f"\n<recent_messages>\n{context}\n</recent_messages>\n\n"
            prompt += (
                "Maybe you should write to them? Think about what to say.\n"
                "Use send_message(chat_id, text) to reach out.\n"
                "Or use wait() if you don't want to write right now."
            )

            await self._nm.push(Notification(
                priority=0,
                message=prompt,
                pin="<act_proactively />",
                metadata={"source": "proactive_outreach", "target_chat_id": chat_id},
            ))
            logger.info("Proactive outreach to %s (chat_id=%s)", name, chat_id)
            return

        # Fallback: рефлексия на основе дневника
        entry = self._diary.random_entry()
        if not entry:
            logger.debug("Proactive: no diary entries, skipping")
            return

        prompt = (
            f"<diary_entry>\n{entry.text}\n</diary_entry>\n\n"
            "It's time to reflect on your thoughts!\n"
            "- Maybe make some reasoning?\n"
            "- Maybe do some reflection?\n"
            "- Maybe write to a person and initiate a dialogue?\n"
            "Act proactively!"
        )

        await self._nm.push(Notification(
            priority=0,
            message=prompt,
            pin="<act_proactively />",
            metadata={"source": "proactive_reflection"},
        ))
        logger.info("Proactive reflection: %s", entry.text[:80])
