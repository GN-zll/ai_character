from __future__ import annotations

import asyncio
import logging
import os
import random

from src.core.notification import Notification, NotificationManager
from src.memory.diary import Diary
from src.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class ProactiveScheduler:
    """Планировщик proactive действий (kuni-style).

    Каждые N минут с шансом X% пушит нотификацию для рефлексии
    или proactive outreach.
    """

    def __init__(
        self,
        notification_manager: NotificationManager,
        diary: Diary,
        llm: LLMProvider,
    ) -> None:
        self._nm = notification_manager
        self._diary = diary
        self._llm = llm

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

    async def _act_proactively(self) -> None:
        """Создать proactive нотификацию."""
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
            metadata={"source": "proactive"},
        ))
        logger.info("Proactive notification pushed: %s", entry.text[:80])
