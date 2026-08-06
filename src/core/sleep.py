from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from src.config import Config
from src.core.notification import Notification, NotificationManager
from src.memory.diary import Diary
from src.memory.working_memory import WorkingMemory
from src.llm.provider import LLMProvider, ChatMessage

logger = logging.getLogger(__name__)

MSK_OFFSET = 3  # UTC+3


def _msk_now() -> datetime:
    """Текущее время по МСК."""
    return datetime.now(timezone.utc) + timedelta(hours=MSK_OFFSET)


class SleepManager:
    """Управление сном персонажа (kuni-style).

    Логика:
    1. LLM вызывает sleep(wake_hour) — начинает процесс засыпания
    2. Tool handler предлагает перед сном: написать кому-то, записать в дневник
    3. LLM делает дела, потом вызывает confirm_sleep()
    4. LLM вызывает set_alarm(hour, minute) — ставит будильник
    5. Consolidation: сжатие дневника, обновление working memory
    6. Персонаж "засыпает"
    7. Когда будильник срабатывает → нотификация "Доброе утро!"
    """

    def __init__(
        self,
        config: Config,
        notification_manager: NotificationManager,
        diary: Diary,
        llm: LLMProvider,
        working_memory: WorkingMemory,
        on_wake_callback: object = None,  # Callable[[], None]
    ) -> None:
        self._config = config
        self._nm = notification_manager
        self._diary = diary
        self._llm = llm
        self._working_memory = working_memory
        self._on_wake_callback = on_wake_callback

        self._last_wake_time = datetime.now(timezone.utc)
        self._alarm: datetime | None = None
        self._is_sleeping = False

        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._alarm_loop())
        logger.info("SleepManager started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def is_sleeping(self) -> bool:
        return self._is_sleeping

    def awake_hours(self) -> float:
        """Сколько часов персонаж не спит."""
        return (datetime.now(timezone.utc) - self._last_wake_time).total_seconds() / 3600

    async def start_sleep_process(self, wake_hour: int, wake_minute: int = 0) -> str:
        """Начать процесс засыпания (вызывается когда LLM вызывает sleep tool)."""
        awake_h = self.awake_hours()
        msk = _msk_now()

        # Рассчитываем когда проснёмся
        wake_time_msk = msk.replace(hour=wake_hour, minute=wake_minute, second=0, microsecond=0)
        if wake_time_msk <= msk:
            wake_time_msk += timedelta(days=1)

        sleep_hours = (wake_time_msk - msk).total_seconds() / 3600

        prompt = (
            f"Current time (MSK): {msk.strftime('%H:%M')}\n"
            f"You've been awake for {awake_h:.0f} hours.\n"
            f"Target wake-up time: {wake_hour:02d}:{wake_minute:02d} MSK ({sleep_hours:.0f} hours of sleep)\n\n"
        )

        if awake_h < 16:
            prompt += (
                "You haven't been awake for very long. Maybe you don't need to sleep yet?\n"
                "If you still want to sleep, that's okay. But consider staying awake a bit longer.\n\n"
            )
        elif awake_h > 24:
            prompt += (
                "You've been awake for a very long time! You should definitely get some rest.\n\n"
            )
        else:
            prompt += (
                "You've been awake for a while. It's a good time to sleep.\n\n"
            )

        prompt += (
            "Before going to sleep, you might want to:\n"
            "- Write to someone (use send_message)\n"
            "- Write in your diary (use diary_write)\n"
            "- Or just say goodnight\n\n"
            "When you're ready to sleep, call confirm_sleep().\n"
            "After that, call set_alarm(wake_hour, wake_minute) to set your alarm."
        )

        self._alarm = None  # Reset alarm
        return prompt

    async def confirm_sleep(self) -> str:
        """Подтвердить сон — consolidation + засыпание."""
        if self._is_sleeping:
            return "You are already sleeping!"

        self._is_sleeping = True

        # Diary consolidation
        consolidated = await self._consolidate_diary()

        # Обновляем working memory
        await self._update_working_memory()

        msk = _msk_now()
        alarm_str = ""
        if self._alarm:
            alarm_str = f" Alarm set for {self._alarm.strftime('%H:%M')} MSK."

        return (
            f"Goodnight! 💤 Time: {msk.strftime('%H:%M')} MSK.{alarm_str}\n"
            f"Diary consolidated: {consolidated} entries processed.\n"
            f"You are now sleeping. Sweet dreams!"
        )

    async def set_alarm(self, wake_hour: int, wake_minute: int = 0) -> str:
        """Поставить будильник."""
        msk = _msk_now()
        wake_time_msk = msk.replace(hour=wake_hour, minute=wake_minute, second=0, microsecond=0)
        if wake_time_msk <= msk:
            wake_time_msk += timedelta(days=1)

        # Конвертируем в UTC
        self._alarm = wake_time_msk - timedelta(hours=MSK_OFFSET)

        sleep_hours = (wake_time_msk - msk).total_seconds() / 3600
        logger.info("Alarm set for %s MSK (%.1f hours)", wake_time_msk.strftime('%H:%M'), sleep_hours)
        return f"Alarm set for {wake_hour:02d}:{wake_minute:02d} MSK. Sweet dreams! 🌙"

    async def _alarm_loop(self) -> None:
        """Проверяет будильник каждую минуту."""
        while self._running:
            try:
                await asyncio.sleep(60)

                if self._is_sleeping and self._alarm:
                    now = datetime.now(timezone.utc)
                    if now >= self._alarm:
                        await self._wake_up()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("SleepManager alarm loop error")

    async def _wake_up(self) -> None:
        """Разбудить персонажа."""
        self._is_sleeping = False
        self._last_wake_time = datetime.now(timezone.utc)
        alarm = self._alarm
        self._alarm = None

        # Вызываем callback для сброса mode worker'а
        if self._on_wake_callback:
            try:
                self._on_wake_callback()
            except Exception:
                logger.exception("Wake callback failed")

        msk = _msk_now()
        sleep_duration = ""
        if alarm:
            hours = (datetime.now(timezone.utc) - alarm).total_seconds() / 3600
            sleep_duration = f" You slept for about {hours:.0f} hours."

        await self._nm.push(Notification(
            priority=20,  # высокий приоритет — проснулась
            message=(
                f"Good morning! ☀️ It's {msk.strftime('%H:%M')} MSK.{sleep_duration}\n"
                f"You just woke up. Time to check your messages and start a new day!"
            ),
            metadata={"source": "alarm_wake_up"},
        ))
        logger.info("Woke up at %s MSK", msk.strftime('%H:%M'))

    async def _consolidate_diary(self) -> int:
        """Консолидация дневника — сжатие похожих записей."""
        entries = self._diary.list_entries(limit=50)
        if len(entries) < 2:
            return 0

        # Берём записи для консолидации (последние 20)
        to_consolidate = entries[:20]
        entries_text = "\n---\n".join(
            f"[{e.id}] {e.text}" for e in to_consolidate
        )

        prompt = ChatMessage(
            role="user",
            content=(
                "You are consolidating diary entries during sleep. "
                "Review these entries and create a consolidated summary.\n\n"
                "Rules:\n"
                "- Merge similar entries into one\n"
                "- Keep important facts, emotions, and details\n"
                "- Remove duplicates and low-value entries\n"
                "- Write in third person\n"
                "- Be concise\n\n"
                f"Entries to consolidate:\n{entries_text}\n\n"
                "Write a consolidated summary (just the content):"
            ),
        )

        try:
            response = await self._llm.chat(
                messages=[
                    ChatMessage(role="system", content="You are a memory consolidator during sleep."),
                    prompt,
                ]
            )
            if response.content:
                # Сохраняем консолидированную запись
                self._diary.add(response.content, source="consolidated")
                logger.info("Diary consolidated: %d entries → 1 summary", len(to_consolidate))
                return len(to_consolidate)
        except Exception:
            logger.exception("Diary consolidation failed")
        return 0

    async def _update_working_memory(self) -> None:
        """Обновить working memory перед сном."""
        current = self._working_memory.get()

        prompt = ChatMessage(
            role="user",
            content=(
                "You are updating working memory before sleep.\n\n"
                f"Current working memory:\n{current if current else '(empty)'}\n\n"
                "Rules:\n"
                "- Keep: unfinished tasks, important reminders\n"
                "- Remove: completed tasks, things no longer relevant\n"
                "- Add: any final thoughts before sleep\n"
                "- Be concise, max 500 chars\n\n"
                "Write updated working memory (just the content):"
            ),
        )

        try:
            response = await self._llm.chat(
                messages=[
                    ChatMessage(role="system", content="You are a memory manager."),
                    prompt,
                ]
            )
            if response.content:
                self._working_memory.update(response.content.strip())
                logger.info("Working memory updated before sleep")
        except Exception:
            logger.exception("Failed to update working memory before sleep")
