from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.config import Config
from src.core.scheduler import MSK_OFFSET, Scheduler, _msk_now
from src.llm.provider import ChatMessage, LLMProvider
from src.memory.diary import Diary
from src.memory.working_memory import WorkingMemory

logger = logging.getLogger(__name__)


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
        notification_manager,
        scheduler: Scheduler,
        diary: Diary,
        llm: LLMProvider,
        working_memory: WorkingMemory,
        contacts: object = None,  # Contacts
        on_wake_callback: object = None,  # Callable[[], None]
    ) -> None:
        self._config = config
        self._nm = notification_manager
        self._scheduler = scheduler
        self._diary = diary
        self._llm = llm
        self._working_memory = working_memory
        self._contacts = contacts
        self._on_wake_callback = on_wake_callback

        self._last_wake_time = datetime.now(timezone.utc)
        self._is_sleeping = False

    def start(self) -> None:
        # Нет собственного цикла — сон управляется снизу через scheduler.
        # start оставляем для консистентности lifecycle.
        logger.info("SleepManager started")

    async def stop(self) -> None:
        logger.info("SleepManager stopped")

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

        return prompt

    async def confirm_sleep(self) -> str:
        """Подтвердить сон — consolidation + засыпание."""
        if self._is_sleeping:
            return "You are already sleeping!"

        self._is_sleeping = True

        # Diary consolidation
        consolidated = await self._consolidate_diary()

        # Relationship summary consolidation
        summaries_updated = await self._consolidate_relationship_summaries()

        # Обновляем working memory
        await self._update_working_memory()

        msk = _msk_now()

        return (
            f"Goodnight! Time: {msk.strftime('%H:%M')} MSK.\n"
            f"Diary consolidated: {consolidated} entries processed.\n"
            f"Relationship summaries updated: {summaries_updated} contacts.\n"
            f"You are now sleeping. Sweet dreams!"
        )

    async def set_alarm(self, wake_hour: int, wake_minute: int = 0) -> str:
        """Поставить будильник (через единый Scheduler)."""
        msk = _msk_now()
        wake_time_msk = msk.replace(hour=wake_hour, minute=wake_minute, second=0, microsecond=0)
        if wake_time_msk <= msk:
            wake_time_msk += timedelta(days=1)

        # Конвертируем в UTC
        fire_utc = wake_time_msk - timedelta(hours=MSK_OFFSET)

        sleep_hours = (wake_time_msk - msk).total_seconds() / 3600
        self._scheduler.add_alarm(fire_utc)
        logger.info("Alarm set for %s MSK (%.1f hours)", wake_time_msk.strftime('%H:%M'), sleep_hours)
        return f"Alarm set for {wake_hour:02d}:{wake_minute:02d} MSK. Sweet dreams! 🌙"

    async def wake(self) -> None:
        """Разбудить персонажа. Вызывается Scheduler'ом при срабатывании alarm."""
        if not self._is_sleeping:
            return

        self._is_sleeping = False
        self._last_wake_time = datetime.now(timezone.utc)

        # Вызываем callback для сброса mode worker'а
        if self._on_wake_callback:
            try:
                self._on_wake_callback()
            except Exception:
                logger.exception("Wake callback failed")

        logger.info("Woke up at %s MSK", _msk_now().strftime('%H:%M'))

    async def _consolidate_relationship_summaries(self) -> int:
        """Обновить summary отношений для контактов с изменениями за день."""
        if not self._contacts:
            return 0

        # Находим контакты с ненулевыми статами
        changed = self._contacts.get_changed_contacts()
        if not changed:
            return 0

        # Загружаем diary entries за сегодня с изменениями статов
        today_entries = self._diary.list_today_changes()
        if not today_entries:
            return 0

        updated = 0
        for contact in changed:
            # Собираем записи для этого контакта
            contact_entries = [e for e in today_entries if e.chat_id == contact.chat_id]
            if not contact_entries:
                continue

            entries_text = "\n".join(f"- {e.text}" for e in contact_entries[:10])
            stat_parts = []
            for stat_name, value in contact.stats.items():
                if value != 0:
                    label = self._contacts.get_stat_level(stat_name, value)
                    stat_parts.append(f"{stat_name}: {value:+d} {label}")
            stats_str = " | ".join(stat_parts) if stat_parts else "no changes"

            prompt = ChatMessage(
                role="user",
                content=(
                    f"Update relationship summary for {contact.name} (chat_id={contact.chat_id}).\n\n"
                    f"Current stats: {stats_str}\n\n"
                    f"Today's interactions:\n{entries_text}\n\n"
                    f"Current summary: {contact.summary or '(none)'}\n\n"
                    f"Write a new summary (50-100 chars, English). Focus on the current state of the relationship, "
                    f"recent dynamics, and key feelings. Be concise and factual.\n\n"
                    f"New summary:"
                ),
            )

            try:
                response = await self._llm.chat(
                    messages=[
                        ChatMessage(role="system", content="You are a relationship analyst. Write concise summaries in English."),
                        prompt,
                    ],
                    reason="sleep:relationship_consolidation",
                )
                if response.content:
                    self._contacts.update_summary(contact.chat_id, response.content.strip())
                    updated += 1
                    logger.info("Summary updated for %s: %s", contact.name, response.content[:80])
            except Exception:
                logger.exception("Failed to update summary for %s", contact.name)

        return updated

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
                ],
                reason="sleep:diary_consolidation",
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
                ],
                reason="sleep:working_memory_update",
            )
            if response.content:
                self._working_memory.update(response.content.strip())
                logger.info("Working memory updated before sleep")
        except Exception:
            logger.exception("Failed to update working memory before sleep")
