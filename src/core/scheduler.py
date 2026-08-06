from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.notification import Notification, NotificationManager

if TYPE_CHECKING:
    from src.config import Config
    from src.core.sleep import SleepManager

logger = logging.getLogger(__name__)

MSK_OFFSET = 3  # UTC+3


def _utc_now() -> datetime:
    """Текущее время UTC."""
    return datetime.now(timezone.utc)


def _msk_now() -> datetime:
    """Текущее время по МСК."""
    return _utc_now() + timedelta(hours=MSK_OFFSET)


class ScheduleKind:
    """Типы расписание-событий."""
    ALARM = "alarm"
    REMINDER = "reminder"
    WAIT = "wait"
    PROACTIVE = "proactive"


@dataclass
class ScheduleItem:
    """Одно запланированное событие."""
    id: str
    kind: str
    fire_utc: datetime
    created_utc: datetime
    note: str = ""
    chat_id: int | None = None
    # Мета-данные специфичные для kind (например, reschedule параметры proactive)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "fire_utc": self.fire_utc.timestamp(),
            "created_utc": self.created_utc.timestamp(),
            "note": self.note,
            "chat_id": self.chat_id,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduleItem":
        return cls(
            id=data["id"],
            kind=data["kind"],
            fire_utc=datetime.fromtimestamp(data["fire_utc"], tz=timezone.utc),
            created_utc=datetime.fromtimestamp(data["created_utc"], tz=timezone.utc),
            note=data.get("note", ""),
            chat_id=data.get("chat_id"),
            meta=data.get("meta", {}),
        )


class Scheduler:
    """Единый синглетон-планировщик.

    Управляет всеми событиями во времени: alarms, reminders, waits,
    proactive. Один цикл проверки, одна persistence.

    kinds:
      - alarm:     будильник → sleep_manager.wake() + notification   [persist]
      - reminder:  напоминание с контекстом → notification           [persist]
      - wait:      запланированный check-in → notification           [persist]
      - proactive: случайное событие → outreach/reflection + respawn [memory]
    """

    _instance: "Scheduler | None" = None

    def __init__(
        self,
        notification_manager: "NotificationManager",
        *,
        config: "Config | None" = None,
        data_file: str | Path = "data/reminders.json",
        check_interval_s: int = 30,
    ) -> None:
        self._nm = notification_manager
        self._config = config
        self._data_file = Path(data_file)
        self._check_interval = check_interval_s

        self._items: list[ScheduleItem] = []
        self._sleep_manager: "SleepManager | None" = None

        # Dependencies для proactive message generation
        self._diary = None
        self._contacts = None
        self._llm = None

        self._task: asyncio.Task | None = None
        self._running = False

    # ── Singleton ──────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "Scheduler":
        if cls._instance is None:
            raise RuntimeError("Scheduler not initialized. Call Scheduler.create() first.")
        return cls._instance

    @classmethod
    def create(
        cls,
        notification_manager: "NotificationManager",
        **kwargs: Any,
    ) -> "Scheduler":
        """Создать (или вернуть существующий) синглтон Scheduler."""
        if cls._instance is not None:
            logger.warning("Scheduler singleton already created, returning existing instance")
            return cls._instance
        cls._instance = cls(notification_manager, **kwargs)
        return cls._instance

    @classmethod
    def _reset_instance(cls) -> None:
        """Сбросить синглтон (для тестов)."""
        cls._instance = None

    # ── DI / конфигурация ──────────────────────────────────────

    def set_sleep_manager(self, sm: "SleepManager | None") -> None:
        self._sleep_manager = sm

    def configure_dependencies(
        self,
        *,
        diary=None,
        contacts=None,
        llm=None,
    ) -> None:
        """Подключить зависимости для генерации proactive сообщений."""
        self._diary = diary
        self._contacts = contacts
        self._llm = llm

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._load()
        self._ensure_proactive_scheduled()
        self._task = asyncio.create_task(self._loop())
        logger.info("Scheduler started (%d items)", len(self._items))

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    # ── Public API: добавление событий ─────────────────────────

    def add_reminder(
        self,
        fire_utc: datetime,
        note: str,
        chat_id: int | None = None,
    ) -> ScheduleItem:
        return self._add_item(ScheduleItem(
            id=uuid.uuid4().hex,
            kind=ScheduleKind.REMINDER,
            fire_utc=fire_utc,
            created_utc=_utc_now(),
            note=note,
            chat_id=chat_id,
        ))

    def add_alarm(self, fire_utc: datetime) -> ScheduleItem:
        return self._add_item(ScheduleItem(
            id=uuid.uuid4().hex,
            kind=ScheduleKind.ALARM,
            fire_utc=fire_utc,
            created_utc=_utc_now(),
        ))

    def add_wait(self, minutes: int, chat_id: int | None = None) -> ScheduleItem:
        return self._add_item(ScheduleItem(
            id=uuid.uuid4().hex,
            kind=ScheduleKind.WAIT,
            fire_utc=_utc_now() + timedelta(minutes=minutes),
            created_utc=_utc_now(),
            chat_id=chat_id,
        ))

    def cancel(self, item_id: str) -> bool:
        for item in self._items:
            if item.id == item_id:
                self._items.remove(item)
                self._save()
                return True
        return False

    def list_active(self, kind: str | None = None) -> list[ScheduleItem]:
        if kind is None:
            return list(self._items)
        return [i for i in self._items if i.kind == kind]

    # ── Внутреннее ─────────────────────────────────────────────

    def _add_item(self, item: ScheduleItem) -> ScheduleItem:
        self._items.append(item)
        self._save()
        logger.info("Scheduled %s [%s] at %s UTC (%s)",
                     item.kind, item.id, item.fire_utc.strftime("%Y-%m-%d %H:%M"),
                     item.note or "-")
        return item

    def _ensure_proactive_scheduled(self) -> None:
        """Убедиться что запланирован хотя бы один proactive item."""
        if any(i.kind == ScheduleKind.PROACTIVE for i in self._items):
            return
        self._schedule_next_proactive()

    def _schedule_next_proactive(self) -> None:
        """Запланировать следующий proactive item с рандомным fire_utc."""
        cfg = self._config.behavior if self._config else None
        base_interval = cfg.proactive_interval_min if cfg else 27
        # Jitter: ±30%
        delay_min = base_interval * 0.7
        delay_max = base_interval * 1.3
        delay = random.uniform(delay_min, delay_max)

        self._items.append(ScheduleItem(
            id=uuid.uuid4().hex,
            kind=ScheduleKind.PROACTIVE,
            fire_utc=_utc_now() + timedelta(minutes=delay),
            created_utc=_utc_now(),
            meta={"interval": base_interval},
        ))
        logger.debug("Next proactive check scheduled in %.1f min", delay)

    async def _loop(self) -> None:
        """Основной цикл: проверка due items каждые check_interval."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                now = _utc_now()

                # Собираем due items и удаляем их из списка
                due = [i for i in self._items if i.fire_utc <= now]
                if not due:
                    continue
                for item in due:
                    self._items.remove(item)
                    try:
                        await self._fire(item)
                    except Exception:
                        logger.exception("Failed to fire scheduler item %s", item.id)

                self._save()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Scheduler loop error")

    async def _fire(self, item: ScheduleItem) -> None:
        """Обработать сработавшее событие."""
        if item.kind == ScheduleKind.ALARM:
            await self._fire_alarm(item)
        elif item.kind == ScheduleKind.REMINDER:
            await self._fire_reminder(item)
        elif item.kind == ScheduleKind.WAIT:
            await self._fire_wait(item)
        elif item.kind == ScheduleKind.PROACTIVE:
            await self._fire_proactive(item)
        else:
            logger.warning("Unknown scheduler kind: %s", item.kind)

    async def _fire_alarm(self, item: ScheduleItem) -> None:
        """Будильник: разбудить + приветствие."""
        if self._sleep_manager is not None:
            await self._sleep_manager.wake()
        msk = _msk_now()
        await self._nm.push(Notification(
            priority=20,
            message=f"Good morning! ☀️ It's {msk.strftime('%H:%M')} MSK.\n"
                    "You just woke up. Time to check your messages and start a new day!",
            metadata={"source": "alarm_wake_up"},
        ))
        logger.info("Alarm fired, woke up at %s MSK", msk.strftime('%H:%M'))

    async def _fire_reminder(self, item: ScheduleItem) -> None:
        """Напоминание: notification с контекстом."""
        await self._nm.push(Notification(
            priority=15,
            message=f"⏰ Reminder: {item.note}",
            metadata={"source": "reminder", "chat_id": item.chat_id, "item_id": item.id},
        ))
        logger.info("Reminder fired: %s", item.note)

    async def _fire_wait(self, item: ScheduleItem) -> None:
        """Запланированный check-in."""
        await self._nm.push(Notification(
            priority=3,
            message=(
                "⏰ Scheduled check-in. "
                "Think about what to do next — check messages, reflect, or continue."
            ),
            metadata={"source": "wait", "chat_id": item.chat_id},
        ))
        logger.info("Wait check-in fired")

    async def _fire_proactive(self, item: ScheduleItem) -> None:
        """Proactive: roll dice → fire or skip → всегда respawn."""
        cfg = self._config.behavior if self._config else None
        chance = cfg.proactive_chance if cfg else 0.5

        if random.random() < chance:
            msg = await self._generate_proactive_message()
            if msg:
                await self._nm.push(Notification(
                    priority=5,
                    message=msg,
                    metadata={"source": "proactive"},
                ))
            logger.info("Proactive fired")
        else:
            logger.info("Proactive skipped (chance)")

        self._schedule_next_proactive()

    async def _generate_proactive_message(self) -> str | None:
        """Сгенерировать proactive сообщение (outreach или reflection).

        Копия логики старого ProactiveScheduler.
        """
        # Outreach к контактам с ненулевыми статами
        if self._contacts:
            candidates = self._contacts.get_changed_contacts()
            if candidates:
                contact = random.choice(candidates)
                return (
                    f"You haven't talked to {contact.name} in a while.\n"
                    f"Use get_chats() to see who you can write to, "
                    f"and get_chat_context() to read recent messages.\n"
                    f"Think about writing to {contact.name}!"
                )

        # Иначе — reflection по записи из дневника
        if self._diary:
            entry = self._diary.random_entry()
            if entry:
                return (
                    f"<diary_entry>\n{entry.text}\n</diary_entry>\n\n"
                    "It's time to reflect on your thoughts!\n"
                    "- Maybe make some reasoning?\n"
                    "- Maybe do some reflection?\n"
                    "- Maybe write to a person and initiate a dialogue? "
                    "You can use get_chats to see who you can write to, "
                    "and get_chat_context to read recent messages.\n"
                    "Act proactively!"
                )

        return None

    # ── Persistence ────────────────────────────────────────────

    def _save(self) -> None:
        """Сохранить persistable items (не proactive)."""
        try:
            persistable = [i for i in self._items if i.kind != ScheduleKind.PROACTIVE]
            self._data_file.parent.mkdir(parents=True, exist_ok=True)
            self._data_file.write_text(
                json.dumps([i.to_dict() for i in persistable], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to save scheduler items")

    def _load(self) -> None:
        """Загрузить сохранённые items из диска."""
        if not self._data_file.exists():
            return

        try:
            data = json.loads(self._data_file.read_text(encoding="utf-8"))
            now = _utc_now()
            loaded = 0
            for entry in data:
                item = ScheduleItem.from_dict(entry)
                # Пропускаем overdue proactive всегда; для alarm/reminder fire сразу
                if item.fire_utc <= now:
                    if item.kind in (ScheduleKind.ALARM, ScheduleKind.REMINDER):
                        # Запускаем fire в фоне
                        asyncio.create_task(self._fire_immediately(item))
                    else:
                        # wait/proactive overdue — выбрасываем (контекст устарел)
                        continue
                else:
                    self._items.append(item)
                    loaded += 1
            logger.info("Loaded %d scheduler items from %s", loaded, self._data_file)
        except Exception:
            logger.exception("Failed to load scheduler items")

    async def _fire_immediately(self, item: ScheduleItem) -> None:
        """Fire overdue persistable item сразу (alarm/reminder после рестарта)."""
        try:
            await self._fire(item)
        except Exception:
            logger.exception("Failed to fire overdue item %s", item.id)
