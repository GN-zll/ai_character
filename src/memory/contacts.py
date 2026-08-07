from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.config import StatLevelsConfig

logger = logging.getLogger(__name__)


@dataclass
class Contact:
    """Контакт в адресной книге с статами отношений."""
    chat_id: int
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)  # {"trust": 0, "closeness": 0, "tension": 0}
    summary: str = ""  # 50-100 char summary, updated during sleep


class Contacts:
    """Адресная книга с статами отношений.

    Хранится в JSON файле. AI может читать и обновлять контакты.
    """

    def __init__(
        self,
        file_path: str | Path = "data/contacts.json",
        stat_names: list[str] | None = None,
        stat_levels: StatLevelsConfig | None = None,
    ) -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._contacts: dict[int, Contact] = {}
        self._stat_names = stat_names or []
        self._stat_levels = stat_levels or StatLevelsConfig()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for item in data:
                    # Миграция: если stats нет — создаём дефолтные
                    if "stats" not in item:
                        item["stats"] = {name: 0 for name in self._stat_names}
                    if "summary" not in item:
                        item["summary"] = ""
                    c = Contact(**item)
                    self._contacts[c.chat_id] = c
                logger.info("Loaded %d contacts", len(self._contacts))
            except Exception:
                logger.exception("Failed to load contacts")

    def _save(self) -> None:
        data = [asdict(c) for c in self._contacts.values()]
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, chat_id: int) -> Contact | None:
        """Получить контакт по chat_id."""
        return self._contacts.get(chat_id)

    def get_or_default(self, chat_id: int) -> Contact:
        """Получить контакт или создать заглушку."""
        if chat_id in self._contacts:
            return self._contacts[chat_id]
        return Contact(
            chat_id=chat_id,
            name=f"User#{chat_id}",
            stats={name: 0 for name in self._stat_names},
        )

    def update(
        self,
        chat_id: int,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> Contact:
        """Создать или обновить контакт."""
        contact = self._contacts.get(chat_id) or Contact(
            chat_id=chat_id,
            name="",
            stats={n: 0 for n in self._stat_names},
        )
        if name is not None:
            contact.name = name
        if description is not None:
            contact.description = description
        if tags is not None:
            contact.tags = tags
        self._contacts[chat_id] = contact
        self._save()
        logger.info("Contact updated: %s (%s)", chat_id, contact.name)
        return contact

    def update_stats(self, chat_id: int, stat_changes: dict[str, int]) -> Contact:
        """Обновить статы отношений для контакта."""
        contact = self.get_or_default(chat_id)
        for stat_name, delta in stat_changes.items():
            old_value = contact.stats.get(stat_name, 0)
            new_value = max(-100, min(100, old_value + delta))
            contact.stats[stat_name] = new_value
            logger.info("Stat %s for %s: %d → %d (%+d)", stat_name, contact.name, old_value, new_value, delta)
        self._contacts[chat_id] = contact
        self._save()
        return contact

    def update_summary(self, chat_id: int, summary: str) -> None:
        """Обновить summary отношений (только во сне)."""
        contact = self.get_or_default(chat_id)
        contact.summary = summary[:100]  # max 100 chars
        self._contacts[chat_id] = contact
        self._save()

    def get_stat_level(self, stat_name: str, value: int) -> str:
        """Получить текстовый label для значения стата."""
        return self._stat_levels.get_label(stat_name, value)

    def get_changed_contacts(self) -> list[Contact]:
        """Получить контакты с ненулевыми статами (для отображения)."""
        return [c for c in self._contacts.values() if any(v != 0 for v in c.stats.values())]

    def list_all(self) -> list[Contact]:
        """Получить все контакты."""
        return list(self._contacts.values())

    def format_relationship(self, chat_id: int) -> str:
        """Форматировать статы отношений для одного контакта."""
        contact = self.get(chat_id)
        if not contact:
            return ""

        if not any(v != 0 for v in contact.stats.values()):
            return ""

        parts = []
        for stat_name, value in contact.stats.items():
            label = self.get_stat_level(stat_name, value)
            parts.append(f"{stat_name}: {value:+d} {label}")

        result = f"<relationship with {contact.name} (chat_id={chat_id})>\n"
        result += " | ".join(parts)
        if contact.summary:
            result += f"\nsummary: {contact.summary}"
        result += "\n</relationship>"
        return result

    def format_for_prompt(self) -> str:
        """Форматировать все контакты для инжекта в prompt."""
        if not self._contacts:
            return ""
        lines = []
        for c in self._contacts.values():
            line = f"- {c.chat_id}: {c.name}"
            if c.description:
                line += f" — {c.description}"
            if c.tags:
                line += f" [{', '.join(c.tags)}]"
            lines.append(line)
        return "\n".join(lines)
