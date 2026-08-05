from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Contact:
    """Контакт в адресной книге."""
    chat_id: int
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)


class Contacts:
    """Адресная книга — маппинг chat_id → имя/описание.

    Хранится в JSON файле. AI может читать и обновлять контакты.
    """

    def __init__(self, file_path: str | Path = "data/contacts.json") -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._contacts: dict[int, Contact] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for item in data:
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
        return self._contacts.get(chat_id) or Contact(chat_id=chat_id, name=f"User#{chat_id}")

    def update(self, chat_id: int, *, name: str | None = None, description: str | None = None, tags: list[str] | None = None) -> Contact:
        """Создать или обновить контакт."""
        contact = self._contacts.get(chat_id) or Contact(chat_id=chat_id, name="")
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

    def list_all(self) -> list[Contact]:
        """Получить все контакты."""
        return list(self._contacts.values())

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
