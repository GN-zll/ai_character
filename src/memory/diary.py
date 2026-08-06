from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DiaryEntry:
    """Запись в дневнике."""
    id: str
    text: str
    created_at: datetime
    source: str = "diary"  # "diary", "consolidated", "chat_dump"
    entry_type: str = "general"  # "general", "person", "reflection"
    chat_id: int | None = None  # ID чата для записей типа "person"
    stat_changes: dict[str, int] = field(default_factory=dict)  # {"trust": 2, "closeness": 1}
    metadata: dict = field(default_factory=dict)


class Diary:
    """Дневник — долгосрочная память в markdown файлах.

    Поддерживает типы записей:
    - general: общая запись
    - person: запись о конкретном человеке (с изменениями статов)
    - reflection: рефлексия над настроением
    """

    def __init__(self, diary_dir: str | Path = "data/diary") -> None:
        self._dir = Path(diary_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[DiaryEntry] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Загрузить существующие записи из файлов."""
        for f in sorted(self._dir.glob("*.md")):
            try:
                text = f.read_text(encoding="utf-8").strip()
                if not text:
                    continue

                # Парсим метаданные из YAML-like front matter
                entry_type = "general"
                chat_id = None
                stat_changes = {}

                if text.startswith("<!--"):
                    end = text.find("-->")
                    if end != -1:
                        header = text[4:end].strip()
                        text = text[end + 3:].strip()

                        for part in header.split():
                            if part.startswith("type:"):
                                entry_type = part[5:]
                            elif part.startswith("chat_id:"):
                                try:
                                    chat_id = int(part[8:])
                                except ValueError:
                                    pass
                            elif part.startswith("stat_changes:"):
                                try:
                                    stat_changes = json.loads(part[13:].replace("'", '"'))
                                except Exception:
                                    pass

                entry = DiaryEntry(
                    id=f.stem,
                    text=text,
                    created_at=datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc),
                    source="diary",
                    entry_type=entry_type,
                    chat_id=chat_id,
                    stat_changes=stat_changes,
                )
                self._entries.append(entry)
            except Exception:
                logger.exception("Failed to load diary entry: %s", f)
        logger.info("Loaded %d diary entries", len(self._entries))

    def add(
        self,
        text: str,
        *,
        source: str = "diary",
        entry_type: str = "general",
        chat_id: int | None = None,
        stat_changes: dict[str, int] | None = None,
    ) -> DiaryEntry:
        """Добавить запись в дневник."""
        now = datetime.now(timezone.utc)
        entry_id = now.strftime("%Y%m%d_%H%M%S_%f")

        if stat_changes is None:
            stat_changes = {}

        entry = DiaryEntry(
            id=entry_id,
            text=text,
            created_at=now,
            source=source,
            entry_type=entry_type,
            chat_id=chat_id,
            stat_changes=stat_changes,
        )
        self._entries.append(entry)

        # Сохраняем в файл с метаданными
        file_path = self._dir / f"{entry_id}.md"
        content = self._format_entry(entry)
        file_path.write_text(content, encoding="utf-8")
        logger.debug("Diary entry added: %s (type=%s)", entry_id, entry_type)
        return entry

    def _format_entry(self, entry: DiaryEntry) -> str:
        """Форматировать запись для сохранения в файл."""
        parts = []

        # YAML-like front matter
        meta_parts = [f"type:{entry.entry_type}"]
        if entry.chat_id is not None:
            meta_parts.append(f"chat_id:{entry.chat_id}")
        if entry.stat_changes:
            meta_parts.append(f"stat_changes:{json.dumps(entry.stat_changes)}")
        parts.append(f"<!-- {' '.join(meta_parts)} -->")
        parts.append("")
        parts.append(entry.text)

        return "\n".join(parts)

    def list_entries(
        self,
        limit: int | None = None,
        entry_type: str | None = None,
        chat_id: int | None = None,
    ) -> list[DiaryEntry]:
        """Получить записи (новые первыми). Можно фильтровать по типу и chat_id."""
        entries = sorted(self._entries, key=lambda e: e.created_at, reverse=True)

        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]
        if chat_id is not None:
            entries = [e for e in entries if e.chat_id == chat_id]
        if limit:
            entries = entries[:limit]
        return entries

    def list_today_changes(self) -> list[DiaryEntry]:
        """Получить записи за сегодня с изменениями статов."""
        today = datetime.now(timezone.utc).date()
        return [
            e for e in self._entries
            if e.created_at.date() == today and e.stat_changes
        ]

    def random_entry(self) -> DiaryEntry | None:
        """Получить случайную запись."""
        import random
        if not self._entries:
            return None
        return random.choice(self._entries)

    def count(self) -> int:
        return len(self._entries)
