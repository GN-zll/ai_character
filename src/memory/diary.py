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
    metadata: dict = field(default_factory=dict)


class Diary:
    """Дневник — долгосрочная память в markdown файлах + векторный поиск."""

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
                entry = DiaryEntry(
                    id=f.stem,
                    text=text,
                    created_at=datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc),
                    source="diary",
                )
                self._entries.append(entry)
            except Exception:
                logger.exception("Failed to load diary entry: %s", f)
        logger.info("Loaded %d diary entries", len(self._entries))

    def add(self, text: str, *, source: str = "diary") -> DiaryEntry:
        """Добавить запись в дневник."""
        now = datetime.now(timezone.utc)
        entry_id = now.strftime("%Y%m%d_%H%M%S_%f")
        entry = DiaryEntry(
            id=entry_id,
            text=text,
            created_at=now,
            source=source,
        )
        self._entries.append(entry)

        # Сохраняем в файл
        file_path = self._dir / f"{entry_id}.md"
        file_path.write_text(text, encoding="utf-8")
        logger.debug("Diary entry added: %s", entry_id)
        return entry

    def list_entries(self, limit: int | None = None) -> list[DiaryEntry]:
        """Получить записи (новые первыми)."""
        entries = sorted(self._entries, key=lambda e: e.created_at, reverse=True)
        if limit:
            entries = entries[:limit]
        return entries

    def random_entry(self) -> DiaryEntry | None:
        """Получить случайную запись."""
        import random
        if not self._entries:
            return None
        return random.choice(self._entries)

    def count(self) -> int:
        return len(self._entries)
