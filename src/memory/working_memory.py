from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Рабочая память — короткочасовая память на 1-3 дня.

    Хранится в файле working_memory.md.
    Инжектится в system prompt как <things_to_remember>.
    Обновляется при diary dump.
    """

    def __init__(self, file_path: str | Path = "data/working_memory.md") -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._content: str = ""
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            self._content = self._path.read_text(encoding="utf-8").strip()
            logger.info("Working memory loaded (%d chars)", len(self._content))

    def get(self) -> str:
        """Получить текущее содержимое рабочей памяти."""
        return self._content

    def update(self, new_content: str) -> None:
        """Обновить рабочую память."""
        self._content = new_content
        self._path.write_text(new_content, encoding="utf-8")
        logger.info("Working memory updated (%d chars)", len(new_content))

    def is_empty(self) -> bool:
        return not self._content.strip()

    def format_for_prompt(self) -> str:
        """Форматировать для инжекта в system prompt."""
        if self.is_empty():
            return ""
        return (
            f"<things_to_remember>\n"
            f"{self._content}\n"
            f"</things_to_remember>"
        )
