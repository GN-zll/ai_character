from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OPEN = "- [ ] "
DONE = "- [x] "


class TodoList:
    """Список задач/напоминаний персонажу.

    Хранится в файле todo.md в формате markdown checklist.
    Инжектится в system prompt как <todo>.
    Напоминания не теряются после перезагрузки.
    """

    def __init__(self, file_path: str | Path = "data/todo.md") -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._open: list[str] = []
        self._done: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(OPEN):
                self._open.append(stripped[len(OPEN):])
            elif stripped.startswith(DONE):
                self._done.append(stripped[len(DONE):])
        logger.info("Todo list loaded (%d open, %d done)", len(self._open), len(self._done))

    def _save(self) -> None:
        lines = [f"{OPEN}{item}" for item in self._open]
        lines += [f"{DONE}{item}" for item in self._done]
        self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def add(self, content: str) -> None:
        """Добавить задачу."""
        content = content.strip()
        if not content:
            return
        self._open.append(content)
        self._save()
        logger.info("Todo added: %s", content)

    def done(self, index: int) -> str:
        """Отметить задачу выполненной (1-based индекс среди открытых)."""
        if index < 1 or index > len(self._open):
            return (
                f"Error: no todo item at index {index}. "
                f"Current {len(self._open)} open todo(s)."
            )
        item = self._open.pop(index - 1)
        self._done.append(item)
        self._save()
        logger.info("Todo done: %s", item)
        return f"Completed todo: {item}"

    def get_open(self) -> list[str]:
        return list(self._open)

    def is_empty(self) -> bool:
        return not self._open

    def format_for_prompt(self) -> str:
        """Форматировать для инжекта в system prompt (только открытые)."""
        if self.is_empty():
            return ""
        items = "\n".join(f"- {i}. {item}" for i, item in enumerate(self._open, start=1))
        return f"<todo>\n{items}\n</todo>"
