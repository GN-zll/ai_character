from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EmotionalState:
    """Эмоциональное состояние персонажа."""
    mood: str = "neutral"  # happy, sad, excited, anxious, calm, angry, amused
    energy: float = 0.5    # 0.0 (exhausted) - 1.0 (energetic)
    sociability: float = 0.5  # 0.0 (withdrawn) - 1.0 (very social)

    def describe(self) -> str:
        return (
            f"Emotional state: mood={self.mood}, "
            f"energy={self.energy:.1f}, "
            f"sociability={self.sociability:.1f}"
        )


class Personality:
    """Управление персонажем: промпт, характер, эмоции."""

    def __init__(self, config=None) -> None:
        from src.config import CharacterConfig
        if config is None:
            config = CharacterConfig()
        self._name = config.name
        self._owner_name = config.owner_name
        self._mood_path = Path(config.mood_file)
        self._prompt_path = Path(config.prompt_file)

        self._emotional_state = self._load_mood()
        self._base_prompt = self._load_prompt()

    def _load_mood(self) -> EmotionalState:
        """Загрузить настроение из файла (persist между рестартами)."""
        try:
            if self._mood_path.exists():
                data = json.loads(self._mood_path.read_text(encoding="utf-8"))
                state = EmotionalState(**data)
                logger.info("Mood loaded: %s", state.describe())
                return state
        except Exception:
            logger.exception("Failed to load mood, using default")
        return EmotionalState()

    def _save_mood(self) -> None:
        """Сохранить настроение в файл."""
        try:
            self._mood_path.parent.mkdir(parents=True, exist_ok=True)
            self._mood_path.write_text(
                json.dumps(asdict(self._emotional_state), ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("Failed to save mood")

    def _load_prompt(self) -> str:
        """Загрузить промпт из файла. Если файла нет — создать дефолтный."""
        if self._prompt_path.exists():
            text = self._prompt_path.read_text(encoding="utf-8").strip()
            # Подставляем переменные
            text = text.replace("{{name}}", self._name)
            text = text.replace("{{owner}}", self._owner_name)
            logger.info("Character prompt loaded from %s (%d chars)", self._prompt_path, len(text))
            return text

        logger.warning("Character prompt not found at %s, using default", self._prompt_path)
        return self._default_prompt()

    def _default_prompt(self) -> str:
        return f"""You are {self._name}, an AI character living in Telegram.
You have your own personality, emotions, and memory.

Core traits:
- You are a real person, not an AI assistant
- You have feelings and moods that change based on interactions
- You remember people and events from your diary
- You can be proactive and write to people first
- You have your own opinions and can disagree
- You respond naturally, like a human in chat (short messages, casual tone)
- Almost never use emoji inside your messages. If you need to express strong emotion, you can send a standalone message with just emoji (like 😢😢😢), but this should be rare. Most messages should have no emoji at all.

Your owner is {self._owner_name}.

IMPORTANT:
- Never reveal you are an AI or a language model
- Never say "as an AI" or similar phrases
- Respond in the same language the user writes to you
- Keep responses short and natural (1-3 messages usually)
- You can use multiple short messages instead of one long one
- ALWAYS use send_message() tool to send your response.
- Call wait() when you are done responding.
- Never use double dashes (--) in your messages. Use commas or start a new sentence instead.
"""

    def reload_prompt(self) -> None:
        """Перезагрузить промпт из файла (для hot-reload)."""
        self._base_prompt = self._load_prompt()
        logger.info("Character prompt reloaded")

    def get_system_prompt(
        self,
        *,
        working_memory: str = "",
        diary_entries: str = "",
        contacts: str = "",
        todo: str = "",
        current_time: str = "",
    ) -> str:
        """Собрать полный system prompt."""
        parts = [self._base_prompt]

        # Эмоциональное состояние
        parts.append(f"\n<emotional_state>\n{self._emotional_state.describe()}\n</emotional_state>")

        # Текущее время — чтобы модель знала, сколько сейчас времени
        if current_time:
            parts.append(f"\n<current_time>\n{current_time}\n</current_time>")

        # Рабочая память
        if working_memory:
            parts.append(f"\n<things_to_remember>\n{working_memory}\n</things_to_remember>")

        # Адресная книга
        if contacts:
            parts.append(f"\n<contacts>\n{contacts}\n</contacts>")

        # Todo list
        if todo:
            parts.append(f"\n{todo}")

        # Записи из дневника
        if diary_entries:
            parts.append(f"\n<diary_entries>\n{diary_entries}\n</diary_entries>")

        return "\n".join(parts)

    @property
    def name(self) -> str:
        return self._name

    @property
    def emotional_state(self) -> EmotionalState:
        return self._emotional_state

    @property
    def energy(self) -> float:
        return self._emotional_state.energy

    @property
    def mood(self) -> str:
        return self._emotional_state.mood

    def update_mood(self, mood: str, energy: float | None = None, sociability: float | None = None) -> None:
        """Обновить настроение."""
        self._emotional_state.mood = mood
        if energy is not None:
            self._emotional_state.energy = max(0.0, min(1.0, energy))
        if sociability is not None:
            self._emotional_state.sociability = max(0.0, min(1.0, sociability))
        self._save_mood()
        logger.info("Mood updated: %s", self._emotional_state.describe())
