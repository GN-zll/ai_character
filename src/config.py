from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import toml
from dotenv import load_dotenv

CONFIG_PATH = "config.toml"


@dataclass
class TelegramConfig:
    client_type: str = "bot"
    whitelist: list[int] = field(default_factory=list)


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    embedding_model: str = "text-embedding-3-small"


@dataclass
class CharacterConfig:
    name: str = "Yuki"
    owner_name: str = "master"
    owner_chat_id: int = 0
    prompt_file: str = "data/character_base.md"


@dataclass
class MemoryConfig:
    diary_dir: str = "data/diary"
    vectors_dir: str = "data/vectors"
    working_memory_file: str = "data/working_memory.md"
    contacts_file: str = "data/contacts.json"
    history_db: str = "data/history.db"
    history_max_per_chat: int = 1000
    stat_levels_file: str = "data/stat_levels.json"
    reminders_file: str = "data/reminders.json"


@dataclass
class BehaviorConfig:
    proactive_interval_min: int = 27
    proactive_chance: float = 0.5
    batch_delay_min: float = 0.0
    batch_delay_max: float = 5.0
    diary_token_trigger: int = 20000
    worker_max_iterations: int = 20
    typing_wpm_min: int = 100
    typing_wpm_max: int = 300
    typing_min_delay: float = 2.0
    typing_max_delay: float = 15.0
    follow_up_chance: float = 0.3
    follow_up_max: int = 3
    anti_repeat_threshold: float = 0.75
    anti_repeat_max_history: int = 20
    typo_swap_chance: float = 0.02
    typo_neighbor_chance: float = 0.02
    typo_correct_chance: float = 0.57
    typo_correct_delay_min: float = 15.0
    typo_correct_delay_max: float = 45.0
    random_sleep_chance: float = 0.01
    random_sleep_min: int = 15
    random_sleep_max: int = 120
    miss_notification_chance: float = 0.05
    notification_preview_length: int = 10
    thinking_delay_min: float = 1.0
    thinking_delay_max: float = 3.0
    batch_window_min: float = 2.0
    batch_window_max: float = 5.0


@dataclass
class NightConfig:
    enabled: bool = True
    start_hour: int = 0
    end_hour: int = 8
    msk_offset: int = 3


@dataclass
class RelationshipStatConfig:
    name: str = ""
    description: str = ""


@dataclass
class StatLevel:
    min: int = 0
    max: int = 0
    label: str = ""


@dataclass
class StatLevelsConfig:
    """Загруженные уровни статов из stat_levels.json."""
    levels: dict[str, list[StatLevel]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = "data/stat_levels.json") -> StatLevelsConfig:
        """Загрузить уровни статов из JSON файла."""
        config = cls()
        if not Path(path).exists():
            return config
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for stat_name, stat_data in data.items():
            config.levels[stat_name] = [
                StatLevel(min=lvl["min"], max=lvl["max"], label=lvl["label"])
                for lvl in stat_data.get("levels", [])
            ]
        return config

    def get_label(self, stat_name: str, value: int) -> str:
        """Получить текстовый label для значения стата."""
        levels = self.levels.get(stat_name, [])
        for lvl in levels:
            if lvl.min <= value < lvl.max:
                return lvl.label
        # Граничное значение
        if levels and value >= levels[-1].max:
            return levels[-1].label
        if levels and value <= levels[0].min:
            return levels[0].label
        return "neutral"


@dataclass
class Config:
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    character: CharacterConfig = field(default_factory=CharacterConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    night: NightConfig = field(default_factory=NightConfig)
    relationship_stats: list[RelationshipStatConfig] = field(default_factory=list)
    stat_levels: StatLevelsConfig = field(default_factory=StatLevelsConfig)

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> Config:
        """Загрузить конфиг из config.toml + секреты из .env."""
        load_dotenv()

        # Загружаем config.toml
        data = {}
        if Path(path).exists():
            data = toml.load(path)

        # Секреты из .env
        telegram = TelegramConfig(**data.get("telegram", {}))
        telegram.client_type = os.getenv("TELEGRAM_CLIENT_TYPE", telegram.client_type)

        # Whitelist из .env (comma-separated)
        whitelist_str = os.getenv("WHITELIST_CHAT_IDS", "")
        if whitelist_str.strip():
            telegram.whitelist = [int(x.strip()) for x in whitelist_str.split(",") if x.strip()]

        llm = LLMConfig(**data.get("llm", {}))
        llm.api_key = os.getenv("LLM_API_KEY", llm.api_key)

        character = CharacterConfig(**data.get("character", {}))
        # Owner info из .env
        character.owner_name = os.getenv("OWNER_NAME", character.owner_name)
        owner_id_str = os.getenv("OWNER_CHAT_ID", "0")
        character.owner_chat_id = int(owner_id_str) if owner_id_str.strip() else 0
        memory = MemoryConfig(**data.get("memory", {}))
        behavior = BehaviorConfig(**data.get("behavior", {}))
        night = NightConfig(**data.get("night", {}))

        # Relationship stats
        rel_stats = []
        for item in data.get("relationship_stats", []):
            rel_stats.append(RelationshipStatConfig(
                name=item.get("name", ""),
                description=item.get("description", ""),
            ))

        # Stat levels
        stat_levels = StatLevelsConfig.load(memory.stat_levels_file)

        return cls(
            telegram=telegram,
            llm=llm,
            character=character,
            memory=memory,
            behavior=behavior,
            night=night,
            relationship_stats=rel_stats,
            stat_levels=stat_levels,
        )
