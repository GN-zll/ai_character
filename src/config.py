from __future__ import annotations

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


@dataclass
class BehaviorConfig:
    proactive_interval_min: int = 27
    proactive_chance: float = 0.5
    batch_delay_min: float = 0.0
    batch_delay_max: float = 5.0
    diary_token_trigger: int = 20000
    worker_max_iterations: int = 20


@dataclass
class NightConfig:
    enabled: bool = True
    start_hour: int = 0
    end_hour: int = 8


@dataclass
class Config:
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    character: CharacterConfig = field(default_factory=CharacterConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    night: NightConfig = field(default_factory=NightConfig)

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

        llm = LLMConfig(**data.get("llm", {}))
        llm.api_key = os.getenv("LLM_API_KEY", llm.api_key)

        character = CharacterConfig(**data.get("character", {}))
        memory = MemoryConfig(**data.get("memory", {}))
        behavior = BehaviorConfig(**data.get("behavior", {}))
        night = NightConfig(**data.get("night", {}))

        return cls(
            telegram=telegram,
            llm=llm,
            character=character,
            memory=memory,
            behavior=behavior,
            night=night,
        )
