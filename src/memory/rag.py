from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Результат поиска по векторной БД."""
    id: str
    text: str
    distance: float
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """Обёртка над ChromaDB для векторного поиска."""

    def __init__(self, persist_dir: str | Path = "data/vectors") -> None:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="diary",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store: %s (count=%s)", persist_dir, self._collection.count())

    def add(self, text: str, embedding: list[float], metadata: dict | None = None) -> str:
        """Добавить запись в векторную БД."""
        entry_id = str(uuid.uuid4())
        self._collection.add(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
        )
        return entry_id

    def query(
        self,
        embedding: list[float],
        n_results: int = 10,
        *,
        min_distance: float = 0.0,
        max_distance: float = 1.0,
    ) -> list[SearchResult]:
        """Поиск ближайших записей по эмбеддингу."""
        if self._collection.count() == 0:
            return []

        n = min(n_results, self._collection.count())
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "distances", "metadatas"],
        )

        items = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            if distance < min_distance or distance > max_distance:
                continue
            items.append(SearchResult(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                distance=distance,
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            ))
        return items

    def delete(self, entry_id: str) -> None:
        """Удалить запись."""
        self._collection.delete(ids=[entry_id])

    def count(self) -> int:
        """Количество записей."""
        return self._collection.count()
