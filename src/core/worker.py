from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timezone

from src.client.base import BaseTelegramClient
from src.llm.provider import LLMProvider, ChatMessage, ToolCall
from src.memory.diary import Diary
from src.memory.rag import VectorStore
from src.memory.working_memory import WorkingMemory
from src.memory.contacts import Contacts
from src.character.personality import Personality
from src.core.notification import Notification, NotificationManager
from src.core.tools import build_tools, execute_tool, ToolContext

logger = logging.getLogger(__name__)

DIARY_TOKEN_TRIGGER = 20000


class Worker:
    """Worker — корутина, обрабатывающая нотификации через LLM.

    Аналог Worker из kuni: берёт нотификацию → diary lookup →
    LLM с tool calls → обработка → повторяет до wait/pause.
    """

    def __init__(
        self,
        *,
        name: str,
        client: BaseTelegramClient,
        llm: LLMProvider,
        diary: Diary,
        vector_store: VectorStore,
        working_memory: WorkingMemory,
        contacts: Contacts,
        personality: Personality,
        notification_manager: NotificationManager,
    ) -> None:
        self._name = name
        self._client = client
        self._llm = llm
        self._diary = diary
        self._vector_store = vector_store
        self._working_memory = working_memory
        self._contacts = contacts
        self._personality = personality
        self._notification_manager = notification_manager

        self._temporary_context: list[ChatMessage] = []
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        """Запустить worker в фоновой задаче."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Worker %s started", self._name)

    async def stop(self) -> None:
        """Остановить worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Worker %s stopped", self._name)

    async def _loop(self) -> None:
        """Основной цикл worker'а."""
        while self._running:
            try:
                notification = await self._notification_manager.next()
                await self._handle_notification(notification)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Worker %s error", self._name)
                await asyncio.sleep(1)

    async def _handle_notification(self, notification: Notification) -> None:
        """Обработать одну нотификацию."""
        logger.info("Worker %s processing: %s", self._name, notification.message[:80])

        # Добавляем входящее в контекст
        self._temporary_context.append(ChatMessage(
            role="user",
            content=notification.message,
        ))

        # Diary lookup — ищем связанные записи
        diary_context = await self._lookup_diary()

        # Собираем system prompt
        system_prompt = self._personality.get_system_prompt(
            working_memory=self._working_memory.get(),
            diary_entries=diary_context,
            contacts=self._contacts.format_for_prompt(),
        )

        # Tool'ы для LLM
        tools = build_tools()
        tool_ctx = ToolContext(
            client=self._client,
            diary=self._diary,
            vector_store=self._vector_store,
            contacts=self._contacts,
            notification_manager=self._notification_manager,
            llm=self._llm,
            temporary_context=self._temporary_context,
        )

        # Цикл LLM с tool calls (как в kuni)
        max_iterations = 20
        for _ in range(max_iterations):
            messages = [ChatMessage(role="system", content=system_prompt)] + self._temporary_context

            response = await self._llm.chat(messages=messages, tools=tools)

            if response.content:
                logger.info("LLM response: %s", response.content[:200])

            # Добавляем ответ в контекст
            self._temporary_context.append(ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))

            if not response.tool_calls:
                # LLM не вызвал tool'ы — принудительно вызываем wait
                self._temporary_context.append(ChatMessage(
                    role="user",
                    content="You need to call a tool. Use `wait` to finish this turn.",
                ))
                continue

            # Обрабатываем tool calls
            should_break = False
            for tc in response.tool_calls:
                result = await execute_tool(tc, tool_ctx)

                self._temporary_context.append(ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                ))

                if tc.name in ("wait", "pause"):
                    should_break = True

            if should_break:
                break

            # Проверяем размер контекста
            total_chars = sum(len(m.content or "") for m in self._temporary_context)
            if total_chars > DIARY_TOKEN_TRIGGER * 3:  # грубая оценка
                logger.info("Context overflow, dumping to diary")
                await self._dump_to_diary()
                break

        # Проверяем переполнение после обработки
        total_chars = sum(len(m.content or "") for m in self._temporary_context)
        if total_chars > DIARY_TOKEN_TRIGGER * 3:
            await self._dump_to_diary()

    async def _lookup_diary(self) -> str:
        """Поиск связанных записей в дневнике по последним сообщениям."""
        if not self._temporary_context:
            return ""

        # Берём последние 3 сообщения для поиска
        recent = [m.content for m in self._temporary_context[-3:] if m.content]
        if not recent:
            return ""

        query_text = "\n".join(recent)
        try:
            embedding = await self._llm.embed(query_text)
            if embedding is None:
                return ""
            results = self._vector_store.query(embedding, n_results=5, max_distance=0.5)

            if not results:
                return ""

            entries = []
            for r in results:
                entries.append(f"[{r.id}] {r.text}")
            return "\n---\n".join(entries)
        except Exception:
            logger.exception("Diary lookup failed")
            return ""

    async def _dump_to_diary(self) -> None:
        """Сжать контекст в записи дневника и очистить."""
        if not self._temporary_context:
            return

        logger.info("Dumping context to diary (%d messages)", len(self._temporary_context))

        # Просим LLM сжать контекст в дневник
        summary_prompt = ChatMessage(
            role="user",
            content=(
                "Summarize this conversation into a diary entry for long-term memory. "
                "Focus on: facts about people, emotional events, promises, important information. "
                "Write in third person. Be concise but preserve key details.\n\n"
                "Conversation:\n" +
                "\n".join(
                    f"[{m.role}] {m.content}"
                    for m in self._temporary_context
                    if m.content
                )
            ),
        )

        response = await self._llm.chat(
            messages=[
                ChatMessage(role="system", content="You are a memory compressor. Write concise diary entries."),
                summary_prompt,
            ]
        )

        if response.content:
            entry = self._diary.add(response.content, source="chat_dump")

            # Добавляем эмбеддинг в векторную БД
            try:
                embedding = await self._llm.embed(response.content)
                self._vector_store.add(
                    text=response.content,
                    embedding=embedding,
                    metadata={"entry_id": entry.id, "source": "chat_dump"},
                )
            except Exception:
                logger.exception("Failed to add diary entry to vector store")

        # Очищаем контекст
        self._temporary_context.clear()
        logger.info("Context dumped and cleared")
