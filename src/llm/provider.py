from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Результат вызова tool'а от LLM."""
    id: str
    name: str
    arguments: str  # JSON строка


@dataclass
class ChatMessage:
    """Сообщение в диалоге."""
    role: str  # "system", "user", "assistant", "tool"
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class ChatResponse:
    """Ответ LLM."""
    content: str | None
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    model: str


@dataclass
class Tool:
    """Описание tool'а для LLM."""
    name: str
    description: str
    parameters: dict  # JSON Schema


class LLMProvider:
    """Универсальный LLM провайдер (OpenAI-совместимый API)."""

    def __init__(self, config=None, *, log_config=None) -> None:
        from src.config import LLMConfig
        if config is None:
            config = LLMConfig()
        self._api_key = config.api_key
        self._base_url = config.base_url
        self._model = config.model
        self._embedding_model = config.embedding_model

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

        # LLM logger
        self._llm_logger = None
        if log_config and log_config.enabled:
            from src.llm.log import LLMLogger
            self._llm_logger = LLMLogger(log_config)

        logger.info("LLM provider: model=%s, base_url=%s", self._model, self._base_url)

    @property
    def llm_logger(self):
        """Доступ к LLMLogger для логирования tool results извне."""
        return self._llm_logger

    # ── Chat Completion ────────────────────────────────────────

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
        reason: str = "",
        iteration: int = 0,
    ) -> ChatResponse:
        """Отправить сообщения в LLM и получить ответ.

        Args:
            messages: История диалога
            tools: Доступные tool'ы (function calling)
            temperature: Температура генерации
            reason: Причина вызова (для логирования)
            iteration: Номер итерации в цикле tool calls

        Returns:
            ChatResponse с текстом и/или tool calls
        """
        formatted = self._format_messages(messages)
        kwargs: dict = {
            "model": self._model,
            "messages": formatted,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = self._format_tools(tools)
            kwargs["tool_choice"] = "auto"

        # Log request
        if self._llm_logger:
            await self._llm_logger.log_request(messages, tools, temperature, reason, iteration)

        t0 = time.monotonic()
        response = await self._client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))

        result = ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=response.model,
        )

        # Log response
        if self._llm_logger:
            await self._llm_logger.log_response(result, reason, iteration, latency_ms)

        return result

    async def chat_streaming(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[Tool] | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Стриминг ответа LLM (для будущего использования)."""
        formatted = self._format_messages(messages)
        kwargs: dict = {
            "model": self._model,
            "messages": formatted,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = self._format_tools(tools)
            kwargs["tool_choice"] = "auto"

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    # ── Embeddings ─────────────────────────────────────────────

    async def embed(self, text: str) -> list[float] | None:
        """Получить эмбеддинг для текста. Возвращает None если не поддерживается."""
        try:
            t0 = time.monotonic()
            response = await self._client.embeddings.create(
                model=self._embedding_model,
                input=text,
            )
            latency_ms = (time.monotonic() - t0) * 1000

            if self._llm_logger:
                await self._llm_logger.log_embedding(text, latency_ms)

            return response.data[0].embedding
        except Exception as e:
            logger.warning("Embeddings not available: %s", e)
            return None

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Получить эмбеддинги для списка текстов. Возвращает None если не поддерживается."""
        try:
            t0 = time.monotonic()
            response = await self._client.embeddings.create(
                model=self._embedding_model,
                input=texts,
            )
            latency_ms = (time.monotonic() - t0) * 1000

            if self._llm_logger:
                for text in texts:
                    await self._llm_logger.log_embedding(text, latency_ms / len(texts))

            return [item.embedding for item in response.data]
        except Exception as e:
            logger.warning("Embeddings not available: %s", e)
            return None

    # ── Formatting helpers ─────────────────────────────────────

    def _format_messages(self, messages: list[ChatMessage]) -> list[dict]:
        result = []
        for msg in messages:
            entry: dict = {"role": msg.role}
            if msg.content is not None:
                entry["content"] = msg.content
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            result.append(entry)
        return result

    def _format_tools(self, tools: list[Tool]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
