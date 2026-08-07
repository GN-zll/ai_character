from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import LLMLogConfig
    from src.llm.provider import ChatMessage, ChatResponse, Tool

logger = logging.getLogger(__name__)


class LLMLogger:
    """Логгер LLM-вызовов в JSONL формате.

    Каждая строка в файле — валидный JSON с полями:
    - ts: ISO timestamp
    - type: request | response | tool_result | embedding
    - reason: причина вызова (incoming_message, alarm, reminder, ...)
    - + данные специфичные для типа
    """

    def __init__(self, config: LLMLogConfig) -> None:
        self._config = config
        self._path = Path(config.file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info("LLM logger: file=%s", self._path)

    # ── Request ────────────────────────────────────────────────

    async def log_request(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] | None,
        temperature: float,
        reason: str,
        iteration: int,
    ) -> None:
        if not self._config.log_messages and not self._config.log_system_prompt:
            return

        entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "request",
            "reason": reason,
            "iteration": iteration,
        }

        if self._config.log_model:
            entry["temperature"] = temperature

        if self._config.log_tokens:
            entry["estimated_input_chars"] = sum(len(m.content or "") for m in messages)

        # Формируем messages для лога
        if self._config.log_messages:
            formatted = []
            for msg in messages:
                msg_entry: dict = {"role": msg.role}
                if self._config.log_system_prompt or msg.role != "system":
                    if msg.content is not None:
                        msg_entry["content"] = msg.content
                elif msg.role == "system":
                    msg_entry["content"] = f"({len(msg.content or '')} chars)"

                if msg.tool_calls:
                    msg_entry["tool_calls"] = [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in msg.tool_calls
                    ]
                if msg.tool_call_id:
                    msg_entry["tool_call_id"] = msg.tool_call_id
                formatted.append(msg_entry)
            entry["messages"] = formatted

        if self._config.log_tool_calls and tools:
            entry["tools"] = [
                {"name": t.name, "description": t.description}
                for t in tools
            ]

        await self._write(entry)

    # ── Response ───────────────────────────────────────────────

    async def log_response(
        self,
        response: ChatResponse,
        reason: str,
        iteration: int,
        latency_ms: float,
    ) -> None:
        if not self._config.log_response:
            return

        entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "response",
            "reason": reason,
            "iteration": iteration,
        }

        if self._config.log_model:
            entry["model"] = response.model

        if self._config.log_response:
            entry["content"] = response.content

        if self._config.log_tool_calls and response.tool_calls:
            entry["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ]

        if self._config.log_tokens:
            entry["input_tokens"] = response.input_tokens
            entry["output_tokens"] = response.output_tokens

        if self._config.log_latency:
            entry["latency_ms"] = round(latency_ms, 1)

        await self._write(entry)

    # ── Tool Result ────────────────────────────────────────────

    async def log_tool_result(
        self,
        tool_call_id: str,
        name: str,
        arguments: str,
        result: str,
    ) -> None:
        if not self._config.log_tool_results:
            return

        entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "tool_result",
            "tool_call_id": tool_call_id,
            "name": name,
        }

        try:
            entry["arguments"] = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            entry["arguments_raw"] = arguments

        entry["result"] = result

        await self._write(entry)

    # ── Embedding ──────────────────────────────────────────────

    async def log_embedding(self, text: str, latency_ms: float) -> None:
        if not self._config.log_embeddings:
            return

        entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "embedding",
            "text_chars": len(text),
        }

        if self._config.log_latency:
            entry["latency_ms"] = round(latency_ms, 1)

        await self._write(entry)

    # ── Internal ───────────────────────────────────────────────

    async def _write(self, entry: dict) -> None:
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
            async with self._lock:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            logger.exception("Failed to write LLM log")
