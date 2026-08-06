from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.llm.provider import Tool, ToolCall

if TYPE_CHECKING:
    from src.client.base import BaseTelegramClient
    from src.memory.diary import Diary
    from src.memory.rag import VectorStore
    from src.memory.contacts import Contacts
    from src.memory.chat_history import ChatHistory
    from src.core.notification import NotificationManager
    from src.llm.provider import ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Контекст для выполнения tool'ов."""
    client: BaseTelegramClient
    diary: Diary
    vector_store: VectorStore
    contacts: object = None  # Contacts
    chat_history: object = None  # ChatHistory
    notification_manager: NotificationManager = None
    llm: object = None  # LLMProvider (avoid circular import)
    temporary_context: list = field(default_factory=list)


def build_tools() -> list[Tool]:
    """Построить список tool'ов для LLM."""
    return [
        Tool(
            name="send_message",
            description="Send a text message to a Telegram chat. Use this to reply to people.",
            parameters={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "integer",
                        "description": "The Telegram chat ID to send the message to",
                    },
                    "text": {
                        "type": "string",
                        "description": "The message text to send",
                    },
                },
                "required": ["chat_id", "text"],
            },
        ),
        Tool(
            name="diary_write",
            description="Write a thought, event, or reflection to your diary for long-term memory.",
            parameters={
                "type": "object",
                "properties": {
                    "entry": {
                        "type": "string",
                        "description": "The diary entry text",
                    },
                },
                "required": ["entry"],
            },
        ),
        Tool(
            name="ask",
            description="Search your diary/memory for related information about a topic.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in your memory",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="wait",
            description="Wait for the next notification. Use when you have nothing more to do right now.",
            parameters={"type": "object", "properties": {}},
        ),
        Tool(
            name="pause",
            description="Pause and wait for the next event. Same as wait.",
            parameters={"type": "object", "properties": {}},
        ),
        Tool(
            name="contacts_get",
            description="Get info about a contact by chat_id, or list all contacts if no chat_id given.",
            parameters={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "integer",
                        "description": "Chat ID to look up. Omit to list all contacts.",
                    },
                },
            },
        ),
        Tool(
            name="contacts_update",
            description="Create or update a contact in your address book. Set a name, description, or tags for a chat_id.",
            parameters={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "integer",
                        "description": "The chat ID",
                    },
                    "name": {
                        "type": "string",
                        "description": "Contact's name/nickname",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of who this person is",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags like 'friend', 'family', 'work'",
                    },
                },
                "required": ["chat_id"],
            },
        ),
        Tool(
            name="get_chat_context",
            description="Get recent messages from a chat to understand the conversation context. Use before writing to someone.",
            parameters={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "integer",
                        "description": "Chat ID to get context for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent messages (default 20)",
                    },
                },
                "required": ["chat_id"],
            },
        ),
        Tool(
            name="get_chats",
            description="List all chats you can write to, with last activity time. Use to decide who to write to proactively.",
            parameters={"type": "object", "properties": {}},
        ),
    ]


async def execute_tool(tool_call: ToolCall, ctx: ToolContext) -> str:
    """Выполнить tool и вернуть результат."""
    name = tool_call.name
    args = json.loads(tool_call.arguments) if tool_call.arguments else {}

    logger.info("Executing tool: %s(%s)", name, json.dumps(args, ensure_ascii=False)[:200])

    try:
        if name == "send_message":
            return await _tool_send_message(args, ctx)
        elif name == "diary_write":
            return await _tool_diary_write(args, ctx)
        elif name == "ask":
            return await _tool_ask(args, ctx)
        elif name in ("wait", "pause"):
            return "Waiting for next notification."
        elif name == "contacts_get":
            return _tool_contacts_get(args, ctx)
        elif name == "contacts_update":
            return _tool_contacts_update(args, ctx)
        elif name == "get_chat_context":
            return _tool_get_chat_context(args, ctx)
        elif name == "get_chats":
            return _tool_get_chats(args, ctx)
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        logger.exception("Tool execution failed: %s", name)
        return f"Error executing {name}: {e}"


async def _tool_send_message(args: dict, ctx: ToolContext) -> str:
    chat_id = args["chat_id"]
    text = args["text"]

    # Запускаем фоновый typing indicator (каждые 4 сек)
    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(ctx.client, chat_id, typing_stop))

    try:
        # Typing simulation delay (имитация набора текста)
        await _simulate_typing_delay(text)

        # Разбиваем многострочные сообщения
        lines = [l for l in text.split("\n") if l.strip()]
        if len(lines) > 1:
            for i, line in enumerate(lines):
                if i > 0:
                    await _simulate_typing_delay(line)
                sent = await ctx.client.send_message(chat_id, line)
                _save_outgoing(ctx, chat_id, sent.message_id, line)
            return f"Sent {len(lines)} messages to {chat_id}."
        else:
            sent = await ctx.client.send_message(chat_id, text)
            _save_outgoing(ctx, chat_id, sent.message_id, text)
            return f"Message sent to {chat_id}."
    finally:
        # Останавливаем typing indicator
        typing_stop.set()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


async def _typing_loop(client, chat_id: int, stop: asyncio.Event) -> None:
    """Посылает typing indicator каждые 3 сек, пока не вызван stop."""
    try:
        while not stop.is_set():
            try:
                await client.send_chat_action(chat_id, "typing")
            except Exception:
                pass
            await asyncio.sleep(3)
    except asyncio.CancelledError:
        pass


async def _simulate_typing_delay(text: str) -> None:
    """Имитация набора текста — задержка в зависимости от длины сообщения."""
    min_wpm = 100
    max_wpm = 300
    wpm = random.uniform(min_wpm, max_wpm)
    chars_per_sec = wpm * 5 / 60  # ~8-25 символа/сек
    delay = len(text) / chars_per_sec
    delay = min(delay, 15.0)  # макс 15 сек
    delay = max(delay, 2.0)   # мин 2 сек
    logger.debug("Typing delay: %.1fs for %d chars", delay, len(text))
    await asyncio.sleep(delay)


async def _tool_diary_write(args: dict, ctx: ToolContext) -> str:
    entry_text = args["entry"]
    entry = ctx.diary.add(entry_text, source="diary")

    # Добавляем эмбеддинг в векторную БД
    if ctx.llm:
        try:
            embedding = await ctx.llm.embed(entry_text)
            if embedding is not None:
                ctx.vector_store.add(
                    text=entry_text,
                    embedding=embedding,
                    metadata={"entry_id": entry.id, "source": "diary"},
                )
        except Exception:
            logger.exception("Failed to embed diary entry")

    return f"Diary entry saved: {entry.id}"


async def _tool_ask(args: dict, ctx: ToolContext) -> str:
    query = args["query"]

    if not ctx.llm:
        return "Search unavailable (no LLM)."

    try:
        embedding = await ctx.llm.embed(query)
        if embedding is None:
            return "Search unavailable (embeddings not supported by this provider)."
        results = ctx.vector_store.query(embedding, n_results=5, max_distance=0.5)

        if not results:
            return f"No memories found for '{query}'."

        entries = []
        for r in results:
            entries.append(f"[{r.id}] (distance={r.distance:.3f}) {r.text}")
        return "Found memories:\n" + "\n---\n".join(entries)
    except Exception as e:
        return f"Search failed: {e}"


def _tool_contacts_get(args: dict, ctx: ToolContext) -> str:
    if not ctx.contacts:
        return "Contacts not available."

    chat_id = args.get("chat_id")
    if chat_id is not None:
        contact = ctx.contacts.get(chat_id)
        if not contact:
            return f"No contact found for chat_id={chat_id}."
        return f"Contact: id={contact.chat_id}, name={contact.name}, description={contact.description}, tags={contact.tags}"

    # List all
    contacts = ctx.contacts.list_all()
    if not contacts:
        return "Address book is empty."

    lines = []
    for c in contacts:
        line = f"- {c.chat_id}: {c.name}"
        if c.description:
            line += f" — {c.description}"
        if c.tags:
            line += f" [{', '.join(c.tags)}]"
        lines.append(line)
    return "Contacts:\n" + "\n".join(lines)


def _tool_contacts_update(args: dict, ctx: ToolContext) -> str:
    if not ctx.contacts:
        return "Contacts not available."

    chat_id = args["chat_id"]
    name = args.get("name")
    description = args.get("description")
    tags = args.get("tags")

    contact = ctx.contacts.update(chat_id, name=name, description=description, tags=tags)
    return f"Contact updated: {contact.chat_id} = {contact.name}"


def _save_outgoing(ctx: ToolContext, chat_id: int, message_id: int, text: str) -> None:
    """Сохранить исходящее сообщение в историю."""
    if ctx.chat_history:
        try:
            me = ctx.contacts.get_or_default(0) if ctx.contacts else None
            ctx.chat_history.add_message(
                chat_id=chat_id,
                message_id=message_id,
                sender_id=0,
                sender_name=me.name if me else "bot",
                text=text,
                is_outgoing=True,
            )
        except Exception:
            logger.exception("Failed to save outgoing message")


def _tool_get_chat_context(args: dict, ctx: ToolContext) -> str:
    if not ctx.chat_history:
        return "Chat history not available."

    chat_id = args["chat_id"]
    limit = args.get("limit", 20)

    messages = ctx.chat_history.get_messages(chat_id, limit=limit)
    if not messages:
        return f"No messages found for chat {chat_id}."

    lines = []
    for m in messages:
        role = "me" if m.is_outgoing else m.sender_name
        lines.append(f"[{role}]: {m.text}")
    return "\n".join(lines)


def _tool_get_chats(args: dict, ctx: ToolContext) -> str:
    if not ctx.chat_history or not ctx.contacts:
        return "Chat history or contacts not available."

    chats = ctx.chat_history.get_all_chats()
    contacts = {c.chat_id: c for c in ctx.contacts.list_all()}

    if not chats:
        return "No chats yet."

    lines = []
    for chat in chats:
        cid = chat["chat_id"]
        contact = contacts.get(cid)
        name = contact.name if contact else f"User#{cid}"
        desc = f" — {contact.description}" if contact and contact.description else ""
        lines.append(f"- {cid}: {name}{desc} (messages: {chat['message_count']}, last: {chat['last_activity'][:16]})")
    return "Chats:\n" + "\n".join(lines)
