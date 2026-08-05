from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.llm.provider import Tool, ToolCall

if TYPE_CHECKING:
    from src.client.base import BaseTelegramClient
    from src.memory.diary import Diary
    from src.memory.rag import VectorStore
    from src.memory.contacts import Contacts
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
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        logger.exception("Tool execution failed: %s", name)
        return f"Error executing {name}: {e}"


async def _tool_send_message(args: dict, ctx: ToolContext) -> str:
    chat_id = args["chat_id"]
    text = args["text"]
    await ctx.client.send_message(chat_id, text)
    return f"Message sent to {chat_id}."


async def _tool_diary_write(args: dict, ctx: ToolContext) -> str:
    entry_text = args["entry"]
    entry = ctx.diary.add(entry_text, source="diary")

    # Добавляем эмбеддинг в векторную БД
    if ctx.llm:
        try:
            embedding = await ctx.llm.embed(entry_text)
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
