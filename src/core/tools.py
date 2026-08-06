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
    config: object = None  # Config
    temporary_context: list = field(default_factory=list)
    messages_in_a_row: int = field(default=0)
    anti_repeat: object = None  # AntiRepeat


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
                    "reply_to_message_id": {
                        "type": "integer",
                        "description": "Message ID to reply to (from get_chat_context). Omit if not replying.",
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
            ctx.messages_in_a_row = 0
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
    reply_to = args.get("reply_to_message_id")

    # Anti-repeat check
    if ctx.anti_repeat:
        repeat_error = await ctx.anti_repeat.check(chat_id, text)
        if repeat_error:
            return f"Error: {repeat_error}"

    # Запускаем фоновый typing indicator (каждые 3 сек)
    typing_stop = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(ctx.client, chat_id, typing_stop))

    try:
        # Typing simulation delay (имитация набора текста)
        await _simulate_typing_delay(text, ctx.config)

        # Typo simulation (имитация опечаток)
        text = _apply_typos(text)

        # Разбиваем многострочные сообщения
        lines = [l for l in text.split("\n") if l.strip()]
        if len(lines) > 1:
            for i, line in enumerate(lines):
                if i > 0:
                    await _simulate_typing_delay(line)
                    reply_to = None  # reply only to first message
                sent = await ctx.client.send_message(chat_id, line, reply_to=reply_to)
                _save_outgoing(ctx, chat_id, sent.message_id, line)
                if ctx.anti_repeat:
                    await ctx.anti_repeat.record(chat_id, line)
            return _build_send_result(ctx, chat_id, len(lines))
        else:
            sent = await ctx.client.send_message(chat_id, text, reply_to=reply_to)
            _save_outgoing(ctx, chat_id, sent.message_id, text)
            if ctx.anti_repeat:
                await ctx.anti_repeat.record(chat_id, text)
            return _build_send_result(ctx, chat_id, 1)
    finally:
        # Останавливаем typing indicator
        typing_stop.set()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


def _build_send_result(ctx: ToolContext, chat_id: int, count: int) -> str:
    """Построить результат отправки с follow-up промптом."""
    ctx.messages_in_a_row += count
    result = f"Sent {count} message(s) to {chat_id}."

    # Follow-up: шанс отправить ещё одно сообщение
    if ctx.config:
        chance = ctx.config.behavior.follow_up_chance
        max_count = ctx.config.behavior.follow_up_max
    else:
        chance = 0.3
        max_count = 3

    if ctx.messages_in_a_row < max_count and random.random() < chance:
        result += "\n\nYou should add a follow-up #send_telegram_message."
    elif ctx.messages_in_a_row >= max_count:
        result += f"\n\nWarning: you have sent {ctx.messages_in_a_row} messages in a row! Give your participant space to breathe!"

    return result


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


async def _simulate_typing_delay(text: str, config=None) -> None:
    """Имитация набора текста — задержка в зависимости от длины сообщения."""
    from src.config import BehaviorConfig
    if config is None:
        cfg = BehaviorConfig()
    else:
        cfg = config.behavior
    wpm = random.uniform(cfg.typing_wpm_min, cfg.typing_wpm_max)
    chars_per_sec = wpm * 5 / 60
    delay = len(text) / chars_per_sec
    delay = min(delay, cfg.typing_max_delay)
    delay = max(delay, cfg.typing_min_delay)
    logger.debug("Typing delay: %.1fs for %d chars", delay, len(text))
    await asyncio.sleep(delay)


KEYBOARD_NEIGHBORS = {
    'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etfds', 't': 'ryghs', 'y': 'tughj',
    'u': 'yihjk', 'i': 'uojkl', 'o': 'iplk', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv',
    'h': 'gyujnb', 'j': 'huiknm', 'k': 'jiolm', 'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}


def _apply_typos(text: str) -> str:
    """Случайные опечатки: swap соседних букв или замена на соседнюю клавишу."""
    if len(text) < 3:
        return text

    result = list(text)

    # 2% шанс: swap двух соседних символов
    if random.random() < 0.02:
        idx = random.randint(0, len(result) - 2)
        result[idx], result[idx + 1] = result[idx + 1], result[idx]

    # 2% шанс: замена на соседнюю клавишу
    if random.random() < 0.02:
        idx = random.randint(0, len(result) - 1)
        char = result[idx].lower()
        if char in KEYBOARD_NEIGHBORS:
            neighbors = KEYBOARD_NEIGHBORS[char]
            replacement = random.choice(neighbors)
            result[idx] = replacement if result[idx].islower() else replacement.upper()

    return "".join(result)


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
        lines.append(f"[id={m.message_id} {role}]: {m.text}")
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
