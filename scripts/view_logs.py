#!/usr/bin/env python3
"""Интерактивный просмотрщик LLM логов (JSONL).

Навигация:
  n / Enter  — следующая запись
  p          — предыдущая
  N          — следующее начало цепочки (новая нотификация: сообщение, будильник и т.д.)
  P          — предыдущее начало цепочки
  <num>G     — перейти к записи #num
  f <filter> — фильтр (f type:response, f reason:alarm, f clear)
  s          — сводка по текущему фильтру
  q          — выход
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

DEFAULT_FILE = "data/llm_logs.jsonl"

REASON_LABELS = {
    "incoming_message": "💬 Сообщение",
    "alarm_wake_up": "⏰ Будильник",
    "reminder": "🔔 Напоминание",
    "wait_checkin": "⏳ Check-in",
    "proactive": "🚀 Proactive",
    "continuation:tool_calls": "🔄 Продолжение",
    "diary_dump": "📝 Diary dump",
    "working_memory_update": "🧠 WM update",
    "sleep:relationship_consolidation": "😴 Отношения",
    "sleep:diary_consolidation": "😴 Дневник",
    "sleep:working_memory_update": "😴 Память",
}

TYPE_ICONS = {
    "request": "📤",
    "response": "📥",
    "tool_result": "🔧",
    "embedding": "📊",
}


def load_entries(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                console.print(f"[red]Bad JSON line skipped:[/red] {line[:80]}")
    return entries


def filter_entries(entries: list[dict], filt: str) -> list[dict]:
    if not filt or filt == "clear":
        return entries

    key, _, val = filt.partition(":")
    if not val:
        return entries

    key = key.strip().lower()
    val = val.strip().lower()

    result = []
    for e in entries:
        if key == "type" and e.get("type", "").lower() == val:
            result.append(e)
        elif key == "reason" and val in e.get("reason", "").lower():
            result.append(e)
        elif key == "name" and val in e.get("name", "").lower():
            result.append(e)
    return result


# Reasons that mark the START of a new interaction chain
# (as opposed to continuation/internal steps within a chain)
_CHAIN_START_REASONS = {
    "incoming_message",
    "alarm_wake_up",
    "reminder",
    "wait_checkin",
    "proactive",
    "mood_reflection",
}

# Reasons that are internal steps WITHIN a chain (continuations, dumps, sleep)
_NON_CHAIN_START_REASONS = {
    "continuation:tool_calls",
    "diary_dump",
    "working_memory_update",
}


def is_chain_start(entry: dict) -> bool:
    """Начало ли entry новой цепочки (новая нотификация)."""
    if entry.get("type") != "request":
        return False
    reason = entry.get("reason", "")
    if not reason:
        return False
    if reason in _NON_CHAIN_START_REASONS or reason.startswith("sleep:"):
        return False
    return True


def find_next_chain_start(entries: list[dict], pos: int) -> int:
    """Найти позицию следующего начала цепочки (строго дальше pos)."""
    for i in range(pos + 1, len(entries)):
        if is_chain_start(entries[i]):
            return i
    return pos


def find_prev_chain_start(entries: list[dict], pos: int) -> int:
    """Найти позицию предыдущего начала цепочки (строго раньше pos)."""
    for i in range(pos - 1, -1, -1):
        if is_chain_start(entries[i]):
            return i
    return pos


def format_header(entry: dict, pos: int, total: int, filt: str) -> Text:
    ts = entry.get("ts", "?")[:19]
    etype = entry.get("type", "?")
    reason = entry.get("reason", "")
    iteration = entry.get("iteration", "")

    icon = TYPE_ICONS.get(etype, "❓")
    reason_label = REASON_LABELS.get(reason, reason)

    header = Text()
    header.append(f"  [{pos}/{total}]  ", style="bold white")
    header.append(f"{icon} {etype}", style="bold cyan")
    if reason:
        header.append(f"  │  {reason_label}", style="yellow")
    if iteration != "":
        header.append(f"  │  iter={iteration}", style="dim")
    header.append(f"  │  {ts}", style="dim")
    if filt:
        header.append(f"  │  filter: {filt}", style="dim magenta")
    return header


def format_request(entry: dict) -> Panel:
    parts = []

    # Messages
    messages = entry.get("messages", [])
    if messages:
        parts.append("[bold]Messages:[/bold]")
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            role_style = {
                "system": "bold magenta",
                "user": "bold green",
                "assistant": "bold blue",
                "tool": "bold yellow",
            }.get(role, "white")

            parts.append(f"  [{role_style}]{role}:[/{role_style}]")
            if content:
                if role == "system" and content.startswith("(") and content.endswith("chars)"):
                    parts.append(f"    {content}")
                else:
                    for line in content.split("\n"):
                        parts.append(f"    {line}")
            if tool_calls:
                for tc in tool_calls:
                    args_short = tc.get('arguments', '')[:100]
                    parts.append(f"    [yellow]→ {tc['name']}[/yellow]({args_short})")

    # Tools
    tools = entry.get("tools", [])
    if tools:
        parts.append("\n[bold]Tools:[/bold]")
        for t in tools:
            parts.append(f"  [cyan]{t['name']}[/cyan]: {t.get('description', '')[:80]}")

    # Meta
    meta_parts = []
    if "temperature" in entry:
        meta_parts.append(f"temp={entry['temperature']}")
    if "estimated_input_chars" in entry:
        meta_parts.append(f"~{entry['estimated_input_chars']} chars")
    if meta_parts:
        parts.append(f"\n[dim]{' │ '.join(meta_parts)}[/dim]")

    content = "\n".join(parts)
    return Panel(content, title="[bold]📤 Request[/bold]", border_style="cyan")


def format_response(entry: dict) -> Panel:
    parts = []

    content = entry.get("content")
    if content:
        parts.append("[bold]Content:[/bold]")
        for line in content.split("\n"):
            parts.append(f"  {line}")
    else:
        parts.append("[dim](no content)[/dim]")

    tool_calls = entry.get("tool_calls", [])
    if tool_calls:
        parts.append("\n[bold]Tool calls:[/bold]")
        for tc in tool_calls:
            name = tc.get("name", "?")
            args = tc.get("arguments", "")
            parts.append(f"  [yellow]→ {name}[/yellow]")
            try:
                parsed = json.loads(args)
                formatted = json.dumps(parsed, ensure_ascii=False, indent=4)
                parts.append(formatted)
            except (json.JSONDecodeError, TypeError):
                parts.append(f"    {args[:200]}")

    # Stats
    stats = []
    if "model" in entry:
        stats.append(f"model={entry['model']}")
    if "input_tokens" in entry:
        stats.append(f"in={entry['input_tokens']}tok")
    if "output_tokens" in entry:
        stats.append(f"out={entry['output_tokens']}tok")
    if "latency_ms" in entry:
        stats.append(f"{entry['latency_ms']}ms")
    if stats:
        parts.append(f"\n[dim]{' │ '.join(stats)}[/dim]")

    text = "\n".join(parts)
    return Panel(text, title="[bold]📥 Response[/bold]", border_style="green")


def format_tool_result(entry: dict) -> Panel:
    parts = []

    name = entry.get("name", "?")
    parts.append(f"[bold yellow]→ {name}[/bold yellow]")

    args = entry.get("arguments")
    args_raw = entry.get("arguments_raw")
    if args:
        parts.append("\n[bold]Arguments:[/bold]")
        formatted = json.dumps(args, ensure_ascii=False, indent=2)
        parts.append(formatted)
    elif args_raw:
        parts.append(f"\n[bold]Arguments (raw):[/bold] {args_raw[:200]}")

    result = entry.get("result", "")
    if result:
        parts.append("\n[bold]Result:[/bold]")
        for line in result.split("\n"):
            parts.append(f"  {line}")

    text = "\n".join(parts)
    return Panel(text, title="[bold]🔧 Tool Result[/bold]", border_style="yellow")


def format_embedding(entry: dict) -> Panel:
    parts = []
    if "text_chars" in entry:
        parts.append(f"Text length: {entry['text_chars']} chars")
    if "latency_ms" in entry:
        parts.append(f"Latency: {entry['latency_ms']}ms")
    return Panel("\n".join(parts), title="[bold]📊 Embedding[/bold]", border_style="dim")


def format_entry(entry: dict) -> Panel:
    etype = entry.get("type", "unknown")
    if etype == "request":
        return format_request(entry)
    elif etype == "response":
        return format_response(entry)
    elif etype == "tool_result":
        return format_tool_result(entry)
    elif etype == "embedding":
        return format_embedding(entry)
    else:
        return Panel(json.dumps(entry, ensure_ascii=False, indent=2), title=f"[bold]{etype}[/bold]")


def show_summary(entries: list[dict]) -> None:
    table = Table(title="Log Summary", show_lines=True)
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Reasons", style="yellow")

    by_type: dict[str, list[dict]] = {}
    for e in entries:
        by_type.setdefault(e.get("type", "?"), []).append(e)

    for etype, items in sorted(by_type.items()):
        reasons: dict[str, int] = {}
        for item in items:
            r = item.get("reason", "")
            if r:
                reasons[r] = reasons.get(r, 0) + 1
        reasons_str = ", ".join(
            f"{k}: {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])
        )
        table.add_row(etype, str(len(items)), reasons_str or "-")

    # Token totals
    total_in = sum(e.get("input_tokens", 0) for e in entries)
    total_out = sum(e.get("output_tokens", 0) for e in entries)
    total_latency = [e.get("latency_ms", 0) for e in entries if "latency_ms" in e]
    avg_latency = sum(total_latency) / len(total_latency) if total_latency else 0

    console.print(table)
    console.print(
        f"\n  Tokens: [cyan]{total_in:,}[/cyan] in / [green]{total_out:,}[/green] out"
        f"  │  Avg latency: [yellow]{avg_latency:.0f}ms[/yellow]"
        f"  │  Total entries: [white]{len(entries)}[/white]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive LLM log viewer")
    parser.add_argument("--file", "-f", default=DEFAULT_FILE, help="Path to JSONL log file")
    parser.add_argument("--tail", "-t", type=int, default=0, help="Show only last N entries")
    parser.add_argument(
        "--filter", type=str, default="",
        help="Initial filter (e.g. type:response)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        console.print("[dim]Set enabled=true in config.toml [llm_log] to start logging.[/dim]")
        sys.exit(1)

    entries = load_entries(path)
    if not entries:
        console.print("[yellow]Log file is empty.[/yellow]")
        sys.exit(0)

    if args.tail > 0:
        entries = entries[-args.tail:]

    current_filter = args.filter
    filtered = filter_entries(entries, current_filter) if current_filter else entries

    pos = len(filtered) - 1  # Start at last entry
    running = True

    while running:
        console.clear()
        if not filtered:
            console.print("[yellow]No entries match the filter.[/yellow]")
            console.print("\n[dim]Commands: f clear | q[/dim]")
        else:
            entry = filtered[pos]
            header = format_header(entry, pos + 1, len(filtered), current_filter)
            console.print(header)
            console.print()
            console.print(format_entry(entry))

        console.print()
        console.print(
            "[dim]n/p: ±1 entry │ N/P: next/prev chain start │ <num>G: goto │ f: filter │ s: summary │ q: quit[/dim]",
            highlight=False,
        )

        try:
            cmd = console.input("[bold green]>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd or cmd == "n":
            if pos < len(filtered) - 1:
                pos += 1
        elif cmd == "p":
            if pos > 0:
                pos -= 1
        elif cmd == "N":
            pos = find_next_chain_start(filtered, pos)
        elif cmd == "P":
            pos = find_prev_chain_start(filtered, pos)
        elif cmd.endswith("G") and cmd[:-1].strip().isdigit():
            target = int(cmd[:-1].strip()) - 1
            if 0 <= target < len(filtered):
                pos = target
        elif cmd == "s":
            console.clear()
            show_summary(filtered)
            console.print("\n[dim]Press Enter to continue...[/dim]")
            try:
                console.input()
            except (EOFError, KeyboardInterrupt):
                pass
        elif cmd.startswith("f ") or cmd == "f":
            current_filter = cmd[2:].strip() if len(cmd) > 2 else ""
            filtered = filter_entries(entries, current_filter)
            pos = min(pos, max(len(filtered) - 1, 0))
        elif cmd == "q":
            break


if __name__ == "__main__":
    main()
