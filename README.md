# AI Character — Telegram Personality Simulation

An AI personality simulation that lives in Telegram. Inspired by [kuni](https://github.com/alex2772/kuni), built in Python.

The character has relationships with people, emotions, memory, and can sleep, reflect, and proactively reach out to contacts.

## Features

- **Relationship system** — trust, closeness, tension stats that change through diary entries
- **Sleep cycle** — diary consolidation, working memory updates, alarm system
- **RAG memory** — diary entries searchable via vector embeddings
- **Proactive behavior** — AI decides when to write to people first
- **Typo simulation** — realistic typos with auto-correction
- **Anti-repeat** — prevents repetitive responses
- **Typing indicator** — shows "typing..." while processing
- **Message batching** — debounce for rapid messages
- **Configurable** — all settings in `config.toml`

## Quick Start

```bash
# Clone
git clone https://github.com/yourname/ai_character.git
cd ai_character

# Install
python3 -m venv .venv
source .venv/bin/activate  # or .venv/bin/activate.fish for fish
pip install -e .

# Configure
cp .env.example .env
nano .env          # fill in API keys
nano config.toml   # adjust settings

# Run
python -m src.main
```

## Architecture

```
Telegram → NotificationManager → Worker → LLM (with tools) → Actions
                                         ↕
                                    Diary (RAG)
                                    Working Memory
                                    Contacts (relationships)
```

**Core loop (kuni-style):**
1. Message arrives → notification in queue
2. Worker picks notification
3. Diary lookup (related memories via embeddings)
4. LLM processes with tools (send_message, diary_write, sleep, etc.)
5. LLM calls tools in loop until `wait`/`pause`

## Tools

| Tool | Description |
|---|---|
| `send_message` | Send text to a chat (with reply support) |
| `diary_write` | Write to diary with type and stat changes |
| `ask` | Search diary/memory via RAG |
| `reflect_mood` | Update emotional state |
| `update_relationship` | Update trust/closeness/tension |
| `contacts_get` | View contacts and relationships |
| `contacts_update` | Edit contact info |
| `get_chat_context` | Read recent messages from a chat |
| `get_chats` | List all chats with activity |
| `sleep` | Start sleep process |
| `confirm_sleep` | Confirm ready to sleep |
| `set_alarm` | Set wake-up alarm |
| `wait` / `pause` | Wait for next notification |

## Configuration

### `.env` — secrets only

```
TELEGRAM_BOT_TOKEN=your_bot_token
LLM_API_KEY=your_api_key
```

### `config.toml` — all settings

```toml
[telegram]
client_type = "bot"
whitelist = [893193762]

[llm]
base_url = "https://api.deepseek.com"
model = "deepseek-v4-flash"

[character]
name = "Yuki"
owner_name = "YourName"

[behavior]
typing_wpm_min = 30
typo_correct_chance = 0.57
proactive_interval_min = 27
```

Full reference: see `config.toml` in the repository.

## Deployment (VPS + systemd)

### Server requirements

- **RAM:** 2 GB (recommended)
- **CPU:** 1 core
- **Disk:** 5 GB
- **OS:** Debian/Ubuntu

### Setup

```bash
# On server
sudo apt update && sudo apt install python3 python3-venv git
sudo useradd -r -s /bin/false bot

# Clone repo
cd /opt
git clone https://github.com/yourname/ai_character.git
cd ai_character
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env
nano .env

# Create systemd service
sudo nano /etc/systemd/system/ai-character.service
```

**Service file:**
```ini
[Unit]
Description=AI Character Telegram Bot
After=network.target

[Service]
Type=simple
User=bot
WorkingDirectory=/opt/ai_character
ExecStart=/opt/ai_character/.venv/bin/python -m src.main
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-character
sudo systemctl start ai-character
sudo journalctl -u ai-character -f
```

### Deploy updates

```bash
git push deploy master  # auto-deploy via post-receive hook
```

## Makefile

```bash
make install    # install dependencies
make run        # run locally
make deploy     # push to server
make logs       # view logs
make restart    # restart service
```

## Project Structure

```
src/
├── client/          # Telegram clients (Bot API + Userbot)
├── llm/             # LLM provider (OpenAI-compatible)
├── memory/          # Diary, working memory, contacts, RAG
├── character/       # Personality, emotions
├── core/            # Worker, tools, sleep, proactive, batcher
├── config.py        # Config system
└── main.py          # Entry point

data/
├── character_base.md    # Character prompt (editable)
├── stat_levels.json     # Relationship stat levels
├── diary/               # Diary entries
├── vectors/             # Vector store
└── history.db           # Chat history
```

## License

MIT License — see [LICENSE](LICENSE) for details.
