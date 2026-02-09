## Unsloth Fine-Tuning Starter

> **Privacy & Respect Note**
> Train responsibly. Your dataset probably includes other people's messages, get consent where possible, remove sensitive details, and avoid sharing private data.

## What you do

1. Set up the project.
2. Create your `.env` file.
3. Drop Discord export files into `datasets/input/`.
4. Build a training dataset.
5. Pick one model folder and follow its README.

## Setup

### Option A: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
./scripts/install_llamacpp.sh
```

### Option B: UV 
(better if you have it)

```bash
uv venv
source .venv/bin/activate
uv sync --extra train --extra bot
cp .env.example .env
./scripts/install_llamacpp.sh
```

If you prefer requirements files with UV:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
```

## Configure `.env`

Open `.env` and set at least:

```bash
DISCORD_BOT_TOKEN=your_token_here
TARGET_USER_IDS=123456789012345678
# TARGET_USER_NAMES=YourName,YourNickname
```

`clean_data.py` reads directly from `.env`, not `.env.example`.
Training wrappers (`models/*/train.sh`) read `TRAIN_*` values from `.env`.
Export wrappers (`models/*/export_gguf.sh`) read `QUANT` by default.

## Prepare your data

1. Export chats with [Discord Chat Exporter](https://github.com/Tyrrrz/DiscordChatExporter), export as json.
(Reccomendation: delete any exports under 20kb, as it's too small to really matter)
2. Put the JSON files in `datasets/input/`.
3. Run:

```bash
python clean_data.py
```

This creates:

`datasets/output/final_dataset.jsonl`

## Choose your model

Training, export, and bot commands are documented inside each model folder:

- `models/trinity/README.md`
- `models/llama31/README.md`
- `models/gemma3_12b/README.md`

Small tip: each model has a different vibe/personality in responses, so it is worth testing all three before you choose a final one. 

Each model saves outputs in its own folder:

- `<model>/output/lora`
- `<model>/output/gguf`
