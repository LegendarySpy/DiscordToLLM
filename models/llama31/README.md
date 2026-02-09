# Llama 3.1 8B

## 1) Train

```bash
./models/llama31/train.sh
```

Optional examples:

```bash
TRAIN_BATCH_SIZE=4 TRAIN_EPOCHS=3 ./models/llama31/train.sh
TRAIN_CHAT_TEMPLATE=llama-3 ./models/llama31/train.sh
```

## 2) Export GGUF

```bash
./models/llama31/export_gguf.sh
```

Optional quantization example:

```bash
QUANT=q4_k_m ./models/llama31/export_gguf.sh
```

## 3) Run Discord bot

```bash
./models/llama31/run_discord_bot.sh
```

This is the full flow for Llama 3.1: train -> export -> run bot.
