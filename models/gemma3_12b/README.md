# Gemma 3 12B

## 1) Train

```bash
./models/gemma3_12b/train.sh
```

Optional examples:

```bash
TRAIN_BATCH_SIZE=4 TRAIN_EPOCHS=3 ./models/gemma3_12b/train.sh
TRAIN_CHAT_TEMPLATE=gemma-3 ./models/gemma3_12b/train.sh
```

## 2) Export GGUF

```bash
./models/gemma3_12b/export_gguf.sh
```

Optional quantization example if you want to overwrite .env:

```bash
QUANT=q4_k_m ./models/gemma3_12b/export_gguf.sh
```

## 3) Run Discord bot

```bash
./models/gemma3_12b/run_discord_bot.sh
```