# Trinity Nano

## 1) Train

```bash
./models/trinity/train.sh
```

Optional examples:

```bash
TRAIN_BATCH_SIZE=4 TRAIN_EPOCHS=3 ./models/trinity/train.sh
TRAIN_WANDB_PROJECT=trinity-finetune TRAIN_RUN_NAME=run-01 ./models/trinity/train.sh
```

## 2) Export GGUF

```bash
./models/trinity/export_gguf.sh
```

Optional quantization example:

```bash
QUANT=q4_k_m ./models/trinity/export_gguf.sh
```

## 3) Run Discord bot

```bash
./models/trinity/run_discord_bot.sh
```
