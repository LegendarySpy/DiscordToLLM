from __future__ import annotations

import os
import re
import sys
import asyncio
import time
import json
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Set

from dotenv import load_dotenv
from pathlib import Path
import discord
import torch
import unsloth  # must be imported before peft/transformers for patches
from peft import PeftModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _env_str(key: str, default: str = "") -> str:
    return (os.getenv(key, default) or "").strip()


def _env_int(key: str, default: int) -> int:
    raw = _env_str(key, str(default))
    m = re.search(r"-?\d+", raw)
    return int(m.group(0)) if m else default


def _env_bool01(key: str, default: bool = False) -> bool:
    v = _env_str(key, "")
    if not v:
        return default
    return v.lower() in {"1", "true", "yes", "y", "on"}


def _env_ids(key: str) -> Set[int]:
    raw = _env_str(key, "")
    if not raw:
        return set()
    return {int(x) for x in re.findall(r"\d+", raw)}


DISCORD_BOT_TOKEN = _env_str("DISCORD_BOT_TOKEN")
if not DISCORD_BOT_TOKEN:
    raise RuntimeError("missing DISCORD_BOT_TOKEN")

SYSTEM_PROMPT = _env_str(
    "SYSTEM_PROMPT", "You are talking to others on discord."
).strip()
NO_TOOLS_PROMPT = "Do not output JSON or tool calls. Respond in plain text only."

ALLOWED_GUILD_ID = _env_int("ALLOWED_GUILD_ID", 0) or None
ALLOWED_CHANNEL_IDS = _env_ids("ALLOWED_CHANNEL_IDS")
ALLOWED_ROLE_IDS = _env_ids("ALLOWED_ROLE_IDS")

MENTION_ONLY = _env_bool01("MENTION_ONLY", False)
REQUIRE_CUDA = _env_bool01("REQUIRE_CUDA", True)
LOG_TIMINGS = _env_bool01("LOG_TIMINGS", False)
RESET_COMMAND = _env_str("RESET_COMMAND", "!reset")
ADD_SPEAKER_TAGS = _env_bool01("ADD_SPEAKER_TAGS", False)

BASE_MODEL = _env_str("BASE_MODEL", "google/gemma-3-12b-it")
LORA_DIR = _env_str("LORA_DIR", "models/gemma3_12b/output/lora")
USE_4BIT = _env_bool01("USE_4BIT", True)
TRUST_REMOTE_CODE = _env_bool01("TRUST_REMOTE_CODE", False)
MERGE_LORA = _env_bool01("MERGE_LORA", False)
LLAMA_SERVER_URL = _env_str("LLAMA_SERVER_URL", "")
CHAT_TEMPLATE = _env_str("CHAT_TEMPLATE", "auto")

MAX_SEQ_LEN = _env_int("MAX_SEQ_LEN", 2048)
MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 512)
TEMPERATURE = float(_env_str("TEMPERATURE", "0.4") or "0.4")
TOP_P = float(_env_str("TOP_P", "0.85") or "0.85")
TOP_K = _env_int("TOP_K", 50)
REPETITION_PENALTY = float(_env_str("REPETITION_PENALTY", "1.2") or "1.2")
REPEAT_LAST_N = _env_int("REPEAT_LAST_N", 256)

IGNORE_OTHER_BOTS = True
HISTORY_PRIME_COUNT = _env_int("HISTORY_PRIME_COUNT", 20)
HISTORY_MAX_TURNS = _env_int("HISTORY_MAX_TURNS", 20)
GEN_TIMEOUT_SECONDS = _env_int("GEN_TIMEOUT_SECONDS", 120)


@dataclass
class Turn:
    role: str
    content: str


def _log_cuda_env() -> None:
    print(f"python={sys.executable}")
    print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")
    print(
        f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}"
    )
    if torch.cuda.is_available():
        print(f"device0={torch.cuda.get_device_name(0)}")
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}")


def _strip_bot_mention(bot_user: discord.ClientUser, text: str) -> str:
    return (
        text.replace(f"<@{bot_user.id}>", "").replace(f"<@!{bot_user.id}>", "").strip()
    )


def _speaker_tag(msg: discord.Message) -> str:
    name = ""
    if msg.guild is not None and isinstance(msg.author, discord.Member):
        name = msg.author.display_name or msg.author.name or ""
    else:
        name = getattr(msg.author, "display_name", "") or msg.author.name or ""
    name = re.sub(r"\s+", " ", name).strip()
    return name or "User"


def _is_command(bot_user: discord.ClientUser, text: str, command: str) -> bool:
    if not text:
        return False
    stripped = _strip_bot_mention(bot_user, text).strip()
    lower = stripped.lower()
    cmd_lower = command.lower()
    return lower == cmd_lower or lower.startswith(cmd_lower + " ")


def _is_reset_command(bot_user: discord.ClientUser, text: str) -> bool:
    if not text:
        return False
    stripped = _strip_bot_mention(bot_user, text).strip()
    return stripped.lower() == RESET_COMMAND.lower()


def _message_to_turn(
    bot_user: discord.ClientUser, msg: discord.Message
) -> Turn | None:
    if getattr(msg.author, "bot", False) and msg.author != bot_user:
        return None
    content = (msg.content or "").strip()
    if not content:
        return None
    if msg.author == bot_user:
        return Turn("assistant", content)
    text = content
    if msg.guild is not None and MENTION_ONLY:
        text = _strip_bot_mention(bot_user, text)
        if not text:
            return None
    if ADD_SPEAKER_TAGS:
        tag = _speaker_tag(msg)
        if not text.startswith(f"{tag}:"):
            text = f"{tag}: {text}"
    return Turn("user", text)


def _allowed_message(bot_user: discord.ClientUser, msg: discord.Message) -> bool:
    if msg.author == bot_user:
        return False
    if IGNORE_OTHER_BOTS and getattr(msg.author, "bot", False):
        return False
    if msg.guild is None:
        return True
    if ALLOWED_GUILD_ID is not None and msg.guild.id != ALLOWED_GUILD_ID:
        return False
    if ALLOWED_CHANNEL_IDS and msg.channel.id not in ALLOWED_CHANNEL_IDS:
        return False
    if MENTION_ONLY and bot_user not in msg.mentions:
        return False
    return True


async def _has_allowed_role(msg: discord.Message) -> bool:
    if not ALLOWED_ROLE_IDS or msg.guild is None:
        return True
    if isinstance(msg.author, discord.Member):
        if {r.id for r in msg.author.roles} & ALLOWED_ROLE_IDS:
            return True
    try:
        m = await msg.guild.fetch_member(msg.author.id)
        return bool({r.id for r in m.roles} & ALLOWED_ROLE_IDS)
    except Exception:
        return False


def _build_messages(system_prompt: str, turns: Deque[Turn]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "system", "content": NO_TOOLS_PROMPT})
    else:
        msgs.append({"role": "system", "content": NO_TOOLS_PROMPT})
    for t in turns:
        msgs.append({"role": t.role, "content": t.content})
    return msgs


def _load_model():
    _log_cuda_env()
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Set REQUIRE_CUDA=0 to allow CPU.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=USE_4BIT,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    if CHAT_TEMPLATE != "auto":
        tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    elif getattr(tokenizer, "chat_template", None) is None:
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    if MERGE_LORA:
        if USE_4BIT:
            raise RuntimeError(
                "MERGE_LORA=1 requires USE_4BIT=0 (merge into full-precision weights)."
            )
        model = model.merge_and_unload()
    FastLanguageModel.for_inference(model)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    try:
        p = next(model.parameters())
        print(f"model device={p.device} dtype={p.dtype}")
    except Exception as e:
        print(f"model device check failed: {type(e).__name__}: {e}")
    return model, tokenizer


def _generate(model, tokenizer, messages: List[Dict[str, str]]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    device = next(model.parameters()).device
    if REQUIRE_CUDA and device.type != "cuda":
        raise RuntimeError(
            f"Model is on {device}, expected cuda. Set REQUIRE_CUDA=0 to allow CPU."
        )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    if LOG_TIMINGS:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        in_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            use_cache=True,
        )
    if LOG_TIMINGS:
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out_len = out.shape[-1]
        gen_tokens = max(out_len - in_len, 0)
        tps = (gen_tokens / dt) if dt > 0 else 0.0
        print(
            f"gen: in={in_len} out={out_len} new={gen_tokens} time={dt:.2f}s tok/s={tps:.2f}"
        )
    input_len = inputs["input_ids"].shape[-1]
    new_tokens = out[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


async def _model_respond(model, tokenizer, messages: List[Dict[str, str]]) -> str:
    if LLAMA_SERVER_URL:
        return await asyncio.wait_for(
            asyncio.to_thread(_generate_remote, messages),
            timeout=GEN_TIMEOUT_SECONDS + 30,
        )
    return await asyncio.wait_for(
        asyncio.to_thread(_generate, model, tokenizer, messages),
        timeout=GEN_TIMEOUT_SECONDS,
    )


def _generate_remote(messages: List[Dict[str, str]]) -> str:
    url = LLAMA_SERVER_URL.rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url = url + "/v1/chat/completions"
    payload = {
        "model": "local",
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repeat_penalty": REPETITION_PENALTY,
        "repeat_last_n": REPEAT_LAST_N,
        "max_tokens": MAX_NEW_TOKENS,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=GEN_TIMEOUT_SECONDS + 30) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    try:
        return out["choices"][0]["message"]["content"].strip()
    except Exception:
        return str(out)


class SimpleBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        super().__init__(intents=intents)

        self.model = None
        self.tokenizer = None
        self.history: Dict[int, Deque[Turn]] = defaultdict(
            lambda: deque(maxlen=HISTORY_MAX_TURNS)
        )
        self.locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.history_loaded: Set[int] = set()

    async def _prime_history(
        self,
        channel: discord.abc.Messageable,
        bot_user: discord.ClientUser,
        current_message_id: int | None = None,
    ) -> None:
        channel_id = getattr(channel, "id", None)
        if channel_id is None or channel_id in self.history_loaded:
            return
        self.history_loaded.add(channel_id)
        if HISTORY_PRIME_COUNT <= 0:
            return
        try:
            async for msg in channel.history(
                limit=HISTORY_PRIME_COUNT, oldest_first=True
            ):
                if current_message_id is not None and msg.id == current_message_id:
                    continue
                turn = _message_to_turn(bot_user, msg)
                if turn is not None:
                    self.history[channel_id].append(turn)
        except Exception as e:
            print(
                f"history prime failed for channel={channel_id}: {type(e).__name__}: {e}"
            )

    async def setup_hook(self) -> None:
        if LLAMA_SERVER_URL:
            self.model, self.tokenizer = None, None
            print(f"using llama-server at {LLAMA_SERVER_URL}")
        else:
            self.model, self.tokenizer = await asyncio.to_thread(_load_model)

    async def on_ready(self) -> None:
        print(f"✅ logged in as {self.user} (id={self.user.id if self.user else None})")

    async def on_message(self, message: discord.Message) -> None:
        if self.user is None:
            return
        if not LLAMA_SERVER_URL and (self.model is None or self.tokenizer is None):
            return
        if not _allowed_message(self.user, message):
            return
        if not await _has_allowed_role(message):
            return

        text = (message.content or "").strip()
        if not text:
            return

        if _is_reset_command(self.user, text):
            self.history[message.channel.id].clear()
            await message.channel.send(
                "context reset.", allowed_mentions=discord.AllowedMentions.none()
            )
            return

        if message.guild is not None and MENTION_ONLY:
            text = _strip_bot_mention(self.user, text)
            if not text:
                return
        if ADD_SPEAKER_TAGS:
            tag = _speaker_tag(message)
            if not text.startswith(f"{tag}:"):
                text = f"{tag}: {text}"

        channel_id = message.channel.id
        async with self.locks[channel_id]:
            await self._prime_history(
                message.channel, self.user, current_message_id=message.id
            )
            self.history[channel_id].append(Turn("user", text))

            messages = _build_messages(SYSTEM_PROMPT, self.history[channel_id])

            try:
                async with message.channel.typing():
                    reply = await _model_respond(self.model, self.tokenizer, messages)
            except asyncio.TimeoutError:
                reply = "timed out. try a shorter message."
            except Exception as e:
                reply = f"{type(e).__name__}: {e}"

            reply = _strip_bot_mention(self.user, reply)
            if not reply:
                reply = "..."

            self.history[channel_id].append(Turn("assistant", reply))
            await message.channel.send(
                reply, allowed_mentions=discord.AllowedMentions.none()
            )


def main() -> None:
    SimpleBot().run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
