import json
import re
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

_REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env")


def _env_str(key: str, default: str = "") -> str:
    return (os.getenv(key, default) or "").strip()


def _csv_set(raw: str) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in re.split(r"[,\n;]", raw) if part.strip()}


def _repo_path(raw: str, default: Path) -> str:
    if not raw:
        return str(default)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return str(p)


DATASET_FOLDER = _repo_path(
    _env_str("CLEAN_DATASET_FOLDER"),
    _REPO_ROOT / "datasets" / "input",
)
OUTPUT_FILE = _repo_path(
    _env_str("CLEAN_OUTPUT_FILE"),
    _REPO_ROOT / "datasets" / "output" / "final_dataset.jsonl",
)

# Prefer ID match for accuracy across servers/nick changes.
# Set TARGET_USER_IDS to an empty set to disable ID matching.
TARGET_USER_IDS = _csv_set(_env_str("TARGET_USER_IDS"))
TARGET_USER_NAMES = {name.lower() for name in _csv_set(_env_str("TARGET_USER_NAMES"))}

# System prompt is intentionally omitted from dataset output.
SYSTEM_PROMPT = ""

SESSION_TIMEOUT_SECONDS = 3600        # 1 hour gap => new session
MAX_MESSAGES_PER_WINDOW = 8          # keep context manageable
WINDOW_STRIDE = MAX_MESSAGES_PER_WINDOW  # set to MAX_MESSAGES_PER_WINDOW for no overlap
MERGE_WITHIN_SECONDS = 180           # burst merge window (same speaker)

# Quality gates (tune for quality > quantity)
MIN_LAST_ASSISTANT_CHARS = 12
MIN_ASSISTANT_TOTAL_CHARS = 20
MIN_TOTAL_CHARS = 60
MAX_SHORT_ASSISTANT_RATIO = 0.40
MIN_MESSAGE_CHARS = 3
MIN_ALNUM_CHARS = 3
SHORT_ASSISTANT_CHARS = 12
DROP_PII = True
ADD_SPEAKER_TAGS = False
DROP_MENTIONS = True
DROP_COMMANDISH = True
MAX_MESSAGE_CHARS = 300
MIN_ALPHA_RATIO = 0.35
MIN_ALPHA_CHECK_CHARS = 12
DROP_GIBBERISH = True
GIBBERISH_MIN_WORDS = 2
GIBBERISH_MIN_WORD_LEN = 4
GIBBERISH_BAD_WORD_RATIO = 0.6
GIBBERISH_MIN_CONSONANT_RUN = 6
DROP_REPETITIVE_USER = True
DROP_MOD_TONE = True
REPETITIVE_MIN_TOKENS = 6
REPETITIVE_MAX_UNIQUE = 2
REPETITIVE_DOMINANCE = 0.6
DROP_REPETITIVE_ASSISTANT = True

BOT_ROT_WORDS = {
    "ok", "ye", "yes", "no", "what", "lol", "[link]", "1 sec",
    "join", "kk", "yup", "yea", "idk", "wait", "si", "hi", "hey"
}

# only skip obvious command prefixes; don't delete normal punctuation like '?' or '-'
COMMAND_PREFIXES = ("!", "/", ".")

# ==========================================
# 🧹 CLEANING UTILS
# ==========================================

def parse_ts(ts_iso: str) -> float:
    return datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp()

def get_author_name(author_obj: Dict[str, Any]) -> str:
    return author_obj.get("nickname") or author_obj.get("name") or "User"

def resolve_mentions(content: str, mentions: List[Dict[str, Any]]) -> str:
    """Converts <@123> / <@!123> to @Name using the export mentions array."""
    if not mentions:
        return content
    out = content
    for user in mentions:
        user_id = user.get("id")
        if not user_id:
            continue
        name = user.get("nickname") or user.get("globalName") or user.get("name") or "User"
        out = re.sub(fr"<@!?{re.escape(str(user_id))}>", f"@{name}", out)
    return out

def _dedupe_repeated_lines(text: str, keep_repeats: int = 2) -> str:
    """
    Compress repeated identical lines (ping spam / copy spam).
    Keeps at most `keep_repeats` consecutive duplicates.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    out: List[str] = []
    last = None
    streak = 0
    for ln in lines:
        if ln == last:
            streak += 1
            if streak >= keep_repeats:
                continue
        else:
            last = ln
            streak = 0
        out.append(ln)
    return "\n".join(out)

def clean_content(text: str) -> str:
    """
    Keep style. Clean junk.
    - links -> removed
    - emoji IDs -> :emoji:
    - tame only INSANE repeats (keep 'noooo' energy)
    - compress ping spam lines
    """
    if not text:
        return ""

    # turn literal "\\n" into actual newline for consistent formatting
    text = text.replace("\\n", "\n")

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<a?:(.+?):\d+>", r":\1:", text)
    text = text.replace("[link]", " ")
    if DROP_MENTIONS:
        text = _MENTION_PATTERN.sub("", text)

    # insane repeats: 20+ -> 10
    text = re.sub(r"(.)\1{20,}", lambda m: m.group(1) * 10, text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    text = _dedupe_repeated_lines(text, keep_repeats=2)
    # remove newlines and collapse whitespace
    text = " ".join(text.split())
    return text.strip()

_ONLY_EMOJI_OR_PUNCT = re.compile(r"^[\W_\d\s]+$")
_MENTION_PATTERN = re.compile(r"(?<!\\w)@[^\\s]+")
_PII_PATTERNS = [
    re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),  # email
    re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),  # phone
    re.compile(r"\b-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+\b"),  # lat/long
]
_COMMANDISH_PATTERNS = [
    re.compile(r"\b(schedule|remind|reminder|set a reminder|set a timer|timer)\b", re.I),
    re.compile(r"\b(ping me|ping @|can you ping|schedule a message)\b", re.I),
]
_AUTHORITY_WORDS = re.compile(r"\b(mods?|admin|helper|manager|staff)\b", re.I)
_ENFORCEMENT_WORDS = re.compile(r"\b(ban(?:ned|ning)?|unban|mute(?:d)?|kick(?:ed)?)\b", re.I)
_SERVER_WORDS = re.compile(r"\b(server|discord)\b", re.I)

def _has_pii(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _PII_PATTERNS)

def _is_low_signal(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) < MIN_MESSAGE_CHARS:
        return True
    if t.lower() in BOT_ROT_WORDS:
        return True
    if t == "[link]":
        return True
    if _ONLY_EMOJI_OR_PUNCT.match(t):
        return True
    core = re.sub(r"\[link\]", "", t, flags=re.I)
    core = re.sub(r"[^A-Za-z0-9]+", "", core)
    if len(core) < MIN_ALNUM_CHARS:
        return True
    return False

def _is_commandish(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _COMMANDISH_PATTERNS)

def _is_symbol_heavy(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < MIN_ALPHA_CHECK_CHARS:
        return False
    total = sum(1 for ch in t if not ch.isspace())
    if total == 0:
        return True
    letters = sum(1 for ch in t if ch.isalpha())
    return (letters / total) < MIN_ALPHA_RATIO

def _is_listy(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.count(",") >= 8:
        return True
    if t.count("\n") >= 8:
        return True
    return False

def _is_gibberish(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # long consonant runs are a strong signal
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{%d,}" % GIBBERISH_MIN_CONSONANT_RUN, t):
        return True
    words = re.findall(r"[a-z]{%d,}" % GIBBERISH_MIN_WORD_LEN, t)
    if len(words) < GIBBERISH_MIN_WORDS:
        return False
    bad = 0
    for w in words:
        vowels = sum(1 for ch in w if ch in "aeiou")
        ratio = vowels / max(1, len(w))
        if ratio < 0.2 or ratio > 0.85:
            bad += 1
    return (bad / len(words)) >= GIBBERISH_BAD_WORD_RATIO

def _is_repetitive_text(text: str) -> bool:
    t = (text or "").strip().lower()
    tokens = re.findall(r"[a-z0-9']+", t)
    if len(tokens) < REPETITIVE_MIN_TOKENS:
        return False
    unique = set(tokens)
    if len(unique) <= REPETITIVE_MAX_UNIQUE:
        return True
    # dominant token check
    counts = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    if max(counts.values()) / len(tokens) >= REPETITIVE_DOMINANCE:
        return True
    # long run of same token
    run = 1
    best = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
    return best >= 5

def _has_repeated_assistant(chunk: List[Dict[str, str]]) -> bool:
    seen = set()
    for m in chunk:
        if m["role"] != "assistant":
            continue
        key = re.sub(r"\s+", " ", (m["content"] or "").strip().lower())
        if not key:
            continue
        if key in seen:
            return True
        seen.add(key)
    return False

def _has_mod_tone(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if not _ENFORCEMENT_WORDS.search(t):
        return False
    return bool(_AUTHORITY_WORDS.search(t) or _SERVER_WORDS.search(t) or "from the server" in t.lower())

def is_system_message(content: str) -> bool:
    c = (content or "").lower()
    noise = [
        "started a call", "missed a call", "pinned a message",
        "joined the call", "joined the server", "left the group",
        "changed the channel", "unmuted", "undeafened"
    ]
    return any(n in c for n in noise)

def squash_consecutive_roles(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge consecutive turns with the same role inside a window."""
    out: List[Dict[str, str]] = []
    for m in msgs:
        if not out:
            out.append({"role": m["role"], "content": m["content"]})
            continue
        if out[-1]["role"] == m["role"]:
            out[-1]["content"] += "\n" + m["content"]
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out

# ==========================================
# 🏗️ DATASET ENGINE
# ==========================================

def process_file(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"error loading {file_path}: {e}")
        return []

    raw = data.get("messages", [])
    if not raw:
        return []

    raw.sort(key=lambda x: x.get("timestamp", ""))

    source_file = os.path.basename(file_path)

    # Step 0: normalize + clean
    processed: List[Dict[str, Any]] = []
    for msg in raw:
        content = msg.get("content", "")
        if not content or is_system_message(content):
            continue

        author_obj = msg.get("author") or {}
        if author_obj.get("isBot", False):
            continue

        if content.startswith(COMMAND_PREFIXES):
            continue

        content = resolve_mentions(content, msg.get("mentions", []))
        content = clean_content(content)
        if not content:
            continue

        author_id = str(author_obj.get("id") or "")
        author = get_author_name(author_obj)
        is_target = (author_id in TARGET_USER_IDS) or (author.lower() in TARGET_USER_NAMES)

        role = "assistant" if is_target else "user"

        # optionally tag non-targets with speaker name
        if ADD_SPEAKER_TAGS and not is_target:
            content = f"{author}: {content}"

        ts_iso = msg.get("timestamp")
        if not ts_iso:
            continue

        processed.append({
            "role": role,
            "speaker": author,  # used for merging
            "content": content,
            "ts": parse_ts(ts_iso),
        })

    if not processed:
        return []

    # Step 1: burst merge (same speaker + role within window)
    merged: List[Dict[str, Any]] = []
    temp = processed[0].copy()

    for i in range(1, len(processed)):
        curr = processed[i]
        same_person = (curr["role"] == temp["role"]) and (curr["speaker"] == temp["speaker"])
        close_in_time = (curr["ts"] - temp["ts"]) < MERGE_WITHIN_SECONDS

        if same_person and close_in_time:
            temp["content"] += "\n" + curr["content"]
            temp["ts"] = curr["ts"]
        else:
            temp["content"] = _dedupe_repeated_lines(temp["content"], keep_repeats=2)
            merged.append(temp)
            temp = curr.copy()
    temp["content"] = _dedupe_repeated_lines(temp["content"], keep_repeats=2)
    merged.append(temp)

    # Step 2: session split + windowing
    final_blocks: List[Dict[str, Any]] = []
    session: List[Dict[str, str]] = []
    session_id = 0

    def flush(sess: List[Dict[str, str]], sid: int) -> None:
        final_blocks.extend(create_windows(sess, source_file, sid))

    for i, m in enumerate(merged):
        if i > 0 and (m["ts"] - merged[i - 1]["ts"]) > SESSION_TIMEOUT_SECONDS:
            flush(session, session_id)
            session = []
            session_id += 1

        session.append({"role": m["role"], "content": m["content"]})

    flush(session, session_id)
    return final_blocks

def create_windows(session: List[Dict[str, str]], source_file: str, session_id: int) -> List[Dict[str, Any]]:
    if len(session) < 2:
        return []

    out: List[Dict[str, Any]] = []
    # window stride (set to MAX_MESSAGES_PER_WINDOW for no overlap)
    for i in range(0, len(session), WINDOW_STRIDE):
        chunk = session[i:i + MAX_MESSAGES_PER_WINDOW]

        # must start with user and end with assistant
        while chunk and chunk[0]["role"] == "assistant":
            chunk.pop(0)
        while chunk and chunk[-1]["role"] == "user":
            chunk.pop(-1)

        if len(chunk) < 2 or chunk[-1]["role"] != "assistant":
            continue

        # merge same-role runs inside the window (prevents assistant->assistant tails)
        chunk = squash_consecutive_roles(chunk)

        # re-check shape after squash
        while chunk and chunk[0]["role"] == "assistant":
            chunk.pop(0)
        while chunk and chunk[-1]["role"] == "user":
            chunk.pop(-1)
        if len(chunk) < 2 or chunk[-1]["role"] != "assistant":
            continue

        # quality filters
        last_raw = (chunk[-1]["content"] or "").strip()
        if len(last_raw) < MIN_LAST_ASSISTANT_CHARS or _is_low_signal(last_raw):
            continue

        if DROP_PII and any(_has_pii(m["content"]) for m in chunk):
            continue

        if any(len(m["content"]) > MAX_MESSAGE_CHARS for m in chunk):
            continue

        if DROP_COMMANDISH and any(m["role"] == "user" and _is_commandish(m["content"]) for m in chunk):
            continue

        if any(_is_symbol_heavy(m["content"]) for m in chunk):
            continue

        if DROP_GIBBERISH and any(m["role"] == "assistant" and _is_gibberish(m["content"]) for m in chunk):
            continue

        if DROP_REPETITIVE_USER and any(m["role"] == "user" and _is_repetitive_text(m["content"]) for m in chunk):
            continue

        if DROP_REPETITIVE_ASSISTANT and any(m["role"] == "assistant" and _is_repetitive_text(m["content"]) for m in chunk):
            continue

        if any(m["role"] == "assistant" and _is_listy(m["content"]) for m in chunk):
            continue

        if DROP_MOD_TONE and any(_has_mod_tone(m["content"]) for m in chunk):
            continue

        if any(m["role"] == "user" and _is_low_signal(m["content"]) for m in chunk):
            continue

        total_chars = sum(len(m["content"]) for m in chunk if m["role"] != "system")
        if total_chars < MIN_TOTAL_CHARS:
            continue

        assistant_msgs = [m for m in chunk if m["role"] == "assistant"]
        if not assistant_msgs:
            continue
        assistant_total = sum(len(m["content"]) for m in assistant_msgs)
        if assistant_total < MIN_ASSISTANT_TOTAL_CHARS:
            continue
        short_assistant = [len(m["content"]) < SHORT_ASSISTANT_CHARS for m in assistant_msgs]
        if short_assistant and (sum(short_assistant) / len(short_assistant)) > MAX_SHORT_ASSISTANT_RATIO:
            continue
        if _has_repeated_assistant(chunk):
            continue

        if any(_is_low_signal(m["content"]) for m in assistant_msgs):
            continue

        out.append({
            "messages": chunk
        })

    return out

# ==========================================
# 🚀 EXECUTION
# ==========================================

def main():
    if not TARGET_USER_IDS and not TARGET_USER_NAMES:
        raise ValueError(
            "No target users configured. Set TARGET_USER_IDS and/or TARGET_USER_NAMES in .env"
        )

    all_rows: List[Dict[str, Any]] = []
    files = glob.glob(os.path.join(DATASET_FOLDER, "*.json"))

    for f in files:
        print(f"refining: {os.path.basename(f)}")
        all_rows.extend(process_file(f))

    # stable dedupe
    unique = {json.dumps(row, sort_keys=True, ensure_ascii=False) for row in all_rows}
    final = [json.loads(s) for s in unique]

    print(f"\n✅ {len(all_rows)} blocks -> {len(final)} unique samples")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in final:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"💾 saved -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
