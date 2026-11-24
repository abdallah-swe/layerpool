import json
import os
import re
import warnings

PREFERENCE = ("dpo", "ipo", "simpo", "orpo")
FILES = (".jsonl", ".json", ".txt", ".md")
PAIRS = (("instruction", "output"), ("instruction", "response"), ("prompt", "completion"), ("prompt", "response"), ("question", "answer"),
         ("query", "response"))
PROMPTS = ("prompt", "question", "instruction")
ROLES = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "chatgpt": "assistant",
         "model": "assistant", "tool": "tool", "observation": "tool"}
KNOWN = ("'messages', 'text', 'conversations' (ShareGPT), 'instruction' and 'output' (Alpaca), 'prompt' and 'completion', or 'question' and "
         "'answer'")


class EmptyRow(ValueError):
    pass


def is_hub_name(data: str) -> bool:
    return not os.path.exists(data) and not data.lower().endswith(FILES) and re.fullmatch(r"\w[\w.-]*/\w[\w.-]*", data) is not None


def load(data, method: str = "sft", split: str = "train", subset: str | None = None) -> list:
    """Rows ready to train on: strings and lists of messages, or for a preference method dicts of chosen and rejected messages."""
    if method not in ("sft", "cpt", *PREFERENCE):
        raise ValueError(f"method must be one of sft, cpt, {', '.join(PREFERENCE)}")
    name = data if isinstance(data, str) else "the data"
    if isinstance(data, str):
        rows = _read(data, split, subset)
    elif isinstance(data, dict):
        raise ValueError(f"the data is a dict with {', '.join(map(repr, list(data)[:5]))}: pass a list of rows, or one split of a dataset")
    else:
        rows = data.to_list() if hasattr(data, "to_list") else list(data)
    convert, out, empty = _preference if method in PREFERENCE else _example, [], 0
    for i, row in enumerate(rows):
        try:
            out.append(convert(row))
        except EmptyRow:
            empty += 1
        except ValueError as e:
            raise ValueError(f"row {i + 1} of {name}: {e}") from None
    if empty and not out:
        raise ValueError(f"every row of {name} has an empty question or reply")
    if empty:
        warnings.warn(f"layerpool: left out {empty} of the {len(out) + empty} rows of {name}: their question or reply is empty", stacklevel=2)
    return out


def _read(data: str, split: str, subset: str | None) -> list:
    if is_hub_name(data):
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(f"reading {data} from Hugging Face needs the datasets package: pip install \"layerpool[hf]\"") from None
        try:
            return load_dataset(data, subset, split=split).to_list()
        except ValueError as e:
            if subset is None and "config name is missing" in str(e).lower():
                from datasets import get_dataset_config_names
                raise ValueError(f"{data} has several configurations, pick one with subset (--subset): "
                                 + ", ".join(get_dataset_config_names(data))) from None
            raise
    if not data.lower().endswith(FILES):
        raise ValueError(f"{data}: give the data as .jsonl or .json rows, as a .txt or .md file of plain text, or as the name of a dataset "
                         "on Hugging Face (user/name)")
    if split != "train" or subset:
        raise ValueError(f"{data} is a file: split and subset are for datasets on Hugging Face")
    with open(data, encoding="utf-8") as f:
        if data.lower().endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        if not data.lower().endswith(".json"):
            return [f.read()]
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{data}: a .json file must hold a list of rows")
    return rows


def _example(row):
    if isinstance(row, str):
        return row
    if isinstance(row, list):
        return _chat(row)
    if not isinstance(row, dict):
        raise ValueError(f"expected a string or an object, got {type(row).__name__}")
    if row.get("messages"):
        return _chat(row["messages"])
    if isinstance(row.get("text"), str):
        return row["text"]
    if row.get("conversations"):
        return _headed(row, _chat(row["conversations"]))
    for ask, reply in PAIRS:
        if _given(row.get(ask)) and _given(row.get(reply)):
            return _headed(row, _turns(_asked(row, ask), "user") + _turns(row[reply], "assistant"))
    if row.get("messages") == [] or row.get("conversations") == [] or \
            any(ask in row and reply in row and _blank(row[ask], row[reply]) for ask, reply in PAIRS):
        raise EmptyRow
    if "chosen" in row and "rejected" in row:
        raise ValueError("this is a preference row (chosen and rejected): train on it with method dpo, ipo, simpo or orpo")
    raise ValueError(f"it has {', '.join(map(repr, row)) or 'no fields'}; a row needs {KNOWN}")


def _preference(row) -> dict:
    if isinstance(row, dict) and "chosen" in row and "rejected" in row and _blank(row["chosen"], row["rejected"]):
        raise EmptyRow
    if not isinstance(row, dict) or not _given(row.get("chosen")) or not _given(row.get("rejected")):
        raise ValueError("a preference row needs 'prompt', 'chosen' and 'rejected'")
    asked = next((k for k in PROMPTS if _given(row.get(k))), None)
    prompt = _turns(_asked(row, asked), "user") if asked else _chat(row.get("conversations") or [])
    out = {}
    for side in ("chosen", "rejected"):
        reply = _turns(row[side], "assistant")
        whole = reply[0]["role"] != "assistant"
        if not whole and not any(m["role"] == "user" for m in prompt):
            raise ValueError(f"there is no prompt: give it as 'prompt', or as the first messages of '{side}'")
        out[side] = _headed(row, reply if whole else prompt + reply)
    return out


def _given(value) -> bool:
    return isinstance(value, (str, list, dict)) and bool(value)


def _blank(*values) -> bool:
    return any(not v for v in values) and all(v is None or isinstance(v, (str, list, dict)) for v in values)


def _asked(row: dict, key: str):
    extra = (row.get("input") or row.get("context")) if key == "instruction" else None
    return f"{row[key]}\n\n{extra}" if extra and isinstance(extra, str) and isinstance(row[key], str) else row[key]


def _headed(row: dict, chat: list[dict]) -> list[dict]:
    text = row.get("system")
    if not text or not isinstance(text, str) or chat[0]["role"] == "system":
        return chat
    return [{"role": "system", "content": text}] + chat


def _turns(value, role: str) -> list[dict]:
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    return _chat([value] if isinstance(value, dict) else value)


def _chat(turns: list) -> list[dict]:
    out = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError(f"a conversation holds messages, not {type(turn).__name__}")
        if "role" in turn:
            out.append(turn)
            continue
        if "from" not in turn or "value" not in turn:
            raise ValueError("a message needs 'role' and 'content', or 'from' and 'value'")
        role = ROLES.get(str(turn["from"]).lower())
        if role is None:
            raise ValueError(f"unknown speaker {turn['from']!r} (known: {', '.join(ROLES)})")
        out.append({"role": role, "content": turn["value"]})
    return out
