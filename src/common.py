"""Dataset and hard-label prediction validation, using the Python standard library."""
import csv
import json
from pathlib import Path

LABELS = ("good", "bad")
TOPICS = ("credit", "financial_position", "savings", "spending")
BASE_FIELDS = {"item_id", "topic", "context", "conversation"}
OPTIONAL_FIELDS = {"label", "input_es", "input_en"}


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields or len(fields) != len(set(fields)):
                raise ValueError(f"{path}: missing or duplicate CSV headers")
            result = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in result):
            raise ValueError(f"{path}: inconsistent CSV column count")
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            result = [json.loads(line) for line in stream if line.strip()]
        if any(not isinstance(row, dict) for row in result):
            raise ValueError(f"{path}: JSONL rows must be objects")
    else:
        raise ValueError("Expected a .csv or .jsonl file")
    if not result:
        raise ValueError(f"{path}: empty input")
    return result


def indexed(rows):
    result = {}
    for row in rows:
        iid = row.get("item_id")
        if not isinstance(iid, str) or not iid.strip():
            raise ValueError("item_id must be a nonempty string")
        if iid in result:
            raise ValueError(f"Duplicate item_id: {iid}")
        result[iid] = row
    if not result:
        raise ValueError("Empty input")
    return result


def check_content(context, conversation):
    if not isinstance(context, dict):
        raise ValueError("context must be a JSON object")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("conversation must be a nonempty JSON array")
    for message in conversation:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Each message must contain exactly role and content")
        if message["role"] not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Invalid message role")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Message content must be a nonempty string")


def load_dataset(path, require_labels=False):
    rows = read_records(path)
    indexed(rows)
    result = []
    for source in rows:
        if not BASE_FIELDS <= set(source) or set(source) - BASE_FIELDS - OPTIONAL_FIELDS:
            raise ValueError("Unexpected dataset columns or missing required columns")
        row = dict(source)
        if row["topic"] not in TOPICS:
            raise ValueError(f"Unknown topic: {row['topic']}")
        for field in ("context", "conversation"):
            if isinstance(row[field], str):
                row[field] = json.loads(row[field])
        check_content(row["context"], row["conversation"])
        if require_labels and "label" not in row:
            raise ValueError("Gold labels are required")
        if "label" in row and row["label"] not in LABELS:
            raise ValueError("Labels must be exactly good or bad")
        for language in ("es", "en"):
            field = f"input_{language}"
            if field in row:
                value = row[field]
                translated = json.loads(value) if isinstance(value, str) else value
                if not isinstance(translated, dict) or set(translated) != {"context", "conversation"}:
                    raise ValueError(f"Invalid {field} structure")
                check_content(translated["context"], translated["conversation"])
        result.append(row)
    return result


def load_labels(path):
    rows = read_records(path)
    indexed(rows)
    for row in rows:
        if set(row) != {"item_id", "label"} or row["label"] not in LABELS:
            raise ValueError("Labels/predictions require exactly item_id,label; label must be good or bad")
    return rows


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
