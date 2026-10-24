"""Loading examples (CSV or JSONL in the organizers' format) and rendering them for prompts."""
import csv
import json
from pathlib import Path

TOPICS = ("credit", "financial_position", "savings", "spending")


def load(path):
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]
    items = []
    for row in rows:
        item = {k: row[k] for k in ("item_id", "topic", "context", "conversation") if k in row}
        for field in ("context", "conversation"):
            if isinstance(item[field], str):
                item[field] = json.loads(item[field])
        if row.get("label"):
            item["label"] = row["label"]
        items.append(item)
    return items


def render(item):
    lines = [f"TOPIC: {item['topic']}", "", "CONTEXT (data the assistant had access to):",
             json.dumps(item["context"], ensure_ascii=False, indent=1), "", "CONVERSATION:"]
    for i, message in enumerate(item["conversation"], 1):
        lines.append(f"[{i}] {message['role'].upper()}: {message['content']}")
    return "\n".join(lines)


def first_user_text(item, limit=160):
    text = next((m["content"] for m in item["conversation"] if m["role"] == "user"), "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."
