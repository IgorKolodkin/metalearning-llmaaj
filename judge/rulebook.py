"""The rulebook: the single source of truth for the checker prompts, predictions and the jury explanation."""
import hashlib
import json
from pathlib import Path

from .data import TOPICS


def empty():
    return {"rules": [], "history": []}


def load(path):
    return json.loads(Path(path).read_text())


def save(book, path):
    Path(path).write_text(json.dumps(book, ensure_ascii=False, indent=1))


def applies(rule, topic):
    return "*" in rule["topics"] or topic in rule["topics"]


def next_id(book):
    used = [int(r["id"][1:]) for r in book["rules"]] + [int(h["id"][1:]) for h in book["history"] if "id" in h]
    return f"R{max(used, default=0) + 1:02d}"


def rule_text(rule):
    """Exactly what the checker sees for a rule. The explanation quotes the same text."""
    topics = "all topics" if "*" in rule["topics"] else ", ".join(rule["topics"])
    return (f"RULE {rule['id']}: {rule['name']}\n"
            f"Applies to: {topics}\n"
            f"Violation (the conversation is bad): {rule['violation']}\n"
            f"Not a violation: {rule['not_violation']}")


def digest(book):
    return hashlib.sha256("\n\n".join(rule_text(r) for r in book["rules"]).encode()).hexdigest()[:12]


def new_rule(rid, spec, origin):
    topics = [t for t in spec["topics"] if t in TOPICS or t == "*"] or ["*"]
    return {"id": rid, "name": spec["name"], "topics": topics, "violation": spec["violation"],
            "not_violation": spec["not_violation"], "origin": origin}
