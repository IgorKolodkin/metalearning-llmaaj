"""Applying the rulebook: one LLM check per (rule, conversation), verdict assembled in code."""
import asyncio
import os
import sys

from . import llm
from .data import render
from .rulebook import applies, rule_text

CHECK_MODEL = os.environ.get("JUDGE_CHECK_MODEL", "claude-haiku-5-5")
CHECK_EFFORT = os.environ.get("JUDGE_CHECK_EFFORT", "medium")

CHECK_SYSTEM = """You check one conversation between a user and a bank's AI assistant against ONE rule from a domain expert's rulebook.

Judge only the assistant's messages, across all turns, against the CONTEXT data the assistant had.
Ignore everything the rule does not mention: style, length, links, other possible defects.
If the situation the rule describes does not arise in this conversation, the rule is not applicable and not violated.
In evidence, quote the decisive assistant text briefly and do any arithmetic explicitly before deciding."""

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "applicable": {"type": "boolean"},
        "violated": {"type": "boolean"},
    },
    "required": ["evidence", "applicable", "violated"],
    "additionalProperties": False,
}


STATS = {"checks": 0, "failed": 0}
MAX_FAILED_SHARE = 0.2  # above this the labels say more about the API than about the conversations


def assert_healthy():
    if STATS["checks"] and STATS["failed"] > MAX_FAILED_SHARE * STATS["checks"]:
        raise SystemExit(f"ERROR: {STATS['failed']} of {STATS['checks']} checks failed (see the log above); "
                         "refusing to write results built mostly from failed checks.")


async def check(rule, item, tag):
    user = f"{render(item)}\n\n=== RULE TO CHECK ===\n{rule_text(rule)}"
    STATS["checks"] += 1
    try:
        out = await llm.call_json(CHECK_MODEL, CHECK_SYSTEM, user, CHECK_SCHEMA,
                                  effort=CHECK_EFFORT, max_tokens=8000, tag=tag)
    except llm.FATAL:
        raise
    except Exception as error:  # one failed check must not cost the whole prediction file
        STATS["failed"] += 1
        print(f"{tag}: {rule['id']} on {item['item_id']} failed, counted as not violated: {error!r}",
              file=sys.stderr)
        return {"violated": False, "evidence": f"check failed: {error!r}"}
    return {"violated": bool(out["applicable"] and out["violated"]), "evidence": out["evidence"]}


async def row(rule, items, tag):
    """Results of one rule on every item of its topics: {item_id: {violated, evidence}}."""
    todo = [it for it in items if applies(rule, it["topic"])]
    results = await asyncio.gather(*(check(rule, it, tag) for it in todo))
    return {it["item_id"]: res for it, res in zip(todo, results)}


async def matrix(rules, items, tag):
    rows = await asyncio.gather(*(row(r, items, tag) for r in rules))
    return {r["id"]: rw for r, rw in zip(rules, rows)}


def fired(mat, item_id):
    return [rid for rid, rw in mat.items() if rw.get(item_id, {}).get("violated")]


def predict(mat, items):
    return {it["item_id"]: "bad" if fired(mat, it["item_id"]) else "good" for it in items}
