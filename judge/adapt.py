"""adapt: delta edits of the rulebook, driven by the judge's errors on the labeled train.

Each round: apply the current rulebook to train, show a strong model every missed bad (next to the most
similar good) and every false bad, collect add / edit / remove proposals, check each proposal on all of
train and greedily accept those that fix errors without breaking correct items.
"""
import asyncio
import copy
import os
import re
import sys

from . import executor, llm
from .data import TOPICS, render
from .rulebook import applies, empty, new_rule, next_id, rule_text

PROPOSE_MODEL = os.environ.get("JUDGE_PROPOSE_MODEL", "claude-opus-5-5")
PROPOSE_EFFORT = os.environ.get("JUDGE_PROPOSE_EFFORT", "high")
MAX_ROUNDS = int(os.environ.get("JUDGE_ROUNDS", "3"))
MIN_FIX = int(os.environ.get("JUDGE_MIN_FIX", "1"))
MAX_BREAK = int(os.environ.get("JUDGE_MAX_BREAK", "1"))

PROPOSE_SYSTEM = """You help reconstruct a banking expert's hidden evaluation rules from labeled conversations between users and a bank's AI assistant.

The expert labels a conversation bad if the assistant violates ANY applicable rule; otherwise good.
The data are synthetic: most bad conversations are a good answer template with one injected defect, so comparing a bad example with the most similar good one usually isolates the defect.

You get the current rulebook and the judge's errors on the training set:
- MISSED BAD: the expert said bad, no rule fired. Each is shown next to the most similar good example of the same topic.
- FALSE BAD: the expert said good, but a rule fired. The rule and the checker's evidence are shown.

Propose operations on the rulebook:
- edit: rewrite an existing rule so it also catches missed bad examples, or stops firing on good ones. Prefer edit over add when the defect is a variant of an existing rule.
- add: a new rule for a defect no rule covers.
- remove: a rule that only produces false bad.

Each rule must be a general convention of the expert that a checker can verify from the CONTEXT data and the conversation alone. No concrete numbers, names or ids from the examples. One defect per rule.
Write a rule from the convention, not from the defect: first state what a correct answer does in this kind of situation (learn it from the good examples), then define the violation as any departure from that convention, not only the particular departures seen in the bad examples. Make the rule apply to every request or situation where the convention matters, as broadly as the good examples support; do not make it depend on incidental details of the bad examples, such as how the user worded the question or the specific wrong value.
In not_violation, describe the closest acceptable behaviour seen in good examples, so the checker does not over-fire. Use topics "*" only for behaviour that is clearly topic-independent.
For every operation list the item_ids it should fix. If an error looks like label noise rather than a rule, do not propose anything for it."""

OPS_SCHEMA = {
    "type": "object",
    "properties": {"ops": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["add", "edit", "remove"]},
            "rule_id": {"type": "string"},
            "name": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string", "enum": list(TOPICS) + ["*"]}},
            "violation": {"type": "string"},
            "not_violation": {"type": "string"},
            "fixes": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": ["op", "rule_id", "name", "topics", "violation", "not_violation", "fixes", "rationale"],
        "additionalProperties": False,
    }}},
    "required": ["ops"],
    "additionalProperties": False,
}


def _words(item):
    return set(re.findall(r"\w+", render(item).lower()))


def nearest_good(item, goods):
    words = _words(item)
    return max(goods, key=lambda g: len(words & _words(g)) / (len(words | _words(g)) or 1), default=None)


def wrong_ids(mat, train):
    pred = executor.predict(mat, train)
    return {it["item_id"] for it in train if pred[it["item_id"]] != it["label"]}


async def propose(book, topic, misses, false_bads, train, mat, tag):
    goods = [it for it in train if it["topic"] == topic and it["label"] == "good"]
    rules = [r for r in book["rules"] if applies(r, topic)]
    parts = [f"TOPIC OF THIS BATCH: {topic}", "", "CURRENT RULEBOOK (rules that apply to this topic):"]
    parts += [rule_text(r) + "\n" for r in rules] or ["(empty)\n"]
    for it in misses:
        parts.append(f"### MISSED BAD {it['item_id']} (expert: bad, judge: good)\n{render(it)}")
        good = nearest_good(it, goods)
        if good:
            parts.append(f"--- most similar GOOD example {good['item_id']} (expert: good) ---\n{render(good)}")
    for it in false_bads:
        why = "; ".join(f"{rid}: {mat[rid][it['item_id']]['evidence']}"
                        for rid in executor.fired(mat, it["item_id"]))
        parts.append(f"### FALSE BAD {it['item_id']} (expert: good, judge: bad)\nFired: {why}\n{render(it)}")
    parts.append("Propose rulebook operations that fix these errors.")
    try:
        out = await llm.call_json(PROPOSE_MODEL, PROPOSE_SYSTEM, "\n\n".join(parts), OPS_SCHEMA,
                                  effort=PROPOSE_EFFORT, max_tokens=64000, tag=tag)
    except llm.FATAL:
        raise
    except Exception as error:  # a failed batch only means no proposals for this topic this round
        print(f"{tag}: proposals for {topic} failed, skipped: {error!r}", file=sys.stderr)
        return []
    return out["ops"]


async def apply_ops(ops, book, mat, train, round_no, tag):
    """Check every proposal on train, then accept greedily by net fixed errors. Mutates book and mat."""
    ids = {r["id"] for r in book["rules"]}
    cands = []
    for op in ops:
        if op["op"] in ("edit", "remove") and op["rule_id"] not in ids:
            continue
        rid = op["rule_id"] if op["op"] != "add" else next_id(book)
        rule = None
        if op["op"] != "remove":
            rule = new_rule(rid, op, {"round": round_no, "op": op["op"], "fixes": op["fixes"]})
        if op["op"] == "add":
            book["history"].append({"id": rid, "round": round_no, "op": "reserved"})
        cands.append({"op": op, "rid": rid, "rule": rule})
    rows = await asyncio.gather(*(executor.row(c["rule"], train, tag) if c["rule"] else asyncio.sleep(0, {})
                                  for c in cands))
    for c, rw in zip(cands, rows):
        c["row"] = rw

    labels = {it["item_id"]: it["label"] for it in train}
    accepted, touched = [], set()
    while True:
        wrong = wrong_ids(mat, train)
        best = None
        for c in cands:
            if c in accepted or c["rid"] in touched:
                continue
            trial = dict(mat)
            if c["op"]["op"] == "remove":
                trial.pop(c["rid"])
            else:
                trial[c["rid"]] = c["row"]
            wrong2 = wrong_ids(trial, train)
            c["fixed"], c["broken"] = sorted(wrong - wrong2), sorted(wrong2 - wrong)
            gain = len(c["fixed"]) - len(c["broken"])
            # tie-break toward the more general rule: it fires on more expert-bad train items overall
            reach = sum(r["violated"] for i, r in c["row"].items() if labels[i] == "bad")
            if len(c["fixed"]) >= MIN_FIX and len(c["broken"]) <= MAX_BREAK and gain > 0:
                if best is None or (gain, reach) > best[0]:
                    best = ((gain, reach), c)
        if best is None:
            break
        c = best[1]
        accepted.append(c)
        touched.add(c["rid"])
        if c["op"]["op"] == "remove":
            mat.pop(c["rid"])
            book["rules"] = [r for r in book["rules"] if r["id"] != c["rid"]]
        else:
            mat[c["rid"]] = c["row"]
            book["rules"] = [r for r in book["rules"] if r["id"] != c["rid"]] + [c["rule"]]
    book["history"] = [h for h in book["history"] if h["op"] != "reserved"]
    for c in cands:
        book["history"].append({"id": c["rid"], "round": round_no, "op": c["op"]["op"],
                                "accepted": c in accepted, "name": c["op"]["name"],
                                "fixed": c.get("fixed", []), "broken": c.get("broken", []),
                                "proposed_fixes": c["op"]["fixes"], "rationale": c["op"]["rationale"]})
    book["rules"].sort(key=lambda r: r["id"])
    return accepted


async def adapt(train, prior=None, tag="adapt"):
    """Returns the adapted rulebook and its check matrix on train."""
    book = copy.deepcopy(prior) if prior else empty()
    mat = await executor.matrix(book["rules"], train, f"{tag}.check")
    for round_no in range(1, MAX_ROUNDS + 1):
        pred = executor.predict(mat, train)
        errors = [it for it in train if pred[it["item_id"]] != it["label"]]
        if not errors:
            break
        batches = []
        for topic in TOPICS:
            misses = [it for it in errors if it["topic"] == topic and it["label"] == "bad"]
            false_bads = [it for it in errors if it["topic"] == topic and it["label"] == "good"]
            if misses or false_bads:
                batches.append(propose(book, topic, misses, false_bads, train, mat, f"{tag}.propose"))
        ops = [op for batch in await asyncio.gather(*batches) for op in batch]
        accepted = await apply_ops(ops, book, mat, train, round_no, f"{tag}.validate")
        if not accepted:
            break
    return book, mat
