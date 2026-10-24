"""CLI: `python -m judge adapt|eval`, the two stages the organizers run."""
import argparse
import asyncio
import csv
import json
import sys
import traceback
from pathlib import Path

from . import adapt as adapt_mod, executor, explain, llm, rulebook
from .data import load


def write_explanation(book, mat, train, out):
    try:
        text = explain.render(book, mat, train)
    except Exception:  # the rulebook is already saved: a broken explanation must not fail adapt
        traceback.print_exc()
        text = f"# Explanation unavailable\n\nRendering failed:\n\n```\n{traceback.format_exc()}```\n"
    (out / "explanation.md").write_text(text)


async def run_adapt(args):
    train = load(args.train)
    prior = rulebook.load(args.prior) if args.prior else None
    await llm.preflight([executor.CHECK_MODEL, adapt_mod.PROPOSE_MODEL])
    book, mat = await adapt_mod.adapt(train, prior)
    executor.assert_healthy()
    out = Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    rulebook.save(book, out / "rulebook.json")
    write_explanation(book, mat, train, out)
    print(json.dumps({"rules": len(book["rules"]), "spend": llm.spend()}))


async def run_explain(args):
    """Re-render explanation.md for saved artifacts; the train checks come from the cache when present."""
    train = load(args.train)
    book = rulebook.load(Path(args.artifacts) / "rulebook.json")
    mat = await executor.matrix(book["rules"], train, "explain.check")
    write_explanation(book, mat, train, Path(args.artifacts))
    print(json.dumps({"rules": len(book["rules"]), "spend": llm.spend()}), file=sys.stderr)


async def run_eval(args):
    book = rulebook.load(Path(args.artifacts) / "rulebook.json")
    test = load(args.test)
    await llm.preflight([executor.CHECK_MODEL])
    pred = executor.predict(await executor.matrix(book["rules"], test, "eval.check"), test)
    executor.assert_healthy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "label"])
        writer.writerows((it["item_id"], pred[it["item_id"]]) for it in test)
    print(json.dumps({"n": len(test), "spend": llm.spend()}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("adapt")
    a.add_argument("--train", default="/input/train.jsonl")
    a.add_argument("--artifacts", default="/artifacts")
    a.add_argument("--prior", help="start from this rulebook.json")
    e = sub.add_parser("eval")
    e.add_argument("--test", default="/input/test.jsonl")
    e.add_argument("--artifacts", default="/artifacts")
    e.add_argument("--out", default="/output/predictions.csv")
    x = sub.add_parser("explain", help="re-render explanation.md for saved artifacts (dev)")
    x.add_argument("--train", default="/input/train.jsonl")
    x.add_argument("--artifacts", default="/artifacts")
    args = parser.parse_args()
    run = {"adapt": run_adapt, "eval": run_eval, "explain": run_explain}[args.cmd]
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
