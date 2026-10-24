"""Anthropic calls with structured JSON output, a disk cache and a spend ledger."""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import anthropic

# $ per 1M tokens (input, output)
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
CACHE_DIR = Path(os.environ.get("JUDGE_CACHE", Path(__file__).resolve().parents[1] / "cache"))

# Errors that hit every request alike (bad key, no access, unknown model): never swallowed as one failed check
FATAL = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError)

LEDGER = []  # one dict per call: tag, model, input, output, usd, usd_full, cached, secs
_client = None
_sem = None
_inflight = {}


def _get():
    global _client, _sem
    if _client is None:
        _client = anthropic.AsyncAnthropic(max_retries=8)
        _sem = asyncio.Semaphore(int(os.environ.get("JUDGE_CONCURRENCY", "32")))
    return _client, _sem


async def preflight(models):
    """Stop before any work if the API is unusable: no key, a rejected key, no model access, no network.

    Otherwise every check would fail on its own, be counted as "not violated" and eval would quietly
    write a file of all-good labels.
    """
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit("ERROR: ANTHROPIC_API_KEY is not set. Both adapt and eval call the Anthropic API: "
                         "export the key on the host and pass it with `docker run -e ANTHROPIC_API_KEY ...` "
                         "(see SOLUTION.md).")
    client, _ = _get()
    for model in dict.fromkeys(models):
        try:
            await client.messages.create(model=model, max_tokens=16,
                                         messages=[{"role": "user", "content": "Reply with: ok"}])
        except Exception as error:
            raise SystemExit(f"ERROR: test request to {model} failed: {error!r}. Check that the key is valid, "
                             "has access to this model and that the container has network access.")


def _usd(model, usage):
    pin, pout = PRICES.get(model, (0.0, 0.0))
    tokens_in = usage.input_tokens + (usage.cache_creation_input_tokens or 0) * 1.25 \
        + (usage.cache_read_input_tokens or 0) * 0.1
    return (tokens_in * pin + usage.output_tokens * pout) / 1e6


async def _fetch(key, req, tag):
    """Disk cache or one API call. Returns (data, ledger fields of a fresh call or None if from disk)."""
    path = CACHE_DIR / key[:2] / f"{key}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        return rec["data"], rec["usd"], None
    client, sem = _get()
    async with sem:
        start = time.time()
        if req["max_tokens"] > 16000:
            async with client.messages.stream(**req) as stream:
                msg = await stream.get_final_message()
        else:
            msg = await client.messages.create(**req)
        secs = time.time() - start
    if msg.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"{tag}: stop_reason={msg.stop_reason} ({msg._request_id})")
    data = json.loads(next(b.text for b in msg.content if b.type == "text"))
    usd = _usd(req["model"], msg.usage)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"data": data, "usd": usd}, ensure_ascii=False))
    os.replace(tmp, path)
    return data, usd, dict(input=msg.usage.input_tokens, output=msg.usage.output_tokens, secs=secs)


async def call_json(model, system, user, schema, *, effort="medium", max_tokens=8000, tag=""):
    """One request with output constrained to `schema`; identical requests are served from disk.

    Identical requests in flight at the same time share one API call, so two rulebooks with a common
    rule always get the same verdict for it (otherwise sampling noise shows up as R0 vs R1 flips).
    """
    req = dict(model=model, max_tokens=max_tokens, system=system,
               messages=[{"role": "user", "content": user}],
               output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}})
    # JUDGE_SALT forces fresh samples of the same requests, to measure checker noise
    salted = dict(req, salt=os.environ["JUDGE_SALT"]) if os.environ.get("JUDGE_SALT") else req
    key = hashlib.sha256(json.dumps(salted, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    owner = key not in _inflight
    if owner:
        _inflight[key] = asyncio.ensure_future(_fetch(key, req, tag))
    try:
        data, usd, fresh = await asyncio.shield(_inflight[key])
    finally:
        if owner:
            _inflight.pop(key, None)
    if owner and fresh:
        LEDGER.append(dict(tag=tag, model=model, usd=usd, usd_full=usd, cached=False, **fresh))
    else:
        LEDGER.append(dict(tag=tag, model=model, usd=0.0, usd_full=usd if owner else 0.0, cached=True,
                           input=0, output=0, secs=0.0))
    return data


def spend(prefix=""):
    """Totals of the ledger for tags starting with `prefix`."""
    rows = [r for r in LEDGER if r["tag"].startswith(prefix)]
    return dict(calls=len(rows), cached=sum(r["cached"] for r in rows),
                usd=round(sum(r["usd"] for r in rows), 4),
                usd_without_cache=round(sum(r["usd_full"] for r in rows), 4),
                input=sum(r["input"] for r in rows), output=sum(r["output"] for r in rows))
