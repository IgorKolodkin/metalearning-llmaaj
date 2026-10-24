# Solution: a rulebook judge that learns from the expert's labels

> **Requirements: Docker and an Anthropic API key.** Both `adapt` and `eval` call the Anthropic API
> (`claude-opus-5-5` in `adapt`, `claude-haiku-5-5` in both), so the key needs access to these two models and
> the containers need internet access. Nothing else has to be configured: every other setting has a built-in
> default that is the submitted configuration.
>
> ```sh
> export ANTHROPIC_API_KEY=<your Anthropic API key>
> docker build -t plata-judge .
> docker run --rm -e ANTHROPIC_API_KEY -v "$PWD/in:/input:ro" -v "$PWD/artifacts:/artifacts" plata-judge adapt
> docker run --rm -e ANTHROPIC_API_KEY -v "$PWD/in:/input:ro" -v "$PWD/artifacts:/artifacts:ro" \
>   -v "$PWD/out:/output" plata-judge eval
> ```
>
> Each stage first sends one tiny test request per model and stops with a clear error and a nonzero exit code
> if the key is missing or rejected, a model is not available to it, or there is no network.

The judge keeps the expert's criteria as an explicit **rulebook**: a list of rules, each with the topics it
applies to, what counts as a violation and what does not. The same rulebook drives prediction and the jury
explanation, so what the jury reads is exactly what classifies the test set.

- **`adapt`** starts from an empty rulebook and runs up to `JUDGE_ROUNDS` rounds on the labeled train:
  apply the rulebook to every train conversation, show a strong model (`JUDGE_PROPOSE_MODEL`) every missed
  `bad` next to the most similar `good` and every false `bad`, collect add / edit / remove proposals, check
  every proposal on the whole train and accept greedily those that fix at least `JUDGE_MIN_FIX` errors and
  break at most `JUDGE_MAX_BREAK` correct items. Output: `rulebook.json` and `explanation.md`.
- **`eval`** checks every (rule, conversation) pair of a matching topic with a small model
  (`JUDGE_CHECK_MODEL`), one rule per call. A conversation is `bad` if any applicable rule is violated,
  otherwise `good`. The verdict is assembled in code, not by the model.
- **`explanation.md`** (the artifact for the jury) is rendered from the final rulebook and the check results on
  train: the decision procedure and priorities, every rule verbatim as the checker sees it, training examples
  where each rule fires with the expert label and the checker's evidence, every train example where the judge
  and the expert disagree, the rejected candidate rules, and the checker instruction used in every call.

Code: [`judge/`](judge/). `rulebook.py` (rule format, the exact text the checker sees), `adapt.py`,
`executor.py` (checks and the verdict), `explain.py`, `llm.py` (Anthropic calls with JSON-schema output),
`__main__.py` (the two stages).

## Build

```sh
docker build -t plata-judge .
```

## Run

Both stages call the Anthropic API, so both containers need network access and `ANTHROPIC_API_KEY`.

```sh
# adapt: labeled /input/train.jsonl -> /artifacts (rulebook.json, explanation.md)
docker run --rm -e ANTHROPIC_API_KEY \
  -v "$PWD/in:/input:ro" -v "$PWD/artifacts:/artifacts" \
  plata-judge adapt

# eval, in a separate container: unlabeled /input/test.jsonl + /artifacts -> /output/predictions.csv
docker run --rm -e ANTHROPIC_API_KEY \
  -v "$PWD/in:/input:ro" -v "$PWD/artifacts:/artifacts:ro" -v "$PWD/out:/output" \
  plata-judge eval
```

Paths can be overridden: `adapt --train PATH --artifacts DIR`, `eval --test PATH --artifacts DIR --out PATH`.
`eval` only reads `/artifacts` and needs nothing else from `adapt`.

Without Docker (Python 3.11+, [uv](https://docs.astral.sh/uv/)):

```sh
uv sync --frozen
uv run python -m judge adapt --train in/train.jsonl --artifacts artifacts
uv run python -m judge eval --test in/test.jsonl --artifacts artifacts --out out/predictions.csv
```

## Configuration (environment variables)

Only `ANTHROPIC_API_KEY` is required; the defaults are the submitted configuration.
Use the same values for `adapt` and `eval`: the checker model and effort must match the ones the rulebook was
adapted with.

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key |
| `ANTHROPIC_BASE_URL` | Anthropic API | Alternative endpoint (proxy / gateway), read by the Anthropic SDK |
| `JUDGE_CHECK_MODEL` | `claude-haiku-5-5` | Model for the per-rule checks, in `adapt` and `eval` |
| `JUDGE_CHECK_EFFORT` | `medium` | Reasoning effort of the checks |
| `JUDGE_PROPOSE_MODEL` | `claude-opus-5-5` | Model that proposes rulebook edits in `adapt` |
| `JUDGE_PROPOSE_EFFORT` | `high` | Reasoning effort of the proposals |
| `JUDGE_ROUNDS` | `3` | Maximum adapt rounds |
| `JUDGE_MIN_FIX` | `1` | A proposal is accepted only if it fixes at least this many train errors |
| `JUDGE_MAX_BREAK` | `1` | ... and breaks at most this many correctly judged train items |
| `JUDGE_CONCURRENCY` | `32` | Parallel API requests (lower it if the key hits rate limits; retries are automatic) |
| `JUDGE_CACHE` | `/tmp/judge-cache` | Disk cache of API responses; only saves money on reruns, never needed by `eval` |

No credentials are stored in the repository or the image.

## Cost and time

Measured on the public data (150 train, 80 test) from an empty cache, Docker on a laptop, default settings:

| Stage | API calls | Cost | Wall time |
|---|---|---|---|
| `adapt` (150 train) | 1002 | $0.95 | ~2 min |
| `eval` (80 test) | 412 | $0.11 | ~30 s |

The private run (190 train, 296 test) scales roughly linearly with the number of conversations: expect about
$1.5 and 3 to 5 minutes for `adapt` and about $0.5 and 2 minutes for `eval`.

If a single check or proposal fails after the SDK's automatic retries (refusal, malformed output), it is logged
to stderr and counted as "not violated" (or "no proposals"), so `eval` always writes a label for every test ID.
Problems that affect every request stop the run with a nonzero exit code instead of producing labels: a missing
or invalid key, no access to a model or no network (caught by the test request at start), and more than 20% of
checks failing during the run.

Inputs are used in the original language (`prepare_inputs.py` without `--language`).

## Reproduce on the public data

```sh
python3 src/prepare_inputs.py --train data/public_train.csv --test data/public_test.csv --out-dir out/public_input
mkdir -p out/in-train out/in-test out/artifacts out/pred
cp out/public_input/train.jsonl out/in-train/ && cp out/public_input/test.jsonl out/in-test/
docker run --rm -e ANTHROPIC_API_KEY -v "$PWD/out/in-train:/input:ro" -v "$PWD/out/artifacts:/artifacts" plata-judge adapt
docker run --rm -e ANTHROPIC_API_KEY -v "$PWD/out/in-test:/input:ro" -v "$PWD/out/artifacts:/artifacts:ro" \
  -v "$PWD/out/pred:/output" plata-judge eval
python3 src/evaluate.py --test data/public_test.csv --predictions out/pred/predictions.csv --out out/score.json
```
