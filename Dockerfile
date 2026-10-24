FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY judge ./judge

# The LLM response cache lives outside /artifacts: it only saves money on reruns, eval never needs it
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 JUDGE_CACHE=/tmp/judge-cache

# docker run IMAGE adapt  -> /input/train.jsonl to /artifacts (rulebook.json, explanation.md)
# docker run IMAGE eval   -> /input/test.jsonl + /artifacts to /output/predictions.csv
ENTRYPOINT ["python", "-m", "judge"]
