# ==============================================================================
# CIRCLE Part 11: Production Evaluator / Critic Microservice Dockerfile
# Ultra-lightweight slim container for 70B Groq API critique & failure parsing.
# ==============================================================================

FROM python:3.10-slim AS builder

WORKDIR /build

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir groq pydantic python-dotenv requests pyyaml

FROM python:3.10-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

COPY --from=builder /root/.local /root/.local

RUN mkdir -p /app/logs && chmod -R 777 /app

COPY eval/ ./eval/
COPY requirements.txt .

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import eval.failure_parser, eval.prompt_writer; print('Eval Runtime Healthy')" || exit 1

ENTRYPOINT ["python", "-m", "eval.critique"]
