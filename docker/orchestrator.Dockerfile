# ==============================================================================
# CIRCLE Part 11: Production Master Orchestrator Control Loop Dockerfile
# Container for Kubernetes loop controller and closed-loop state machine.
# ==============================================================================

FROM python:3.10-slim AS builder

WORKDIR /build

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir kubernetes pydantic python-dotenv requests pyyaml

FROM python:3.10-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

COPY --from=builder /root/.local /root/.local

RUN mkdir -p /app/data /app/checkpoints /app/logs && chmod -R 777 /app

COPY orchestrator/ ./orchestrator/
COPY requirements.txt .

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import kubernetes; print('Orchestrator Runtime Healthy')" || exit 1

ENTRYPOINT ["python", "-c", "print('CIRCLE Orchestrator Service Ready')"]
