# ==============================================================================
# CIRCLE Part 11: Production QLoRA Trainer Dockerfile
# Optimized multi-stage build for PyTorch, CUDA, and QLoRA fine-tuning.
# ==============================================================================

# --- Stage 1: Build Dependencies & Wheels ---
FROM python:3.10-slim AS builder

WORKDIR /build

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Stage 2: Final Runtime Image ---
FROM python:3.10-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH \
    CUDA_VISIBLE_DEVICES=0 \
    TORCH_HOME=/app/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Create non-root runtime directories with permissions
RUN mkdir -p /app/data /app/checkpoints /app/logs /app/.cache && \
    chmod -R 777 /app

# Copy application source code
COPY trainer/ ./trainer/
COPY requirements.txt .

# Healthcheck verifying Python environment and PyTorch import
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import torch, transformers, peft; print('Trainer Runtime Healthy')" || exit 1

ENTRYPOINT ["python", "-m", "trainer.train"]
