# CIRCLE Project Breakdown (16 Parts)

This document breaks down the entire **CIRCLE** (**C**ontinuous **I**terative **R**efinement through **C**ritique & **L**earning **E**ngine) system into 16 logical, modular, and sequential implementation parts.

---

## Part 1: Project Architecture & Environment Setup
- **Goal:** Establish the complete project scaffolding, environment configuration templates, dependency manifests, container configurations, Kubernetes deployment templates, and version control initialization.
- **Key Deliverables:**
  - Base project folder structure (`trainer/`, `eval/`, `generator/`, `orchestrator/`, `docker/`).
  - Configuration files (`.gitignore`, `.env.example`, `requirements.txt`).
  - Docker containers specification (`docker/*.Dockerfile`).
  - Kubernetes manifests (`orchestrator/k8s/*.yaml`).
  - Skeleton entrypoints for each pipeline service.

---

## Part 2: Base PyTorch Trainer & QLoRA Configuration
- **Goal:** Build the fine-tuning engine using PyTorch, Hugging Face `transformers`, and QLoRA (`peft` + `bitsandbytes`).
- **Key Deliverables:**
  - Base model loader supporting 4-bit quantization optimized for low VRAM (4GB VRAM target).
  - Training loop script (`trainer/train.py`) supporting gradient accumulation, mixed precision (FP16/BF16), and checkpoint management.
  - Adapter save/load routines for per-stage QLoRA weights.

---

## Part 3: Curriculum Stage Configuration & Dataset Schemas
- **Goal:** Define the developmental curriculum framework and dataset loading pipelines across all 5 stages.
- **Key Deliverables:**
  - Stage configuration files (`trainer/curriculum/stage_*.json` or YAML).
  - Stage 1 (Base English), Stage 2 (Summarization), Stage 3 (Grammar), Stage 4 (Punctuation), Stage 5 (Conversational Understanding).
  - Flexible PyTorch Dataset classes and tokenization utilities.

---

## Part 4: Replay Buffer & Anti-Forgetting Mechanism
- **Goal:** Prevent catastrophic forgetting as the model advances through sequential developmental stages.
- **Key Deliverables:**
  - Replay buffer manager (`trainer/replay_buffer.py`).
  - Dynamic dataset blending logic that samples prior stage samples and merges them into the current training epoch.
  - Stratified replay sampling controls based on stage difficulty and retention metrics.

---

## Part 5: Evaluator / Critic Agent Integration
- **Goal:** Implement the high-capacity evaluation agent connecting to hosted 70B models (via Groq API).
- **Key Deliverables:**
  - Groq API client interface (`eval/critique.py`).
  - Prompt templates tailored for qualitative feedback generation across probe outputs.
  - Error handling, rate-limiting, and response parsing.

---

## Part 6: Structured Failure Mode Analysis & Parser
- **Goal:** Transform unstructured LLM critiques into machine-readable, categorized failure modes.
- **Key Deliverables:**
  - Pydantic models for structured failure mode reports (`eval/failure_parser.py`).
  - Categorization system (e.g., entity dropping, clause boundary error, speaker drift).
  - Failure severity matrix and dataset commissioning triggers.

---

## Part 7: Targeted Prompt Writer
- **Goal:** Automatically synthesize data-generation prompts tailored to specific identified model weaknesses.
- **Key Deliverables:**
  - Prompt generation logic (`eval/prompt_writer.py`).
  - Dynamic prompt synthesis that instructs the data generator to stress-test specific weak scenarios.
  - Meta-prompt templates for targeted dataset synthesis.

---

## Part 8: Lightweight Student Distillation Evaluator
- **Goal:** Reduce inference cost and latency during frequent training validation checks.
- **Key Deliverables:**
  - Distillation evaluator student module (`eval/distilled_eval.py`).
  - Fast scoring heuristic model trained on prior 70B critic judgements.
  - Tiered evaluation pipeline: lightweight fast checks vs full 70B deep critique runs.

---

## Part 9: Local DeepSeek Data Generator Engine
- **Goal:** Build the local synthetic data generation pipeline powered by DeepSeek 32B.
- **Key Deliverables:**
  - Local inference runner (`generator/generate.py`) supporting Ollama, vLLM, or GGUF endpoints.
  - Prompt execution queue and throughput management.
  - Batching and stream parsing for raw generated synthetic samples.

---

## Part 10: Synthetic Data Post-Processing & Validation Gating
- **Goal:** Filter out noise, hallucinations, invalid formatting, and duplicates before dataset insertion.
- **Key Deliverables:**
  - Quality validator (`generator/validator.py`).
  - Deduplication engine, regex formatting checks, and schema validation.
  - Dataset merger appending validated samples to the active training buffer.

---

## Part 11: Containerization & Build Pipeline Optimization
- **Goal:** Package all services into reproducible Docker images.
- **Key Deliverables:**
  - `docker/trainer.Dockerfile`, `docker/eval.Dockerfile`, `docker/generator.Dockerfile`.
  - Multi-stage Docker builds optimized for PyTorch, CUDA runtime, and minimal image size.
  - Container healthchecks and entrypoint scripts.

---

## Part 12: Kubernetes Manifests & Pod Management Specs
- **Goal:** Configure production-grade Kubernetes orchestration specs.
- **Key Deliverables:**
  - Deployment and Job manifests for trainer, evaluator, generator, and orchestrator.
  - ConfigMaps and Secrets for API keys and environment variables.
  - PersistentVolumeClaims (PVC) for shared checkpoint and dataset storage.

---

## Part 13: Orchestrator Control Loop & State Engine
- **Goal:** Implement the central closed-loop master state controller.
- **Key Deliverables:**
  - Kubernetes-aware loop controller (`orchestrator/loop_controller.py`).
  - Closed-loop state machine (Train → Eval → Generate → Retrain).
  - State persistence, resume-on-failure logic, and automated stage advancement thresholds.

---

## Part 14: Stage 1-3 Curriculum Probes & Initial Benchmark Datasets
- **Goal:** Curate seed datasets and evaluation probes for early developmental stages.
- **Key Deliverables:**
  - Stage 1 (Base English) seed data and evaluation probe prompts.
  - Stage 2 (Summarization) seed data and evaluation probe prompts.
  - Stage 3 (Grammar) seed data and evaluation probe prompts.

---

## Part 15: Stage 4-5 Curriculum Probes & Dialogue Datasets
- **Goal:** Curate seed datasets and evaluation probes for advanced developmental stages.
- **Key Deliverables:**
  - Stage 4 (Punctuation & Clause Boundaries) seed data and probe prompts.
  - Stage 5 (Conversational Understanding & Speaker Tracking) seed data and probe prompts.
  - Probe task execution harness.

---

## Part 16: End-to-End System Integration, Verification & Deployment
- **Goal:** Execute full system validation, end-to-end integration tests, and finalize project setup.
- **Key Deliverables:**
  - End-to-end test runner executing a mini-loop cycle.
  - Documentation and deployment verification scripts.
  - Final repository release and push to remote `main` branch.
