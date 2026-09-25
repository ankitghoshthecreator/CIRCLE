# CIRCLE
**C**ontinuous **I**terative **R**efinement through **C**ritique & **L**earning **E**ngine
 
A from-scratch language model training system built around a staged curriculum and a closed-loop, agent-driven data generation pipeline — where a critic model evaluates outputs and autonomously commissions the exact training data needed to fix what it finds wrong.
 
---
 
## Table of Contents
 
1. [Motivation](#motivation)
2. [High-Level Architecture](#high-level-architecture)
3. [The Curriculum](#the-curriculum)
4. [The Closed Loop: Train → Eval → Generate → Retrain](#the-closed-loop-train--eval--generate--retrain)
5. [Model Roles](#model-roles)
6. [Training Details](#training-details)
7. [Evaluation & Distillation](#evaluation--distillation)
8. [Infrastructure: Docker & Kubernetes](#infrastructure-docker--kubernetes)
9. [Tech Stack](#tech-stack)
10. [Repository Structure](#repository-structure)
11. [Setup & Usage](#setup--usage)
12. [Limitations](#limitations)
13. [Future Work](#future-work)
---
 
## Motivation
 
Most LLM projects start from a pretrained checkpoint and fine-tune toward a task. CIRCLE started from a different question: **in what order does a language model actually need to learn language?**
 
Rather than exposing the model to a single undifferentiated blend of text, CIRCLE trains through explicit developmental stages — mirroring (loosely) how language competence is scaffolded: raw fluency first, then compression (summarization), then structural correctness (grammar), then the fine-grained semantics of punctuation, and only then the pragmatics of multi-turn conversation.
 
A second, equally important question: **can evaluation itself drive data generation?** Instead of manually curating a bigger dataset when the model underperforms, CIRCLE closes the loop — a large critic model diagnoses *specific* weaknesses after each round and writes generation prompts that produce exactly the data needed to address them.
 
---
 
## High-Level Architecture
 
```
                    ┌─────────────────────────────────────────┐
                    │              ORCHESTRATOR                │
                    │         (Kubernetes-managed loop)         │
                    └───────────────────┬───────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
┌───────────────┐              ┌──────────────────┐            ┌────────────────────┐
│  TRAIN STAGE   │◄─────────────│  DATA GENERATOR   │◄───────────│    EVAL / CRITIC    │
│  GPT-2 + QLoRA │   new data    │  DeepSeek 32B      │  targeted  │   Groq-hosted 70B    │
│  (PyTorch)     │──────────────►│  (local inference) │  prompts   │   analyzes outputs   │
└───────────────┘  checkpoint    └──────────────────┘            └────────────────────┘
        ▲                                                                  │
        │                                                                  │
        └──────────────────────── next training round ◄────────────────────┘
```
 
Each stage runs as an independent, containerized service. The orchestrator advances the loop only once the current stage reports completion, so any single component can be restarted, scaled, or swapped without taking down the pipeline.
 
---
 
## The Curriculum
 
CIRCLE trains through five sequential stages. Each stage is trained to a competence threshold (as judged by the eval agent, see below) before the model advances.
 
| Stage | Objective | What it teaches the model |
|---|---|---|
| 1. Base English | Next-token prediction on general English corpora | Token-level fluency, basic syntax, vocabulary distribution |
| 2. Summarization | Compress long passages into short, faithful summaries | Meaning extraction over surface-form copying; forces internal representation of "what matters" in a passage rather than local n-gram prediction |
| 3. Grammar | Structured grammatical correctness tasks | Explicit syntactic well-formedness, agreement, tense consistency |
| 4. Punctuation | Fine-grained use of `. , ! ?` and clause boundaries | Punctuation is trained as its own signal, not a byproduct of fluency — a period changes the semantic scope of everything before it, and models trained without emphasis on this often produce fluent-but-flat, run-on text |
| 5. Conversational Understanding | Multi-turn dialogue, turn-taking, speaker tracking | Pragmatics: who is speaking, what's being responded to, and how context carries across turns |
 
Advancing a stage does not discard prior stages' data — each new stage's dataset is blended with a retained sample of prior-stage data to guard against catastrophic forgetting as the curriculum progresses.
 
---
 
## The Closed Loop: Train → Eval → Generate → Retrain
 
This is the core novelty of CIRCLE — evaluation doesn't just *score* the model, it **drives what gets trained on next.**
 
**Step-by-step:**
 
1. **Train**: The current-stage dataset is used to fine-tune the model (GPT-2 base + QLoRA adapters) for one round.
2. **Generate output samples**: The freshly trained checkpoint produces a batch of outputs across a fixed set of probe tasks for the current curriculum stage.
3. **Critique (Eval Agent)**: A 70B model hosted on Groq analyzes the output batch, identifying specific, categorized failure modes (e.g., "summaries drop named entities," "run-on sentences with no clause boundary," "loses track of speaker after 3rd turn").
4. **Targeted prompt writing**: For each failure mode identified, the eval agent writes a generation prompt designed to produce training examples that specifically stress-test and correct that weakness — rather than generic additional data.
5. **Data generation**: A local 32B DeepSeek model receives these prompts and generates the requested dataset.
6. **Retrain**: The newly generated, weakness-targeted dataset is merged into the training set for the next round, and the loop returns to Step 1.
This turns evaluation from a passive scorecard into an active data-sourcing mechanism — the system tells itself what it doesn't know and then teaches itself that specific thing.
 
---
 
## Model Roles
 
| Role | Model | Where it runs | Purpose |
|---|---|---|---|
| Base trainable model | GPT-2 | Local (PyTorch) | The model actually being trained via the curriculum |
| Eval / Critic agent | 70B model | Groq-hosted (inference API) | Diagnoses failure modes in current-checkpoint outputs; writes next-round data generation prompts |
| Data generator | DeepSeek 32B | Local inference | Generates targeted training data on demand, based on the critic's prompts |
 
Splitting critic and generator across a hosted high-capacity model (Groq 70B, for fast, high-quality reasoning about failure modes) and a local model (DeepSeek 32B, for cost-controlled bulk data generation) balances evaluation quality against generation throughput and cost.
 
---
 
## Training Details
 
- **Base model**: GPT-2
- **Fine-tuning method**: QLoRA — quantized low-rank adapters, chosen to keep training tractable on constrained local hardware (RTX 3050, 4GB VRAM) while still allowing full-curriculum iteration
- **Framework**: PyTorch
- **Per-stage checkpointing**: each curriculum stage's final checkpoint is retained, enabling stage-level rollback and ablation (i.e., comparing "grammar-stage-only" vs "full curriculum" checkpoints)
- **Replay buffer**: a retained sample of prior-stage data is blended into each new stage to mitigate catastrophic forgetting
---
 
## Evaluation & Distillation
 
Knowledge distillation is used specifically for **lightweight eval scoring** — rather than running the full 70B critic model for every intermediate metric, a distilled evaluation signal (student model trained to approximate the 70B critic's judgments on cheaper, faster inference) is used for frequent/lightweight checks, with the full 70B critic reserved for the end-of-round deep critique that drives data generation. This keeps per-round evaluation costs low while preserving high-quality critique at the points that matter most (i.e., right before new data gets generated).
 
---
 
## Infrastructure: Docker & Kubernetes
 
Every stage of the loop is containerized independently:
 
- **Trainer container**: PyTorch + QLoRA training job
- **Eval/critic container**: handles calls to the Groq-hosted 70B model and prompt-writing logic
- **Generator container**: runs local DeepSeek 32B inference for dataset generation
- **Orchestrator**: a Kubernetes-managed control loop that sequences the four steps (train → eval → generate → retrain), tracks per-stage completion state, and restarts any failed stage container without restarting the whole pipeline
Benefits of this layout:
- Each component scales independently (e.g., the generator can be scaled up during heavy dataset-generation phases without touching the trainer)
- Failures are isolated — a crashed eval container doesn't lose in-progress training state
- The pipeline is restart-safe at the stage level, not just at the full-run level
---
 
## Tech Stack
 
- **PyTorch** — model training
- **Python** — orchestration glue, data pipeline scripts
- **GPT-2** — base trainable model
- **QLoRA** — parameter-efficient fine-tuning
- **Knowledge Distillation** — lightweight eval scoring
- **Groq** (70B model) — critic/eval agent, hosted inference
- **Ollama / Qwen 2.5 Coder 7B** — local synthetic data generation
- **Docker** — containerization of each pipeline stage
- **Kubernetes** — orchestration of the train/eval/generate/retrain loop
---
 
## Repository Structure
 
```
circle/
├── trainer/                # PyTorch + QLoRA training job
│   ├── train.py
│   ├── checkpoint_tracker.py# 100th step periodic & lowest loss saver + loss_log.csv
│   ├── curriculum/         # per-stage dataset configs
│   └── checkpoints/
├── eval/                   # Groq-hosted critic agent & RLAIF reward calculator
│   ├── critique.py
│   ├── rl_reward.py
│   └── failure_parser.py
├── generator/              # local Ollama / DeepSeek data generation
│   ├── generate.py
│   └── rl_prompt_builder.py
├── orchestrator/           # Kubernetes control loop & state machine
│   ├── loop_controller.py
│   └── k8s/
├── scripts/                # CLI runners & utilities
│   ├── rl_loop.py          # True RLAIF training loop runner
│   ├── chat.py             # Interactive LLM terminal chat CLI
│   ├── generate_chat_dataset.py # 1,000 diverse chat pair generator
│   └── rl_monitor.py       # Live terminal training dashboard
├── tests/                  # Full regression test suite
└── README.md
```
 
---
 
## Setup & Usage
 
### 1. Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run True RLAIF Training Loop
```powershell
# Mock Mode (Fast test execution)
python scripts/rl_loop.py --stages 1 2 3 --rl-steps 5 --mock

# Live Mode (Requires GROQ_API_KEY in .env + Ollama)
python scripts/rl_loop.py --stages 1 2 3 4 5 --rl-steps 2000 --no-mock
```

### 3. Generate 1,000 Diverse Conversational Pairs & QLoRA Fine-Tune
```powershell
# Step A: Generate 1,000 high-variety chat pairs via Ollama
python scripts/generate_chat_dataset.py --count 1000

# Step B: Train for 5,000 steps with automatic 100-step checkpointing & CSV loss logging
python trainer/train.py --stage 5 --epochs 5 --batch-size 1 --grad-accum 1
```

### 4. Interactive Terminal Chat CLI
Converse with your fine-tuned model checkpoint:
```powershell
python scripts/chat.py --checkpoint trainer/checkpoints/checkpoint_best
```

---
 
## Limitations & Features
 
- **100th-Step Periodic & Best Loss Checkpointing**: Saves `checkpoint_step_100`, `checkpoint_step_200`, etc., and overwrites `checkpoint_best` whenever a new lowest loss is achieved.
- **CSV Loss Monitoring**: Tracks `step` and `loss` on every step in `trainer/checkpoints/loss_log.csv` to monitor for overfitting.
- **Smart Adapter Resolution**: `scripts/chat.py` automatically resolves QLoRA adapters from curriculum stage folders (`stage_5`, `stage_4`, etc.) and formats input as `User: <prompt>\nAssistant:`.
- **4GB VRAM Optimization**: Uses QLoRA 4-bit quantization and gradient accumulation to run full curriculum fine-tuning on constrained hardware.

 