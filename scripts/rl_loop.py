"""
scripts/rl_loop.py
CIRCLE Part 17: True RLAIF (Reinforcement Learning from AI Feedback) Runner

This is the real RL training loop that tightly wires together:
  1. GPT-2 inference   — generates text for the current stage
  2. Groq critic       — scores outputs and identifies failure modes
  3. RLRewardCalculator— converts critique to a scalar reward (0–1)
  4. RLPromptBuilder   — builds targeted Ollama prompts from Groq evidence
  5. Ollama generator  — generates corrective training pairs
  6. Dataset merger    — merges new pairs into the stage dataset
  7. GPT-2 trainer     — retrains on enriched dataset (1 epoch per RL step)

Loop per stage:
    for rl_step in 1..max_rl_steps:
        probe → Groq critique → reward
        if reward < pass_threshold:
            build Ollama prompts from failure evidence
            generate corrective pairs via Ollama (or mock)
            validate + merge into dataset
            retrain GPT-2 on enriched dataset
        log RL step results
    advance to next stage when reward >= threshold

Usage:
    # Mock mode (no Groq API, no Ollama, no GPU needed)
    python scripts/rl_loop.py --stages 1 --rl-steps 2 --mock

    # Live mode (requires GROQ_API_KEY in .env and Ollama running)
    python scripts/rl_loop.py --stages 1 2 --rl-steps 3 --no-mock

    # Resume from a previous run
    python scripts/rl_loop.py --stages 1 --rl-steps 3 --resume-run rl_20260922_...
"""

import os
import sys
import json
import time
import logging
import argparse
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RLLoop")


# ──────────────────────────────────────────────────────────────
# Lazy imports (allow CPU-only or no-GPU execution for testing)
# ──────────────────────────────────────────────────────────────

def _load_critic(api_key: str):
    try:
        from eval.critique import GroqCriticAgent
        return GroqCriticAgent(api_key=api_key)
    except Exception as e:
        logger.warning(f"[RLLoop] GroqCriticAgent not loadable: {e}")
        return None


def _load_train_fn():
    try:
        from trainer.train import train_stage
        return train_stage
    except Exception as e:
        logger.warning(f"[RLLoop] trainer.train not importable: {e}")
        return None


def _load_probe_runner():
    try:
        from orchestrator.probe_runner import ProbeRunner
        return ProbeRunner
    except Exception as e:
        logger.warning(f"[RLLoop] ProbeRunner not importable: {e}")
        return None


def _load_generator():
    try:
        from generator.generate import LocalDataGeneratorEngine
        return LocalDataGeneratorEngine
    except Exception as e:
        logger.warning(f"[RLLoop] LocalDataGeneratorEngine not importable: {e}")
        return None


def _load_validator():
    try:
        from generator.validator import SyntheticDataValidator, DatasetMerger
        return SyntheticDataValidator, DatasetMerger
    except Exception as e:
        logger.warning(f"[RLLoop] Validator/Merger not importable: {e}")
        return None, None


# ──────────────────────────────────────────────────────────────
# RL Step record
# ──────────────────────────────────────────────────────────────

class RLStepRecord:
    """Structured log entry for one RL step."""

    def __init__(
        self,
        run_id: str,
        stage_id: int,
        rl_step: int,
        reward: float,
        grade: str,
        samples_generated: int,
        samples_merged: int,
        training_loss: Optional[float],
        backend_used: str,
        failure_modes: List[str],
        advance: bool,
    ):
        self.run_id = run_id
        self.stage_id = stage_id
        self.rl_step = rl_step
        self.reward = reward
        self.grade = grade
        self.samples_generated = samples_generated
        self.samples_merged = samples_merged
        self.training_loss = training_loss
        self.backend_used = backend_used
        self.failure_modes = failure_modes
        self.advance = advance
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "rl_step": self.rl_step,
            "reward": round(self.reward, 4),
            "grade": self.grade,
            "samples_generated": self.samples_generated,
            "samples_merged": self.samples_merged,
            "training_loss": self.training_loss,
            "backend_used": self.backend_used,
            "failure_modes": self.failure_modes,
            "advance": self.advance,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        loss_str = f"{self.training_loss:.4f}" if self.training_loss else "N/A"
        return (
            f"  Stage {self.stage_id:2d} | RL-Step {self.rl_step:2d} | "
            f"Reward {self.reward:.3f} | Grade: {self.grade:7s} | "
            f"Samples: +{self.samples_merged:3d} | Loss: {loss_str} | "
            f"{'→ ADVANCE' if self.advance else '  RETRAIN'}"
        )


# ──────────────────────────────────────────────────────────────
# Core RL Loop
# ──────────────────────────────────────────────────────────────

class RLOrchestrator:
    """
    True RLAIF closed-loop orchestrator.

    Wires Groq → RLRewardCalculator → RLPromptBuilder → Ollama → trainer.
    Each stage runs up to max_rl_steps reinforcement learning iterations
    before advancing to the next stage.
    """

    def __init__(
        self,
        stages: List[int],
        max_rl_steps: int = 3,
        pass_threshold: float = 0.85,
        data_dir: str = "./data",
        checkpoint_dir: str = "./trainer/checkpoints",
        log_dir: str = "./logs",
        mock_mode: bool = True,
        groq_api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.stages = stages
        self.max_rl_steps = max_rl_steps
        self.pass_threshold = pass_threshold
        self.data_dir = os.path.abspath(data_dir)
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.log_dir = os.path.abspath(log_dir)
        self.mock_mode = mock_mode
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model or os.getenv("GENERATOR_MODEL", "qwen2.5-coder:7b")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.run_id = run_id or (
            "rl_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            + "_" + uuid.uuid4().hex[:6]
        )
        self.log_path = os.path.join(self.log_dir, f"{self.run_id}.jsonl")
        self.history: List[RLStepRecord] = []

        # Lazy-load heavy components
        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self._critic = None if mock_mode else _load_critic(api_key)
        self._train_fn = _load_train_fn()

        ProbeRunnerCls = _load_probe_runner()
        self._probe_runner_cls = ProbeRunnerCls

        GeneratorCls = _load_generator()
        self._generator = GeneratorCls(
            backend="ollama" if not mock_mode else "mock",
            endpoint_url=ollama_url,
            model_name=self.ollama_model,
        ) if GeneratorCls else None

        ValidatorCls, MergerCls = _load_validator()
        self._validator_cls = ValidatorCls
        self._merger_cls = MergerCls

        # RL reward calculator
        from eval.rl_reward import RLRewardCalculator
        self._reward_calc = RLRewardCalculator(pass_threshold=pass_threshold)

        # RL prompt builder
        from generator.rl_prompt_builder import RLPromptBuilder
        self._prompt_builder = RLPromptBuilder()

    # ──────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────

    def run(self) -> List[RLStepRecord]:
        """Execute the full multi-stage RL loop."""
        logger.info(
            f"\n{'#'*65}\n"
            f"  CIRCLE RLAIF Loop — Run: {self.run_id}\n"
            f"  Stages: {self.stages}  |  Max RL Steps/Stage: {self.max_rl_steps}\n"
            f"  Pass Threshold: {self.pass_threshold}  |  Mock: {self.mock_mode}\n"
            f"{'#'*65}\n"
        )

        for stage_id in self.stages:
            self._run_stage(stage_id)

        self._print_final_summary()
        return self.history

    # ──────────────────────────────────────────────────────────
    # Stage loop
    # ──────────────────────────────────────────────────────────

    def _run_stage(self, stage_id: int) -> None:
        logger.info(
            f"\n{'='*65}\n"
            f"  STAGE {stage_id} — Starting RL loop\n"
            f"{'='*65}"
        )

        for rl_step in range(1, self.max_rl_steps + 1):
            logger.info(
                f"\n>>> Stage {stage_id} | RL Step {rl_step}/{self.max_rl_steps}"
            )

            # Step 1: Probe (generate model outputs for eval)
            probe_outputs = self._step_probe(stage_id)

            # Step 2: Critic (Groq or mock) + reward
            reward_obj, critique_raw = self._step_critique(stage_id, probe_outputs)

            # Step 3: If below threshold → generate corrective data + retrain
            samples_generated = 0
            samples_merged = 0
            training_loss = None
            backend_used = "none"

            if reward_obj.should_generate:
                # Build targeted Ollama prompts from Groq failure evidence
                prompt_batch = self._step_build_prompts(
                    stage_id=stage_id,
                    rl_step=rl_step,
                    reward_obj=reward_obj,
                    critique_raw=critique_raw,
                )

                # Generate corrective pairs via Ollama (or mock)
                gen_batches, samples_generated, backend_used = self._step_generate(
                    stage_id=stage_id,
                    prompt_batch=prompt_batch,
                    reward=reward_obj.reward,
                )

                # Validate + merge into dataset
                samples_merged = self._step_merge(stage_id, gen_batches)

                # Retrain GPT-2 on enriched dataset
                training_loss = self._step_retrain(stage_id)

            # Record and log
            failure_modes = []
            if reward_obj.failure_report:
                failure_modes = [
                    str(fm.category)
                    for fm in reward_obj.failure_report.failure_modes
                ]

            advance = reward_obj.should_advance or (rl_step >= self.max_rl_steps)

            record = RLStepRecord(
                run_id=self.run_id,
                stage_id=stage_id,
                rl_step=rl_step,
                reward=reward_obj.reward,
                grade=reward_obj.grade,
                samples_generated=samples_generated,
                samples_merged=samples_merged,
                training_loss=training_loss,
                backend_used=backend_used,
                failure_modes=failure_modes,
                advance=advance,
            )
            self.history.append(record)
            self._write_log(record)

            logger.info(str(record))

            if reward_obj.should_advance:
                logger.info(
                    f"\n  ✓ Stage {stage_id} PASSED (reward={reward_obj.reward:.3f} "
                    f">= {self.pass_threshold}). Advancing.\n"
                )
                break

            if rl_step >= self.max_rl_steps:
                logger.warning(
                    f"\n  ⚠ Stage {stage_id} hit max RL steps ({self.max_rl_steps}). "
                    f"Forcing advance. Final reward: {reward_obj.reward:.3f}\n"
                )

    # ──────────────────────────────────────────────────────────
    # Individual steps
    # ──────────────────────────────────────────────────────────

    def _step_probe(self, stage_id: int) -> List[Dict[str, str]]:
        """Run model inference probes for this stage."""
        if self._probe_runner_cls is None or self.mock_mode:
            # Mock probe outputs
            mock_outputs = [
                {"prompt": "The primary purpose of a language model is to",
                 "output": "generate text text text text generate text text text."},
                {"prompt": "Explain what photosynthesis does.",
                 "output": "Photosynthesis is photosynthesis which does photosynthesis"},
                {"prompt": "Write a sentence about the ocean.",
                 "output": "The ocean is big and it is very very very very big the ocean"},
            ]
            logger.info(f"[Probe] Mock mode — using {len(mock_outputs)} mock probe outputs.")
            return mock_outputs

        runner = self._probe_runner_cls(checkpoint_path=None, use_mock=False)
        results = runner.run_probes(stage_id=stage_id)
        logger.info(f"[Probe] Stage {stage_id}: {len(results)} probe outputs collected.")
        return results

    def _step_critique(self, stage_id: int, probe_outputs: List[Dict]) -> tuple:
        """
        Send probe outputs to Groq critic (or mock) and compute reward.

        Returns:
            (RLReward, critique_raw_str)
        """
        from eval.rl_reward import RLRewardCalculator

        # Stage name lookup
        stage_name = f"Stage {stage_id}"
        try:
            from trainer.curriculum.stage_config import STAGES
            cfg = STAGES.get(stage_id)
            if cfg:
                stage_name = cfg.name
                objectives = cfg.objectives
            else:
                objectives = ["fluency", "coherence", "syntax"]
        except Exception:
            objectives = ["fluency", "coherence", "syntax"]

        if self._critic is not None:
            logger.info(f"[Critique] Sending {len(probe_outputs)} probes to Groq critic...")
            critique_result = self._critic.analyze_stage_outputs(
                stage_id=stage_id,
                stage_name=stage_name,
                objectives=objectives,
                probe_results=probe_outputs,
            )
            critique_raw = critique_result.get("critique_raw", "")
            if critique_raw:
                reward = self._reward_calc.compute(stage_id, stage_name, critique_raw)
                logger.info(
                    f"[Critique] Groq critique received ({len(critique_raw)} chars). "
                    f"Reward: {reward.reward:.3f} ({reward.grade})"
                )
                return reward, critique_raw

        # Mock / fallback — use lightweight student evaluator
        logger.info("[Critique] Mock/fallback mode — using student evaluator for reward.")
        from eval.distilled_eval import LightweightStudentEvaluator
        student = LightweightStudentEvaluator()
        report = student.evaluate_batch(
            stage_id, stage_name, probe_outputs, escalation_threshold=self.pass_threshold
        )
        # Build a synthetic critique text from flags for the failure parser
        flag_text = "; ".join(report.escalation_reasons) if report.escalation_reasons else "no major failures detected"
        reward = self._reward_calc.compute_from_student_score(
            mean_score=report.mean_overall_score,
            escalation_reasons=report.escalation_reasons,
        )
        logger.info(
            f"[Critique] Student eval: score={report.mean_overall_score:.3f}, "
            f"reward={reward.reward:.3f} ({reward.grade})"
        )
        return reward, flag_text

    def _step_build_prompts(self, stage_id: int, rl_step: int, reward_obj, critique_raw: str):
        """Use RLPromptBuilder to create targeted Ollama prompts from Groq evidence."""
        stage_name = f"Stage {stage_id}"
        try:
            from trainer.curriculum.stage_config import STAGES
            cfg = STAGES.get(stage_id)
            if cfg:
                stage_name = cfg.name
        except Exception:
            pass

        if reward_obj.failure_report is not None:
            batch = self._prompt_builder.build_from_report(
                report=reward_obj.failure_report,
                stage_id=stage_id,
                rl_step=rl_step,
                stage_name=stage_name,
            )
        else:
            # Fall back to raw critique text
            batch = self._prompt_builder.build_from_raw_critique(
                critique_raw=critique_raw,
                stage_id=stage_id,
                stage_name=stage_name,
                rl_step=rl_step,
            )

        logger.info(
            f"[PromptBuild] {len(batch.requests)} Ollama requests built "
            f"({batch.total_samples_requested} samples requested)."
        )
        return batch

    def _step_generate(
        self,
        stage_id: int,
        prompt_batch,
        reward: float,
    ) -> tuple:
        """Call Ollama (or mock) for each OllamaPromptRequest in the batch."""
        if self._generator is None:
            logger.warning("[Generate] No generator available. Skipping.")
            return [], 0, "none"

        all_batches = []
        total_samples = 0
        backend = "unknown"

        for req in prompt_batch.requests:
            batch = self._generator.rl_generate_from_request(
                request=req,
                reward_score=reward,
            )
            all_batches.append(batch)
            total_samples += batch.total_samples
            # Detect which backend was used from first sample metadata
            if batch.samples:
                backend = batch.samples[0].generation_metadata.get("backend", "unknown")

        logger.info(
            f"[Generate] Stage {stage_id}: {total_samples} corrective samples "
            f"generated via '{backend}'."
        )
        return all_batches, total_samples, backend

    def _step_merge(self, stage_id: int, gen_batches: list) -> int:
        """Validate and merge corrective samples into the stage training dataset."""
        if not gen_batches or self._validator_cls is None or self._merger_cls is None:
            logger.warning("[Merge] Validator or merger not available. Skipping merge.")
            return 0

        validator = self._validator_cls()
        merger = self._merger_cls()
        total_merged = 0

        for batch in gen_batches:
            _, valid_examples = validator.validate_batch(batch)
            added = merger.merge_into_stage_dataset(
                validated_examples=valid_examples,
                stage_id=stage_id,
                data_dir=self.data_dir,
            )
            total_merged += added

        logger.info(f"[Merge] Stage {stage_id}: {total_merged} net-new examples merged.")
        return total_merged

    def _step_retrain(self, stage_id: int) -> Optional[float]:
        """Retrain GPT-2 on the enriched dataset (1 epoch). Returns avg loss."""
        from trainer.checkpoint_tracker import CheckpointTracker
        tracker = CheckpointTracker(checkpoint_dir=self.checkpoint_dir)

        if self.mock_mode or self._train_fn is None:
            mock_loss = round(1.5 - (len(self.history) * 0.08), 4)
            mock_loss = max(0.3, mock_loss)  # floor
            logger.info(f"[Retrain] Mock mode — simulated loss: {mock_loss:.4f}")
            tracker.record_step(loss=mock_loss)
            return mock_loss

        logger.info(f"[Retrain] Starting real GPT-2 QLoRA retraining for Stage {stage_id}...")
        try:
            checkpoint_path = self._train_fn(
                stage_id=stage_id,
                epochs=1,
                batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "1")),
                grad_accum_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4")),
                lr=float(os.getenv("LEARNING_RATE", "2e-4")),
                checkpoint_dir=self.checkpoint_dir,
                data_dir=self.data_dir,
            )
            logger.info(f"[Retrain] Stage {stage_id} checkpoint saved → {checkpoint_path}")
            # trainer doesn't return loss directly; return None to signal "ran OK"
            return None
        except Exception as e:
            logger.error(f"[Retrain] Training failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # Logging + Summary
    # ──────────────────────────────────────────────────────────

    def _write_log(self, record: RLStepRecord) -> None:
        """Append one JSON line to the run's JSONL log."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def _print_final_summary(self) -> None:
        """Print a structured table of all RL steps across all stages."""
        print(f"\n{'#'*65}")
        print(f"  CIRCLE RLAIF Loop Complete — Run: {self.run_id}")
        print(f"  Stages: {self.stages}  |  Total RL Steps: {len(self.history)}")
        print(f"{'#'*65}")
        print(f"\n  {'Stage':>5} {'RL-Step':>7} {'Reward':>8} {'Grade':>8} "
              f"{'Samples':>8} {'Loss':>8} {'Result':>10}")
        print(f"  {'-'*60}")
        for r in self.history:
            loss_str = f"{r.training_loss:.4f}" if r.training_loss else "    N/A"
            result = "→ ADVANCE" if r.advance else "  RETRAIN"
            print(
                f"  {r.stage_id:>5} {r.rl_step:>7} {r.reward:>8.3f} "
                f"{r.grade:>8} {r.samples_merged:>8} {loss_str:>8} {result:>10}"
            )
        print(f"\n  Log file: {self.log_path}")
        print(f"{'#'*65}\n")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CIRCLE RLAIF Training Loop")
    parser.add_argument(
        "--stages", type=int, nargs="+", default=[1],
        help="Stage IDs to run RL loop over (e.g. --stages 1 2 3)"
    )
    parser.add_argument(
        "--rl-steps", type=int, default=3,
        help="Max RL iterations per stage before forced advance"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Reward threshold to pass a stage (0–1)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./trainer/checkpoints",
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs",
        help="Path to RL log directory"
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama server URL"
    )
    parser.add_argument(
        "--ollama-model", type=str, default=None,
        help="Ollama model name (default: GENERATOR_MODEL env var or qwen2.5-coder:7b)"
    )
    parser.add_argument(
        "--mock", action="store_true", default=True,
        help="Run in mock mode (no GPU/API needed)"
    )
    parser.add_argument(
        "--no-mock", dest="mock", action="store_false",
        help="Run live (requires GROQ_API_KEY + Ollama)"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Custom run ID (auto-generated if not specified)"
    )
    args = parser.parse_args()

    orchestrator = RLOrchestrator(
        stages=args.stages,
        max_rl_steps=args.rl_steps,
        pass_threshold=args.threshold,
        data_dir=os.path.join(PROJECT_ROOT, args.data_dir),
        checkpoint_dir=os.path.join(PROJECT_ROOT, args.checkpoint_dir),
        log_dir=os.path.join(PROJECT_ROOT, args.log_dir),
        mock_mode=args.mock,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        run_id=args.run_id,
    )

    history = orchestrator.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
