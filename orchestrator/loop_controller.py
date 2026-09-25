"""
orchestrator/loop_controller.py
CIRCLE Part 13: Closed-Loop Orchestrator — Master State Controller

Implements the central Train → Eval → Generate → Merge → (Advance|Retrain) loop.
Integrates all pipeline components:
  - trainer.train.train_stage                   (Part 2)
  - eval.distilled_eval.TieredEvaluator         (Part 8)
  - eval.critique.GroqCriticAgent               (Part 5)
  - eval.failure_parser.FailureModeParser       (Part 6)
  - eval.prompt_writer.TargetedPromptWriter     (Part 7)
  - generator.generate.LocalDataGeneratorEngine (Part 9)
  - generator.validator.SyntheticDataValidator  (Part 10)
  - generator.validator.DatasetMerger           (Part 10)
  - orchestrator.state_machine.PipelineState    (Part 13)
  - orchestrator.probe_runner.ProbeRunner       (Part 13)

Features:
  - Full state machine with disk-backed resume-on-failure
  - Stage advancement threshold gating (mean_score >= threshold AND no HIGH/CRITICAL)
  - Max-iteration circuit breaker per stage
  - Dry-run / mock mode for testing without GPU
  - Structured JSON logging of all loop events
"""

import os
import sys
import json
import uuid
import logging
import argparse
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.state_machine import (
    LoopState, PipelineState, StageResult, StatePersistenceManager
)
from orchestrator.probe_runner import ProbeRunner, STAGE_PROBES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Helper: lazy-import heavy pipeline components
# (allows module to import cleanly on CPU-only hosts for testing)
# ─────────────────────────────────────────────────────────────────

def _import_pipeline():
    """Returns dict of pipeline component constructors / callables."""
    components: Dict[str, Any] = {}

    try:
        from eval.distilled_eval import LightweightStudentEvaluator, TieredEvaluator
        components["LightweightStudentEvaluator"] = LightweightStudentEvaluator
        components["TieredEvaluator"] = TieredEvaluator
    except ImportError as e:
        logger.warning(f"[Orchestrator] distilled_eval not importable: {e}")

    try:
        from eval.prompt_writer import TargetedPromptWriter
        components["TargetedPromptWriter"] = TargetedPromptWriter
    except ImportError as e:
        logger.warning(f"[Orchestrator] prompt_writer not importable: {e}")

    try:
        from generator.generate import LocalDataGeneratorEngine
        components["LocalDataGeneratorEngine"] = LocalDataGeneratorEngine
    except ImportError as e:
        logger.warning(f"[Orchestrator] generator not importable: {e}")

    try:
        from generator.validator import SyntheticDataValidator, DatasetMerger
        components["SyntheticDataValidator"] = SyntheticDataValidator
        components["DatasetMerger"] = DatasetMerger
    except ImportError as e:
        logger.warning(f"[Orchestrator] validator not importable: {e}")

    try:
        from eval.failure_parser import FailureModeReport
        components["FailureModeReport"] = FailureModeReport
    except ImportError as e:
        logger.warning(f"[Orchestrator] failure_parser not importable: {e}")

    return components


def _import_trainer():
    """Lazy-import trainer.train.train_stage (requires torch)."""
    try:
        from trainer.train import train_stage
        return train_stage
    except ImportError as e:
        logger.warning(f"[Orchestrator] trainer.train not importable: {e}")
        return None


def _import_groq_critic():
    """Lazy-import GroqCriticAgent (requires groq package)."""
    try:
        from eval.critique import GroqCriticAgent
        return GroqCriticAgent
    except ImportError as e:
        logger.warning(f"[Orchestrator] critique not importable: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

class ClosedLoopOrchestrator:
    """
    The CIRCLE master closed-loop controller.

    Manages the full Train → Eval → Generate → Merge → Advance loop
    across all 5 curriculum stages. State is persisted after every
    transition so the loop can be resumed after pod restarts.
    """

    def __init__(
        self,
        start_stage: int = 1,
        max_iterations: int = 3,
        stage_advance_threshold: float = 0.85,
        data_dir: str = "./data",
        checkpoint_dir: str = "./trainer/checkpoints",
        log_dir: str = "./logs",
        run_id: Optional[str] = None,
        resume: bool = False,
        mock_mode: bool = True,     # True = no GPU required; uses mock/stub pipeline components
        groq_api_key: Optional[str] = None,
    ):
        self.mock_mode = mock_mode
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

        # ── State manager (disk persistence) ─────────────────────
        self.state_manager = StatePersistenceManager(state_dir=log_dir)
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # ── Load or create pipeline state ────────────────────────
        if resume:
            loaded = self.state_manager.load(self.run_id)
            self.state = loaded or self._new_state(
                start_stage, max_iterations, stage_advance_threshold,
                data_dir, checkpoint_dir, log_dir
            )
        else:
            self.state = self._new_state(
                start_stage, max_iterations, stage_advance_threshold,
                data_dir, checkpoint_dir, log_dir
            )

        # ── Lazy-import pipeline components ──────────────────────
        self._components = _import_pipeline()
        self._train_stage_fn = _import_trainer()
        GroqCriticAgentCls = _import_groq_critic()

        # ── Groq critic (optional) ────────────────────────────────
        self._critic = None
        # Enable Groq critic whenever a real API key is present,
        # even in mock_mode — so the reward signal is real even when training is mocked.
        if GroqCriticAgentCls and self.groq_api_key and self.groq_api_key != "your_groq_api_key_here":
            try:
                self._critic = GroqCriticAgentCls(api_key=self.groq_api_key)
                logger.info("[Orchestrator] GroqCriticAgent initialised (live).")
            except Exception as e:
                logger.warning(f"[Orchestrator] GroqCriticAgent init failed: {e}")
                self._critic = None

        # ── Tiered evaluator ─────────────────────────────────────
        if "TieredEvaluator" in self._components and "LightweightStudentEvaluator" in self._components:
            self._tiered_eval = self._components["TieredEvaluator"](
                student_evaluator=self._components["LightweightStudentEvaluator"](),
                teacher_agent=self._critic,
                escalation_threshold=stage_advance_threshold,
            )
        else:
            self._tiered_eval = None

    # ─────────────────────────────────────────────────────────────
    # Factory helpers
    # ─────────────────────────────────────────────────────────────

    def _new_state(
        self, start_stage, max_iterations, threshold, data_dir, checkpoint_dir, log_dir
    ) -> PipelineState:
        os.makedirs(log_dir, exist_ok=True)
        return PipelineState(
            run_id=self.run_id,
            current_stage=start_stage,
            current_iteration=1,
            max_iterations_per_stage=max_iterations,
            stage_advance_threshold=threshold,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )

    # ─────────────────────────────────────────────────────────────
    # Core Step Implementations
    # ─────────────────────────────────────────────────────────────

    def _step_train(self) -> Optional[str]:
        """
        Step 1 — TRAINING: Fine-tune model for current stage.
        Returns checkpoint path or None (mock mode returns a placeholder path).
        """
        s = self.state
        logger.info(
            f"\n{'='*60}\n"
            f"  [STEP 1/5 — TRAINING]  Stage {s.current_stage}  |  Iter {s.current_iteration}\n"
            f"{'='*60}"
        )

        if self.mock_mode or self._train_stage_fn is None:
            ckpt = os.path.join(s.checkpoint_dir, f"stage_{s.current_stage}", "mock_checkpoint")
            logger.info(f"[Train] MOCK MODE — skipping real training. Checkpoint placeholder: {ckpt}")
            return ckpt

        # Gather any newly merged synthetic examples from the data directory
        # so the trainer uses the enriched dataset, not just the static seed.
        custom_examples = None
        try:
            from trainer.curriculum.dataset_handler import load_stage_dataset
            custom_examples = load_stage_dataset(s.current_stage, data_dir=s.data_dir)
            logger.info(f"[Train] Loaded {len(custom_examples)} examples (seed + synthetic) for Stage {s.current_stage}.")
        except Exception as e:
            logger.warning(f"[Train] Could not load enriched dataset: {e}. Using trainer default.")

        try:
            ckpt = self._train_stage_fn(
                stage_id=s.current_stage,
                epochs=1,
                batch_size=1,
                grad_accum_steps=4,
                lr=2e-4,
                checkpoint_dir=s.checkpoint_dir,
                data_dir=s.data_dir,
                custom_examples=custom_examples,
            )
            logger.info(f"[Train] Stage {s.current_stage} training complete -> {ckpt}")
            return ckpt
        except Exception as e:
            s.log_error(f"Training failed: {e}")
            return None

    def _step_evaluate(self, checkpoint_path: Optional[str]) -> Dict[str, Any]:
        """
        Step 2 — EVALUATING: Run probes and evaluate with TieredEvaluator.
        Returns eval result dict with mean_score and failure_mode_report.
        """
        s = self.state
        logger.info(
            f"\n{'='*60}\n"
            f"  [STEP 2/5 — EVALUATING]  Stage {s.current_stage}  |  Iter {s.current_iteration}\n"
            f"{'='*60}"
        )

        # Load checkpoint if available; otherwise mock
        use_mock_probe = self.mock_mode or checkpoint_path is None or "mock_checkpoint" in str(checkpoint_path)
        probe_runner = ProbeRunner(
            checkpoint_path=checkpoint_path,
            use_mock=use_mock_probe,
        )
        probe_results = probe_runner.run_probes(stage_id=s.current_stage)

        if self._tiered_eval is None:
            logger.warning("[Eval] TieredEvaluator not available. Returning mock eval result.")
            return {
                "stage_id": s.current_stage,
                "mean_overall_score": 0.72,
                "escalated_to_teacher": False,
                "failure_mode_report": None,
            }

        # stage_config imports torch transitively — guard for CPU-only hosts
        stage_cfg = None
        try:
            from trainer.curriculum.stage_config import STAGES
            stage_cfg = STAGES.get(s.current_stage)
        except (ImportError, ModuleNotFoundError):
            pass
        objectives = stage_cfg.objectives if stage_cfg else ["fluency", "coherence", "syntax"]

        result = self._tiered_eval.evaluate_stage(
            stage_id=s.current_stage,
            stage_name=stage_cfg.name if stage_cfg else f"Stage {s.current_stage}",
            objectives=objectives,
            probe_outputs=probe_results,
        )

        logger.info(
            f"[Eval] Stage {s.current_stage} | Mean Score: {result.get('mean_overall_score', 0):.3f} "
            f"| Escalated: {result.get('escalated_to_teacher', False)}"
        )
        return result

    def _step_generate(self, eval_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3 — GENERATING: Commission synthetic data for detected failure modes.
        Returns dict with samples_generated count and batch.
        """
        s = self.state
        logger.info(
            f"\n{'='*60}\n"
            f"  [STEP 3/5 — GENERATING]  Stage {s.current_stage}  |  Iter {s.current_iteration}\n"
            f"{'='*60}"
        )

        failure_report_dict = eval_result.get("failure_mode_report")
        if not failure_report_dict:
            logger.info("[Generate] No failure mode report — skipping data generation.")
            return {"samples_generated": 0, "batch": None}

        # Build commission requests into TargetedPromptBatch
        if "TargetedPromptWriter" not in self._components:
            logger.warning("[Generate] TargetedPromptWriter not available. Using mock generation.")
            return {"samples_generated": 5, "batch": None}  # mock

        if "LocalDataGeneratorEngine" not in self._components:
            logger.warning("[Generate] LocalDataGeneratorEngine not available. Using mock generation.")
            return {"samples_generated": 5, "batch": None}

        try:
            from eval.failure_parser import FailureModeReport
            report = FailureModeReport(**failure_report_dict)
            prompt_writer = self._components["TargetedPromptWriter"]()
            prompt_batch = prompt_writer.build_batch_from_report(
                report=report,
                default_samples_per_request=5,
            )


            if not prompt_batch.prompt_specs:
                logger.info("[Generate] No commissionable failure modes found. Skipping generation.")
                return {"samples_generated": 0, "batch": None}

            # Auto-detect backend from env: prefer ollama if GENERATOR_API_BASE is set
            _api_base = os.getenv("GENERATOR_API_BASE", "")
            _backend = "ollama" if "11434" in _api_base or "ollama" in _api_base.lower() else "mock"
            _endpoint = _api_base.rstrip("/v1").rstrip("/") if _api_base else "http://127.0.0.1:11434"

            engine = self._components["LocalDataGeneratorEngine"](
                backend=_backend,
                endpoint_url=_endpoint,
            )
            batch = engine.generate_batch_from_prompts(prompt_batch, max_samples_per_spec=5)
            logger.info(f"[Generate] Generated {batch.total_samples} synthetic samples for Stage {s.current_stage} via '{_backend}'.")
            return {"samples_generated": batch.total_samples, "batch": batch}

        except Exception as e:
            s.log_error(f"Generation step failed: {e}")
            logger.error(f"[Generate] Error: {e}")
            return {"samples_generated": 0, "batch": None}

    def _step_merge(self, gen_result: Dict[str, Any]) -> int:
        """
        Step 4 — MERGING: Validate and merge synthetic samples into training dataset.
        Returns number of net-new examples merged.
        """
        s = self.state
        logger.info(
            f"\n{'='*60}\n"
            f"  [STEP 4/5 — MERGING]  Stage {s.current_stage}  |  Iter {s.current_iteration}\n"
            f"{'='*60}"
        )

        batch = gen_result.get("batch")
        if batch is None:
            logger.info("[Merge] No batch to merge.")
            return 0

        if "SyntheticDataValidator" not in self._components or "DatasetMerger" not in self._components:
            logger.warning("[Merge] Validator/Merger not available. Skipping merge.")
            return 0

        try:
            validator = self._components["SyntheticDataValidator"]()
            _, valid_examples = validator.validate_batch(batch)

            merger = self._components["DatasetMerger"]()
            added = merger.merge_into_stage_dataset(
                validated_examples=valid_examples,
                stage_id=s.current_stage,
                data_dir=s.data_dir,
            )
            logger.info(f"[Merge] Stage {s.current_stage}: {added} net-new examples merged.")
            return added

        except Exception as e:
            s.log_error(f"Merge step failed: {e}")
            logger.error(f"[Merge] Error: {e}")
            return 0

    def _step_advance_or_retrain(
        self,
        eval_result: Dict[str, Any],
        checkpoint_path: Optional[str],
        samples_generated: int,
        samples_merged: int,
    ) -> bool:
        """
        Step 5 — ADVANCING: Decide whether to advance stage or retrain.
        Returns True if advancing to next stage (or completing pipeline).
        """
        s = self.state
        mean_score = eval_result.get("mean_overall_score", 0.0)
        failure_report = eval_result.get("failure_mode_report")

        # Determine if HIGH/CRITICAL failures are present
        has_blocking_failures = False
        failure_modes_found: List[str] = []
        commission_count = 0
        overall_severity = "low"

        if failure_report:
            try:
                from eval.failure_parser import FailureModeReport, Severity
                report = FailureModeReport(**failure_report)
                overall_severity = str(report.overall_severity)
                commission_count = len(report.commission_requests)
                failure_modes_found = [str(fm.category) for fm in report.failure_modes]
                blocking = {Severity.HIGH, Severity.CRITICAL}
                has_blocking_failures = any(fm.severity in blocking for fm in report.failure_modes)
            except (ImportError, ModuleNotFoundError):
                logger.warning("[Advance] failure_parser not available — skipping blocking check.")
            except Exception:
                pass

        # Advancement decision
        score_ok = mean_score >= s.stage_advance_threshold
        advance = score_ok and not has_blocking_failures

        # Build StageResult
        result = StageResult(
            stage_id=s.current_stage,
            iteration=s.current_iteration,
            state=LoopState.ADVANCING if advance else LoopState.TRAINING,
            mean_eval_score=round(mean_score, 4),
            overall_severity=overall_severity,
            advance_to_next_stage=advance,
            failure_modes_found=failure_modes_found,
            commission_requests=commission_count,
            samples_generated=samples_generated,
            samples_validated=samples_merged,
            samples_merged=samples_merged,
            checkpoint_path=checkpoint_path,
        )
        s.record_result(result)

        if advance:
            logger.info(
                f"\n{'='*60}\n"
                f"  [STEP 5/5 — ADVANCING]  "
                f"Stage {s.current_stage} -> Stage {s.current_stage + 1}\n"
                f"  Score: {mean_score:.3f} >= {s.stage_advance_threshold} | No blocking failures\n"
                f"{'='*60}"
            )
            s.current_stage += 1
            s.current_iteration = 1
            if s.current_stage > s.total_stages:
                s.transition(LoopState.COMPLETE)
                logger.info("  *** ALL STAGES COMPLETE — CIRCLE loop finished. ***")
            else:
                s.transition(LoopState.IDLE)
        else:
            # Check iteration ceiling
            if s.current_iteration >= s.max_iterations_per_stage:
                logger.warning(
                    f"[Advance] Stage {s.current_stage} hit max iterations "
                    f"({s.max_iterations_per_stage}). Advancing anyway to prevent infinite loop."
                )
                s.current_stage += 1
                s.current_iteration = 1
                if s.current_stage > s.total_stages:
                    s.transition(LoopState.COMPLETE)
                else:
                    s.transition(LoopState.IDLE)
                return True  # forced advance

            logger.info(
                f"\n{'='*60}\n"
                f"  [STEP 5/5 — RETRAIN]  "
                f"Stage {s.current_stage} needs more iterations.\n"
                f"  Score: {mean_score:.3f} < {s.stage_advance_threshold} "
                f"| Blocking: {has_blocking_failures}\n"
                f"{'='*60}"
            )
            s.current_iteration += 1
            s.transition(LoopState.IDLE)

        self.state_manager.save(s)
        return advance

    # ─────────────────────────────────────────────────────────────
    # Public API: execute_loop
    # ─────────────────────────────────────────────────────────────

    def execute_loop(self) -> PipelineState:
        """
        Run the full CIRCLE closed loop until completion or FAILED state.

        Returns:
            PipelineState: the final state after loop exits.
        """
        s = self.state
        logger.info(
            f"\n{'#'*60}\n"
            f"  CIRCLE Closed-Loop Orchestrator — RUN: {self.run_id}\n"
            f"  Start Stage: {s.current_stage}  |  Max Iters/Stage: {s.max_iterations_per_stage}\n"
            f"  Threshold: {s.stage_advance_threshold}  |  Mock Mode: {self.mock_mode}\n"
            f"{'#'*60}\n"
        )

        while not s.is_complete and not s.is_failed:
            logger.info(
                f"\n>>> Stage {s.current_stage}/{s.total_stages}  "
                f"|  Iteration {s.current_iteration}/{s.max_iterations_per_stage}  "
                f"|  State: {s.current_state}\n"
            )

            # ── Step 1: TRAINING ─────────────────────────────────
            s.transition(LoopState.TRAINING)
            self.state_manager.save(s)
            checkpoint_path = self._step_train()

            # ── Step 2: EVALUATING ───────────────────────────────
            s.transition(LoopState.EVALUATING)
            self.state_manager.save(s)
            eval_result = self._step_evaluate(checkpoint_path)

            # ── Step 3: GENERATING ───────────────────────────────
            s.transition(LoopState.GENERATING)
            self.state_manager.save(s)
            gen_result = self._step_generate(eval_result)

            # ── Step 4: MERGING ──────────────────────────────────
            s.transition(LoopState.MERGING)
            self.state_manager.save(s)
            merged = self._step_merge(gen_result)

            # ── Step 5: ADVANCING ────────────────────────────────
            s.transition(LoopState.ADVANCING)
            self.state_manager.save(s)
            self._step_advance_or_retrain(
                eval_result=eval_result,
                checkpoint_path=checkpoint_path,
                samples_generated=gen_result.get("samples_generated", 0),
                samples_merged=merged,
            )

        # Final state save
        self.state_manager.save(s)
        self._print_summary()
        return s

    def _print_summary(self) -> None:
        """Prints a structured summary of the completed run."""
        s = self.state
        logger.info(
            f"\n{'#'*60}\n"
            f"  CIRCLE Loop Complete — Run: {s.run_id}\n"
            f"  Final State: {s.current_state}\n"
            f"  Stages Completed: {s.stages_completed}/{s.total_stages}\n"
            f"  Total Iterations: {sum(r.iteration for r in s.stage_results)}\n"
            f"  Error Count: {len(s.error_log)}\n"
            f"{'#'*60}\n"
        )
        for i, r in enumerate(s.stage_results, 1):
            logger.info(
                f"  [{i:02d}] Stage {r.stage_id} Iter {r.iteration}: "
                f"score={r.mean_eval_score:.3f} | "
                f"severity={r.overall_severity} | "
                f"advance={r.advance_to_next_stage} | "
                f"merged={r.samples_merged}"
            )


# ─────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(description="CIRCLE Master Closed-Loop Orchestrator")
    parser.add_argument("--start-stage",  type=int,   default=1,    help="Starting curriculum stage (1-5)")
    parser.add_argument("--max-iters",    type=int,   default=3,    help="Max loop iterations per stage")
    parser.add_argument("--threshold",    type=float, default=0.85, help="Stage advance score threshold")
    parser.add_argument("--data-dir",     type=str,   default="./data")
    parser.add_argument("--checkpoint-dir", type=str, default="./trainer/checkpoints")
    parser.add_argument("--log-dir",      type=str,   default="./logs")
    parser.add_argument("--run-id",       type=str,   default=None, help="Resume a specific run ID")
    parser.add_argument("--resume",       action="store_true", help="Resume from saved state")
    parser.add_argument("--mock",         action="store_true", default=True,
                        help="Run in mock mode (no GPU required)")
    parser.add_argument("--no-mock",      dest="mock", action="store_false",
                        help="Run with real pipeline (requires GPU + API keys)")
    args = parser.parse_args()

    orchestrator = ClosedLoopOrchestrator(
        start_stage=args.start_stage,
        max_iterations=args.max_iters,
        stage_advance_threshold=args.threshold,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        run_id=args.run_id,
        resume=args.resume,
        mock_mode=args.mock,
    )
    final_state = orchestrator.execute_loop()
    sys.exit(0 if not final_state.is_failed else 1)


if __name__ == "__main__":
    main()
