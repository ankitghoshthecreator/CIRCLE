"""
orchestrator/state_machine.py
CIRCLE Part 13: Closed-Loop State Machine & State Persistence

Defines the pipeline state model, loop state enum, and disk-backed
persistence layer used by the ClosedLoopOrchestrator to survive
container restarts and enable resume-on-failure.

State Transition Flow:
    IDLE
     └→ TRAINING       (train_stage called)
         └→ EVALUATING  (evaluation triggered after training)
             ├→ GENERATING  (failures detected → commission data)
             │    └→ MERGING  (validated samples merged into dataset)
             │         └→ TRAINING  (retrain on enriched dataset)
             └→ ADVANCING   (no HIGH/CRITICAL failures → next stage)
                  ├→ TRAINING  (start next stage)
                  └→ COMPLETE  (all 5 stages passed)
"""

import os
import json
import logging
from enum import Enum
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# State Enum
# ─────────────────────────────────────────────────────────────────

class LoopState(str, Enum):
    IDLE       = "idle"        # Not yet started / between runs
    TRAINING   = "training"    # QLoRA fine-tuning in progress
    EVALUATING = "evaluating"  # Student/teacher eval in progress
    GENERATING = "generating"  # Synthetic data commissioning
    MERGING    = "merging"     # Validated data being merged
    ADVANCING  = "advancing"   # Stage advancement decision
    COMPLETE   = "complete"    # All stages finished
    FAILED     = "failed"      # Unrecoverable error — human review needed


# ─────────────────────────────────────────────────────────────────
# Stage Result Model
# ─────────────────────────────────────────────────────────────────

class StageResult(BaseModel):
    stage_id: int
    iteration: int
    state: str                              # LoopState value at completion
    mean_eval_score: float = 0.0
    overall_severity: str = "low"
    advance_to_next_stage: bool = False
    failure_modes_found: List[str] = Field(default_factory=list)
    commission_requests: int = 0
    samples_generated: int = 0
    samples_validated: int = 0
    samples_merged: int = 0
    checkpoint_path: Optional[str] = None
    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    model_config = {"use_enum_values": True}


# ─────────────────────────────────────────────────────────────────
# Pipeline State Model  (the persisted snapshot)
# ─────────────────────────────────────────────────────────────────

class PipelineState(BaseModel):
    """
    Full serializable state of the CIRCLE closed loop.
    Persisted to disk after every state transition so the orchestrator
    can be killed and resumed without losing progress.
    """
    # ── Identity ─────────────────────────────────────────────────
    run_id: str = Field(..., description="Unique identifier for this pipeline run")

    # ── Progress ─────────────────────────────────────────────────
    current_stage: int = Field(1, ge=1, le=100)
    current_iteration: int = Field(1, ge=1)
    max_iterations_per_stage: int = Field(3, ge=1)
    total_stages: int = Field(5, ge=1)

    current_state: str = LoopState.IDLE        # LoopState value

    # ── Advancement Threshold ─────────────────────────────────────
    # Mean eval score above this AND no HIGH/CRITICAL failures → advance
    stage_advance_threshold: float = Field(0.85, ge=0.0, le=1.0)

    # ── History ──────────────────────────────────────────────────
    stage_results: List[StageResult] = Field(default_factory=list)
    error_log: List[str] = Field(default_factory=list)

    # ── Timestamps ───────────────────────────────────────────────
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── Paths ─────────────────────────────────────────────────────
    data_dir: str = "./data"
    checkpoint_dir: str = "./trainer/checkpoints"
    log_dir: str = "./logs"

    model_config = {"use_enum_values": True}

    def transition(self, new_state: LoopState) -> None:
        """Transition to a new state and stamp the update time."""
        old = self.current_state
        self.current_state = new_state
        self.last_updated_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"[StateMachine] Stage {self.current_stage} | Iter {self.current_iteration} "
                    f"| {old} -> {new_state}")

    def record_result(self, result: StageResult) -> None:
        """Append a StageResult and update timestamp."""
        self.stage_results.append(result)
        self.last_updated_at = datetime.now(timezone.utc).isoformat()

    def log_error(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self.error_log.append(f"[{ts}] {msg}")
        logger.error(f"[StateMachine] ERROR: {msg}")

    @property
    def is_complete(self) -> bool:
        return self.current_state == LoopState.COMPLETE

    @property
    def is_failed(self) -> bool:
        return self.current_state == LoopState.FAILED

    @property
    def stages_completed(self) -> int:
        """Count of distinct stage IDs that advanced."""
        advanced = {r.stage_id for r in self.stage_results if r.advance_to_next_stage}
        return len(advanced)


# ─────────────────────────────────────────────────────────────────
# State Persistence Manager
# ─────────────────────────────────────────────────────────────────

class StatePersistenceManager:
    """
    Saves and loads PipelineState to/from a JSON file on disk.
    Called after every state transition so the loop is restartable.
    """

    def __init__(self, state_dir: str = "./logs", run_id: Optional[str] = None):
        self.state_dir = state_dir
        self.run_id = run_id
        os.makedirs(state_dir, exist_ok=True)

    def _state_path(self, run_id: str) -> str:
        return os.path.join(self.state_dir, f"pipeline_state_{run_id}.json")

    def save(self, state: PipelineState) -> str:
        """Persist state to disk. Returns the file path written."""
        path = self._state_path(state.run_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(), f, indent=2)
        logger.debug(f"[StateManager] State persisted -> {path}")
        return path

    def load(self, run_id: str) -> Optional[PipelineState]:
        """Load a previously persisted PipelineState. Returns None if not found."""
        path = self._state_path(run_id)
        if not os.path.exists(path):
            logger.info(f"[StateManager] No existing state found at '{path}'. Starting fresh.")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = PipelineState(**data)
            logger.info(
                f"[StateManager] Resumed run '{run_id}' — "
                f"Stage {state.current_stage}, Iter {state.current_iteration}, "
                f"State: {state.current_state}"
            )
            return state
        except Exception as e:
            logger.error(f"[StateManager] Failed to load state from '{path}': {e}")
            return None

    def list_runs(self) -> List[str]:
        """Returns all run_ids found in the state directory."""
        runs = []
        for fname in os.listdir(self.state_dir):
            if fname.startswith("pipeline_state_") and fname.endswith(".json"):
                run_id = fname.replace("pipeline_state_", "").replace(".json", "")
                runs.append(run_id)
        return sorted(runs)

    def delete(self, run_id: str) -> bool:
        """Delete a saved state file. Returns True if deleted."""
        path = self._state_path(run_id)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"[StateManager] Deleted state for run '{run_id}'.")
            return True
        return False
