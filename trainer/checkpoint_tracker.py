"""
trainer/checkpoint_tracker.py
CIRCLE Checkpoint & Overfitting Tracker

Features:
  1. Periodic Checkpoint Saving: Every 100th step, saves a checkpoint (e.g. checkpoint_step_100).
  2. Best Loss Checkpoint Replacement: Overwrites 'checkpoint_best' whenever a new lowest loss is achieved.
  3. Step-by-Step CSV Logging: Maintains 'loss_log.csv' with columns (step, loss) to monitor overfitting.
"""

import os
import csv
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("CheckpointTracker")


class CheckpointTracker:
    """
    Manages model checkpointing and loss logging:
      - Appends (step, loss) to loss_log.csv on every step.
      - Saves checkpoint_step_N every `save_every_n_steps` (default 100).
      - Saves/replaces `checkpoint_best` whenever current loss < best_loss.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./trainer/checkpoints",
        save_every_n_steps: int = 100,
        csv_filename: str = "loss_log.csv",
        resume: bool = True,
    ):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.save_every_n_steps = save_every_n_steps
        self.csv_filename = csv_filename
        self.csv_path = os.path.join(self.checkpoint_dir, csv_filename)
        self.best_checkpoint_dir = os.path.join(self.checkpoint_dir, "checkpoint_best")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.global_step = 0
        self.best_loss = float("inf")
        self.best_step = None

        if resume and os.path.exists(self.csv_path):
            self._load_existing_history()
        else:
            self._init_csv()

    def _init_csv(self) -> None:
        """Create CSV file with header (step, loss)."""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])

    def _load_existing_history(self) -> None:
        """Parse existing CSV to restore global_step and best_loss so far."""
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            s = int(row[0])
                            l = float(row[1])
                            self.global_step = max(self.global_step, s)
                            if l < self.best_loss:
                                self.best_loss = l
                                self.best_step = s
                        except ValueError:
                            continue
            logger.info(
                f"[CheckpointTracker] Resumed history: last step={self.global_step}, "
                f"best_loss={self.best_loss:.4f} at step {self.best_step}"
            )
        except Exception as e:
            logger.warning(f"[CheckpointTracker] Could not parse CSV history: {e}")
            self._init_csv()

    def record_step(
        self,
        loss: float,
        model: Any = None,
        tokenizer: Any = None,
        custom_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Record a single training step:
          1. Log (step, loss) to CSV file.
          2. Save periodic checkpoint if step % save_every_n_steps == 0.
          3. Save/replace best loss checkpoint if loss < best_loss.
        """
        if custom_step is not None:
            self.global_step = custom_step
        else:
            self.global_step += 1

        current_step = self.global_step
        current_loss = float(loss)

        # 1. Log (step, loss) to CSV for overfitting monitoring
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([current_step, round(current_loss, 6)])

        saved_periodic = False
        saved_best = False

        # 2. Every 100th step periodic checkpoint
        if current_step > 0 and current_step % self.save_every_n_steps == 0:
            periodic_dir = os.path.join(
                self.checkpoint_dir, f"checkpoint_step_{current_step}"
            )
            self._save_checkpoint(periodic_dir, model, tokenizer, step=current_step, loss=current_loss)
            logger.info(
                f"💾 Periodic Checkpoint Saved at step {current_step} → '{periodic_dir}'"
            )
            saved_periodic = True

        # 3. Best loss checkpoint comparison & replacement
        if current_loss < self.best_loss:
            prev_best = self.best_loss
            prev_step = self.best_step
            self.best_loss = current_loss
            self.best_step = current_step

            self._save_checkpoint(
                self.best_checkpoint_dir,
                model,
                tokenizer,
                step=current_step,
                loss=current_loss,
            )

            if prev_step is not None:
                logger.info(
                    f"🏆 New Best Checkpoint Saved! Step {current_step} (loss={current_loss:.4f} "
                    f"< prev best {prev_best:.4f} at step {prev_step}) → '{self.best_checkpoint_dir}'"
                )
            else:
                logger.info(
                    f"🏆 Initial Best Checkpoint Saved at Step {current_step} (loss={current_loss:.4f}) → '{self.best_checkpoint_dir}'"
                )
            saved_best = True

        return {
            "step": current_step,
            "loss": current_loss,
            "saved_periodic": saved_periodic,
            "saved_best": saved_best,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "csv_path": self.csv_path,
        }

    def _save_checkpoint(
        self,
        target_dir: str,
        model: Any = None,
        tokenizer: Any = None,
        step: int = 0,
        loss: float = 0.0,
    ) -> None:
        """Save model/tokenizer weights and metadata info to target directory."""
        os.makedirs(target_dir, exist_ok=True)

        if model is not None:
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(target_dir)
            elif hasattr(model, "state_dict"):
                import torch
                torch.save(model.state_dict(), os.path.join(target_dir, "pytorch_model.bin"))

        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(target_dir)

        # Write metadata JSON for tracking
        metadata_path = os.path.join(target_dir, "checkpoint_info.json")
        info = {
            "step": step,
            "loss": loss,
            "best_loss_so_far": self.best_loss,
            "best_step": self.best_step,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
