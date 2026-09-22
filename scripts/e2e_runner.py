"""
scripts/e2e_runner.py
CIRCLE Part 16: End-to-End System Integration & Training Execution Runner

Runs the complete closed-loop master orchestrator cycle across all 5 curriculum stages:
- Train → Evaluate → Generate → Merge → Advance
- Generates execution state persistence JSON and pipeline summary report.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from orchestrator.loop_controller import ClosedLoopOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("E2ERunner")


def run_e2e_pipeline(
    start_stage: int = 1,
    max_iterations: int = 1,
    stage_threshold: float = 0.85,
    mock_mode: bool = True,
    log_dir: str = "./logs"
) -> Dict[str, Any]:
    """Executes full end-to-end CIRCLE pipeline across all stages."""
    logger.info("=== CIRCLE End-to-End Pipeline Execution Started ===")

    orchestrator = ClosedLoopOrchestrator(
        start_stage=start_stage,
        max_iterations=max_iterations,
        stage_advance_threshold=stage_threshold,
        data_dir=os.path.join(PROJECT_ROOT, "data"),
        checkpoint_dir=os.path.join(PROJECT_ROOT, "trainer", "checkpoints"),
        log_dir=os.path.join(PROJECT_ROOT, log_dir),
        mock_mode=mock_mode
    )

    final_state = orchestrator.execute_loop()

    summary = {
        "run_id": final_state.run_id,
        "final_state": str(final_state.current_state),
        "stages_completed": final_state.stages_completed,
        "total_stages": final_state.total_stages,
        "is_complete": final_state.is_complete,
        "error_count": len(final_state.error_log),
        "results_by_stage": [r.model_dump() for r in final_state.stage_results]
    }

    summary_path = os.path.join(PROJECT_ROOT, log_dir, f"e2e_summary_{final_state.run_id}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"=== CIRCLE E2E Pipeline Finished. Summary written to '{summary_path}' ===")
    return summary


def main():
    parser = argparse.ArgumentParser(description="CIRCLE End-to-End Pipeline Runner")
    parser.add_argument("--start-stage", type=int, default=1, help="Starting curriculum stage ID")
    parser.add_argument("--max-iters", type=int, default=1, help="Max iterations per stage")
    parser.add_argument("--threshold", type=float, default=0.85, help="Stage advance score threshold")
    parser.add_argument("--mock", action="store_true", default=True, help="Run in mock/test mode")
    parser.add_argument("--no-mock", dest="mock", action="store_false", help="Run with live GPU/Ollama")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log and state directory")
    args = parser.parse_args()

    summary = run_e2e_pipeline(
        start_stage=args.start_stage,
        max_iterations=args.max_iters,
        stage_threshold=args.threshold,
        mock_mode=args.mock,
        log_dir=args.log_dir
    )

    print(f"\n=================================================")
    print(f"  CIRCLE End-to-End Pipeline Execution Summary")
    print(f"=================================================")
    print(f" Run ID           : {summary['run_id']}")
    print(f" Final State      : {summary['final_state']}")
    print(f" Stages Completed : {summary['stages_completed']} / {summary['total_stages']}")
    print(f" Pipeline Complete: {'YES' if summary['is_complete'] else 'NO'}")
    print(f" Total Errors     : {summary['error_count']}")
    print(f"=================================================\n")


if __name__ == "__main__":
    main()
