"""
scripts/rl_monitor.py
CIRCLE Part 17: Live RL Training Monitor

Reads the JSONL log files produced by rl_loop.py and renders a
live terminal dashboard showing:

  Stage | RL-Step | Reward | Grade | Samples Added | Loss | Status

Usage:
    # Watch a specific run (auto-refreshes every 5s)
    python scripts/rl_monitor.py --run-id rl_20260922_181234_abc123

    # Show latest run automatically
    python scripts/rl_monitor.py

    # One-shot (no refresh)
    python scripts/rl_monitor.py --once
"""

import os
import sys
import json
import time
import glob
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ANSI color codes
RESET   = "\033[0m"
BOLD    = "\033[1m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
GREEN   = "\033[92m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
WHITE   = "\033[97m"
DIM     = "\033[2m"

GRADE_COLOR = {
    "pass":    GREEN,
    "improve": YELLOW,
    "fail":    RED,
}

STATUS_COLOR = {
    "→ ADVANCE": GREEN,
    "  RETRAIN": YELLOW,
    "  FORCED ":  RED,
}


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _find_latest_run(log_dir: str) -> Optional[str]:
    """Find the most recently modified JSONL log file."""
    pattern = os.path.join(log_dir, "rl_*.jsonl")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _load_records(log_path: str) -> List[Dict[str, Any]]:
    """Load all RL step records from a JSONL log file."""
    records = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return records


def _grade_bar(reward: float, width: int = 20) -> str:
    """Render a simple ASCII progress bar for the reward score."""
    filled = int(reward * width)
    bar = "█" * filled + "░" * (width - filled)
    color = GREEN if reward >= 0.85 else (YELLOW if reward >= 0.60 else RED)
    return f"{color}{bar}{RESET}"


def _render_dashboard(log_path: str, records: List[Dict[str, Any]]) -> None:
    """Render the full dashboard to terminal."""
    run_id = records[0]["run_id"] if records else "unknown"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = (
        f"{BOLD}{CYAN}"
        f"  ╔══════════════════════════════════════════════════════════════╗\n"
        f"  ║         CIRCLE RLAIF TRAINING MONITOR                      ║\n"
        f"  ╚══════════════════════════════════════════════════════════════╝{RESET}\n"
        f"  {DIM}Run ID  : {run_id}{RESET}\n"
        f"  {DIM}Log     : {os.path.basename(log_path)}{RESET}\n"
        f"  {DIM}Updated : {now}{RESET}\n"
    )
    print(header)

    if not records:
        print(f"  {YELLOW}Waiting for RL loop to start writing logs...{RESET}\n")
        return

    # Group by stage
    stages: Dict[int, List[Dict]] = {}
    for r in records:
        sid = r.get("stage_id", 0)
        stages.setdefault(sid, []).append(r)

    col_w = [7, 8, 8, 10, 20, 10, 8, 11]
    headers = ["Stage", "RL-Step", "Reward", "Grade", "Reward Bar", "Samples", "Loss", "Status"]
    col_fmt = "  " + "  ".join(f"{h:>{w}}" for h, w in zip(headers, col_w))
    print(f"{BOLD}{WHITE}{col_fmt}{RESET}")
    print(f"  {'─' * (sum(col_w) + len(col_w) * 2 + 2)}")

    for stage_id in sorted(stages.keys()):
        stage_records = stages[stage_id]
        for r in stage_records:
            reward = r.get("reward", 0.0)
            grade  = r.get("grade", "?")
            loss   = r.get("training_loss")
            advance= r.get("advance", False)
            merged = r.get("samples_merged", 0)

            grade_c  = GRADE_COLOR.get(grade, WHITE)
            status   = "→ ADVANCE" if advance else "  RETRAIN"
            status_c = GREEN if advance else YELLOW

            loss_str = f"{loss:.4f}" if loss is not None else "   N/A"
            bar = _grade_bar(reward)

            row = (
                f"  "
                f"{r.get('stage_id', '?'):>{col_w[0]}}  "
                f"{r.get('rl_step', '?'):>{col_w[1]}}  "
                f"{reward:>{col_w[2]}.3f}  "
                f"{grade_c}{grade:>{col_w[3]}}{RESET}  "
                f"{bar}  "  # bar replaces col_w[4] visually
                f"{merged:>{col_w[5]}}  "
                f"{loss_str:>{col_w[6]}}  "
                f"{status_c}{status:>{col_w[7]}}{RESET}"
            )
            print(row)

            # Show failure modes as sub-row
            failure_modes = r.get("failure_modes", [])
            if failure_modes:
                fm_str = ", ".join(failure_modes[:4])
                if len(failure_modes) > 4:
                    fm_str += f" (+{len(failure_modes)-4})"
                print(f"  {DIM}{'':>{col_w[0]}}  {'Failures:':>{col_w[1]}}  {fm_str}{RESET}")

        # Stage separator
        print(f"  {'─' * (sum(col_w) + len(col_w) * 2 + 2)}")

    # Summary stats
    total_steps = len(records)
    total_merged = sum(r.get("samples_merged", 0) for r in records)
    avg_reward = sum(r.get("reward", 0) for r in records) / max(1, total_steps)
    passes = sum(1 for r in records if r.get("grade") == "pass")

    print(f"\n  {BOLD}Summary:{RESET}")
    print(f"  Total RL Steps  : {total_steps}")
    print(f"  Avg Reward      : {avg_reward:.3f}")
    print(f"  Stages Passed   : {passes} / {len(stages)}")
    print(f"  Samples Merged  : {total_merged}")
    print()


def main():
    parser = argparse.ArgumentParser(description="CIRCLE RLAIF Training Monitor")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Specific run ID to monitor (auto-detects latest if not set)")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Directory containing RL JSONL logs")
    parser.add_argument("--refresh", type=int, default=5,
                        help="Refresh interval in seconds (default 5)")
    parser.add_argument("--once", action="store_true",
                        help="Print once and exit (no live refresh)")
    args = parser.parse_args()

    log_dir = args.log_dir

    # Resolve log file
    if args.run_id:
        log_path = os.path.join(log_dir, f"{args.run_id}.jsonl")
    else:
        log_path = _find_latest_run(log_dir)
        if not log_path:
            print(f"No rl_*.jsonl files found in '{log_dir}'. Start rl_loop.py first.")
            sys.exit(1)

    print(f"Monitoring: {log_path}")
    if not args.once:
        print(f"Refreshing every {args.refresh}s. Press Ctrl+C to exit.\n")

    try:
        while True:
            _clear()
            records = _load_records(log_path)
            _render_dashboard(log_path, records)
            if args.once:
                break
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
