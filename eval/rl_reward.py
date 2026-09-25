"""
eval/rl_reward.py
CIRCLE Part 17: RL Reward Signal — Groq Critique → Scalar Reward

Converts the raw qualitative Groq critique text into a structured, numeric
reward signal (0.0 – 1.0) that drives the RLAIF loop.

Reward model:
  base_reward = 1.0
  For each detected failure mode, subtract a penalty:
      CRITICAL  → -0.40
      HIGH      → -0.25
      MEDIUM    → -0.12
      LOW       → -0.04
  Floor at 0.05 (model always gets some credit for producing text).

The reward is also categorised:
  >= 0.85  → "pass"    (stage can advance)
  >= 0.60  → "improve" (needs corrective data, not yet blocked)
  <  0.60  → "fail"    (must generate corrective data + retrain)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from eval.failure_parser import (
    FailureModeParser,
    FailureModeReport,
    FailureMode,
    Severity,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Penalty map  (per detected failure mode)
# ──────────────────────────────────────────────────────────────

SEVERITY_PENALTY: Dict[str, float] = {
    Severity.CRITICAL: 0.40,
    Severity.HIGH:     0.25,
    Severity.MEDIUM:   0.12,
    Severity.LOW:      0.04,
}


# ──────────────────────────────────────────────────────────────
# Reward data container
# ──────────────────────────────────────────────────────────────

@dataclass
class RLReward:
    """Complete reward signal for one RL step."""

    reward: float                               # 0.0 – 1.0
    grade: str                                  # "pass" | "improve" | "fail"
    failure_report: Optional[FailureModeReport] # parsed structured failures
    penalties_applied: List[Dict[str, Any]] = field(default_factory=list)
    critique_raw: str = ""

    # Convenience
    @property
    def should_generate(self) -> bool:
        """True when corrective synthetic data should be commissioned."""
        return self.grade in ("improve", "fail")

    @property
    def should_advance(self) -> bool:
        """True when the model has earned stage advancement."""
        return self.grade == "pass"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward": self.reward,
            "grade": self.grade,
            "should_generate": self.should_generate,
            "should_advance": self.should_advance,
            "failure_modes": (
                [fm.category for fm in self.failure_report.failure_modes]
                if self.failure_report else []
            ),
            "overall_severity": (
                str(self.failure_report.overall_severity)
                if self.failure_report else "low"
            ),
            "penalties": self.penalties_applied,
            "critique_raw_len": len(self.critique_raw),
        }


# ──────────────────────────────────────────────────────────────
# Reward calculator
# ──────────────────────────────────────────────────────────────

class RLRewardCalculator:
    """
    Converts a Groq critique string into a scalar reward via structured
    failure mode detection.

    Usage:
        calc = RLRewardCalculator(pass_threshold=0.85)
        reward = calc.compute(stage_id=1, stage_name="Base English",
                              critique_raw=groq_response_text)
        print(reward.reward, reward.grade)
    """

    def __init__(
        self,
        pass_threshold: float = 0.85,
        improve_threshold: float = 0.60,
        parser: Optional[FailureModeParser] = None,
    ):
        self.pass_threshold = pass_threshold
        self.improve_threshold = improve_threshold
        self._parser = parser or FailureModeParser()

    # ── Public API ────────────────────────────────────────────

    def compute(
        self,
        stage_id: int,
        stage_name: str,
        critique_raw: str,
    ) -> RLReward:
        """
        Parse the Groq critique and produce a numeric reward.

        Args:
            stage_id:     Curriculum stage number.
            stage_name:   Human-readable stage name.
            critique_raw: Raw text from GroqCriticAgent.analyze_stage_outputs().

        Returns:
            RLReward with reward score and structured metadata.
        """
        if not critique_raw or len(critique_raw.strip()) < 10:
            logger.warning("[RLReward] Critique too short — awarding full reward by default.")
            return RLReward(
                reward=1.0,
                grade="pass",
                failure_report=None,
                critique_raw=critique_raw,
            )

        report: FailureModeReport = self._parser.parse(stage_id, stage_name, critique_raw)

        reward, penalties = self._calculate_reward(report.failure_modes)

        grade = self._grade(reward)

        logger.info(
            f"[RLReward] Stage {stage_id} | reward={reward:.3f} | grade={grade} "
            f"| failures={len(report.failure_modes)} | severity={report.overall_severity}"
        )

        return RLReward(
            reward=reward,
            grade=grade,
            failure_report=report,
            penalties_applied=penalties,
            critique_raw=critique_raw,
        )

    def compute_from_student_score(
        self,
        mean_score: float,
        escalation_reasons: List[str],
    ) -> "RLReward":
        """
        Lightweight reward from the fast student evaluator score (no Groq).
        Used in mock mode or when Groq is unavailable.
        """
        grade = self._grade(mean_score)
        return RLReward(
            reward=mean_score,
            grade=grade,
            failure_report=None,
            critique_raw="[student_eval_only] " + "; ".join(escalation_reasons),
        )

    # ── Internal helpers ──────────────────────────────────────

    def _calculate_reward(
        self,
        failure_modes: List[FailureMode],
    ) -> tuple:
        """Subtract penalties from 1.0 for each detected failure mode."""
        base = 1.0
        penalties: List[Dict[str, Any]] = []

        for fm in failure_modes:
            sev_key = str(fm.severity)
            penalty = SEVERITY_PENALTY.get(sev_key, 0.04)
            base -= penalty
            penalties.append({
                "category":  str(fm.category),
                "severity":  sev_key,
                "penalty":   penalty,
            })

        reward = round(max(0.05, base), 4)
        return reward, penalties

    def _grade(self, reward: float) -> str:
        if reward >= self.pass_threshold:
            return "pass"
        if reward >= self.improve_threshold:
            return "improve"
        return "fail"


# ──────────────────────────────────────────────────────────────
# Convenience: reward from a full evaluate_stage() result dict
# ──────────────────────────────────────────────────────────────

def reward_from_eval_result(
    eval_result: dict,
    pass_threshold: float = 0.85,
) -> RLReward:
    """
    Build an RLReward from the dict returned by TieredEvaluator.evaluate_stage().

    Priority:
      1. If a teacher critique is present → use full RL reward.
      2. Otherwise → fall back to student mean score.
    """
    calc = RLRewardCalculator(pass_threshold=pass_threshold)

    # Path 1: Groq critique available
    teacher_critique = eval_result.get("teacher_critique", {})
    if teacher_critique:
        critique_raw = teacher_critique.get("critique_raw", "")
        stage_id = eval_result.get("stage_id", 1)
        stage_name = eval_result.get("stage_name", f"Stage {stage_id}")
        if critique_raw:
            return calc.compute(stage_id, stage_name, critique_raw)

    # Path 2: Student score only
    mean_score = eval_result.get("mean_overall_score", 0.5)
    student_report = eval_result.get("student_report", {})
    reasons = student_report.get("escalation_reasons", [])
    return calc.compute_from_student_score(mean_score, reasons)


# ──────────────────────────────────────────────────────────────
# CLI smoke test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    MOCK_CRITIQUE = """
    The model shows clear signs of degenerate repetition on longer continuations,
    repeatedly looping the same phrase. There is also significant speaker drift
    where the model switches from first-person to third-person mid-response.
    Vocabulary poverty is evident — the model uses only basic common words.
    Overall the output is partially coherent but cut off abruptly without a terminal
    punctuation mark, indicating an abrupt cutoff failure mode as well.
    """

    calc = RLRewardCalculator(pass_threshold=0.85, improve_threshold=0.60)
    reward = calc.compute(stage_id=1, stage_name="Base English", critique_raw=MOCK_CRITIQUE)

    print("\n=== RL Reward Signal ===")
    print(f"  Reward  : {reward.reward:.3f}")
    print(f"  Grade   : {reward.grade}")
    print(f"  Advance : {reward.should_advance}")
    print(f"  Generate: {reward.should_generate}")
    print(f"\n  Penalties Applied:")
    for p in reward.penalties_applied:
        print(f"    [{p['severity'].upper():8s}] {p['category']:<35s} -{p['penalty']:.2f}")
