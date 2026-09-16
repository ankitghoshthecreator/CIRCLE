"""
eval/distilled_eval.py
Part 8: Lightweight Student Distillation Evaluator

Provides fast, local, sub-5ms multi-dimensional quality scoring (fluency, syntax,
semantics, vocabulary) and risk prediction without external cloud API overhead.
Includes a TieredEvaluator that routes high-quality outputs locally and escalates
low-scoring or high-risk outputs to full 70B teacher critique (GroqCriticAgent).
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from eval.failure_parser import FailureModeParser, FailureModeReport

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Pydantic Models for Distilled Evaluation
# ──────────────────────────────────────────────────────────────

class DistilledScore(BaseModel):
    fluency_score: float = Field(..., ge=0.0, le=1.0, description="Fluency & repetition score")
    syntax_score: float = Field(..., ge=0.0, le=1.0, description="Syntax & grammar score")
    semantic_score: float = Field(..., ge=0.0, le=1.0, description="Semantic coherence & relevance score")
    vocabulary_score: float = Field(..., ge=0.0, le=1.0, description="Vocabulary richness & diversity score")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Weighted composite quality score")
    predicted_risk: str = Field(..., description="Predicted risk level: low, medium, high, critical")
    detected_flags: List[str] = Field(default_factory=list, description="Diagnostic flags identified by student")

    model_config = {"use_enum_values": True}


class DistilledEvalReport(BaseModel):
    stage_id: int
    stage_name: str
    total_probes: int
    mean_overall_score: float = Field(..., ge=0.0, le=1.0)
    probe_scores: List[DistilledScore] = Field(default_factory=list)
    escalation_required: bool = Field(False, description="True if quality < threshold or high risk detected")
    escalation_reasons: List[str] = Field(default_factory=list)

    model_config = {"use_enum_values": True}


# ──────────────────────────────────────────────────────────────
# Lightweight Student Evaluator (Fast Local Heuristics)
# ──────────────────────────────────────────────────────────────

class LightweightStudentEvaluator:
    """
    Sub-5ms local evaluator using n-gram statistics, lexical diversity (TTR),
    syntactic markers, and semantic overlap heuristics.
    """

    def evaluate_text(self, prompt: str, output: str) -> DistilledScore:
        flags: List[str] = []

        # Strip think blocks if present
        clean_text = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

        # Handle empty/extremely short text
        if len(clean_text) < 5:
            flags.append("empty_or_too_short")
            return DistilledScore(
                fluency_score=0.1,
                syntax_score=0.1,
                semantic_score=0.1,
                vocabulary_score=0.1,
                overall_score=0.1,
                predicted_risk="critical",
                detected_flags=flags
            )

        fluency, f_flags = self._score_fluency(clean_text)
        syntax, s_flags = self._score_syntax(clean_text)
        semantics, sem_flags = self._score_semantics(prompt, clean_text)
        vocab, v_flags = self._score_vocabulary(clean_text)

        flags.extend(f_flags + s_flags + sem_flags + v_flags)

        # Weighted overall composite
        overall = round(
            0.35 * fluency +
            0.25 * syntax +
            0.25 * semantics +
            0.15 * vocab,
            4
        )

        # Risk classification
        if overall < 0.40 or "severe_token_loop" in flags:
            risk = "critical"
        elif overall < 0.60 or "abrupt_cutoff" in flags or "high_repetition" in flags:
            risk = "high"
        elif overall < 0.75 or len(flags) > 0:
            risk = "medium"
        else:
            risk = "low"

        return DistilledScore(
            fluency_score=round(fluency, 4),
            syntax_score=round(syntax, 4),
            semantic_score=round(semantics, 4),
            vocabulary_score=round(vocab, 4),
            overall_score=overall,
            predicted_risk=risk,
            detected_flags=list(set(flags))
        )

    def _score_fluency(self, text: str) -> tuple:
        """Evaluates token loops, n-gram repetition, and truncation."""
        score = 1.0
        flags = []
        words = text.lower().split()

        if len(words) == 0:
            return 0.0, ["empty_output"]

        # Check 3-gram repetition loops
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        if trigrams:
            unique_trigrams = len(set(trigrams))
            trigram_ratio = unique_trigrams / len(trigrams)
            if trigram_ratio < 0.4:
                score -= 0.5
                flags.append("severe_token_loop")
            elif trigram_ratio < 0.7:
                score -= 0.3
                flags.append("high_repetition")

        # Check single word repetition
        if words:
            most_common_cnt = max(words.count(w) for w in set(words))
            if most_common_cnt / len(words) > 0.35 and len(words) > 5:
                score -= 0.3
                flags.append("word_overuse")

        # Check abrupt truncation (no terminal punctuation)
        if not re.search(r"[.!?\"']$", text.strip()):
            score -= 0.2
            flags.append("abrupt_cutoff")

        return max(0.1, score), flags

    def _score_syntax(self, text: str) -> tuple:
        """Evaluates sentence boundaries, run-ons, and basic grammar markers."""
        score = 1.0
        flags = []
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]

        # Extremely long sentences without punctuation (run-on)
        words = text.split()
        if len(sentences) == 1 and len(words) > 40:
            score -= 0.3
            flags.append("run_on_sentence")

        # Check comma splices (excessive commas relative to periods)
        comma_count = text.count(",")
        period_count = max(1, text.count("."))
        if comma_count / period_count > 6:
            score -= 0.25
            flags.append("excessive_commas")

        return max(0.1, score), flags

    def _score_semantics(self, prompt: str, output: str) -> tuple:
        """Evaluates prompt overlap relevance and circular restatement."""
        score = 1.0
        flags = []

        prompt_words = set(re.findall(r"\w+", prompt.lower()))
        output_words = set(re.findall(r"\w+", output.lower()))

        if not output_words:
            return 0.1, ["empty_semantics"]

        # Circular restatement check (output merely repeats prompt words without adding new tokens)
        jaccard = len(prompt_words & output_words) / max(1, len(prompt_words | output_words))
        new_words = output_words - prompt_words

        if len(output_words) > 5 and len(new_words) < 2 and jaccard > 0.6:
            score -= 0.4
            flags.append("semantic_circularity")

        # Off-topic / low overlap when prompt is long
        if len(prompt_words) > 6 and len(prompt_words & output_words) == 0:
            score -= 0.3
            flags.append("topic_drift")

        return max(0.1, score), flags

    def _score_vocabulary(self, text: str) -> tuple:
        """Evaluates Type-Token Ratio (TTR) and vocabulary richness."""
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.1, ["no_vocabulary"]

        ttr = len(set(words)) / len(words)
        score = min(1.0, ttr * 1.3)  # Scale TTR

        flags = []
        if ttr < 0.4 and len(words) > 10:
            flags.append("vocabulary_poverty")

        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3.5:
            score -= 0.15

        return max(0.1, score), flags

    def evaluate_batch(
        self,
        stage_id: int,
        stage_name: str,
        probe_results: List[Dict[str, str]],
        escalation_threshold: float = 0.70
    ) -> DistilledEvalReport:
        """Evaluates a batch of probe results locally."""
        scores: List[DistilledScore] = []
        escalation_reasons: List[str] = []

        for p in probe_results:
            prompt = p.get("prompt", "")
            output = p.get("output", "")
            score = self.evaluate_text(prompt, output)
            scores.append(score)

            if score.overall_score < escalation_threshold:
                escalation_reasons.append(f"Probe overall score {score.overall_score:.2f} < {escalation_threshold}")
            if score.predicted_risk in ("high", "critical"):
                escalation_reasons.append(f"Probe predicted risk '{score.predicted_risk}'")

        mean_score = sum(s.overall_score for s in scores) / max(1, len(scores))
        escalate = len(escalation_reasons) > 0 or mean_score < escalation_threshold

        return DistilledEvalReport(
            stage_id=stage_id,
            stage_name=stage_name,
            total_probes=len(probe_results),
            mean_overall_score=round(mean_score, 4),
            probe_scores=scores,
            escalation_required=escalate,
            escalation_reasons=list(set(escalation_reasons))
        )


# ──────────────────────────────────────────────────────────────
# Tiered Evaluator (Routing & Escalation Logic)
# ──────────────────────────────────────────────────────────────

class TieredEvaluator:
    """
    Tiered evaluation pipeline:
    1. Runs fast student evaluation locally (<5ms).
    2. If quality is high and risk is low, returns student evaluation report.
    3. If quality drops or high risk is detected, escalates to full 70B Groq critic + FailureModeParser.
    """

    def __init__(
        self,
        student_evaluator: Optional[LightweightStudentEvaluator] = None,
        teacher_agent = None,
        escalation_threshold: float = 0.70
    ):
        self.student = student_evaluator or LightweightStudentEvaluator()
        self.teacher = teacher_agent
        self.escalation_threshold = escalation_threshold

    def evaluate_stage(
        self,
        stage_id: int,
        stage_name: str,
        objectives: List[str],
        probe_outputs: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Executes tiered evaluation across probe outputs.
        Returns a dictionary containing evaluation results and escalation routing status.
        """
        logger.info(f"--- [Tier 1] Running Fast Student Evaluation for Stage {stage_id} ({stage_name}) ---")
        student_report = self.student.evaluate_batch(
            stage_id, stage_name, probe_outputs, escalation_threshold=self.escalation_threshold
        )

        result = {
            "stage_id": stage_id,
            "stage_name": stage_name,
            "tiered_eval_mode": "student_fast_pass",
            "escalated_to_teacher": False,
            "mean_overall_score": student_report.mean_overall_score,
            "student_report": student_report.model_dump(),
            "failure_mode_report": None
        }

        # Check if escalation is required
        if not student_report.escalation_required and self.teacher is None:
            logger.info("Fast student pass: Quality meets threshold. Skipping teacher escalation.")
            return result

        if student_report.escalation_required:
            logger.warning(
                f"--- [Tier 2 Escalation Triggered] Mean score {student_report.mean_overall_score:.2f} "
                f"or risk flags detected. Escalating to 70B Teacher Critic. ---"
            )
            result["tiered_eval_mode"] = "teacher_deep_critique"
            result["escalated_to_teacher"] = True

            if self.teacher is not None:
                teacher_critique = self.teacher.analyze_stage_outputs(
                    stage_id, stage_name, objectives, probe_outputs
                )
                raw_critique = teacher_critique.get("critique_raw", "")
                parser = FailureModeParser()
                failure_report = parser.parse(stage_id, stage_name, raw_critique)
                result["teacher_critique"] = teacher_critique
                result["failure_mode_report"] = failure_report.model_dump()
            else:
                logger.info("Teacher agent not configured — mock teacher critique returned for escalation.")
                # Run local failure parser on synthesized critique as fallback
                parser = FailureModeParser()
                synthetic_critique = f"Escalated evaluation for {stage_name}. " + " ".join(student_report.escalation_reasons)
                failure_report = parser.parse(stage_id, stage_name, synthetic_critique)
                result["failure_mode_report"] = failure_report.model_dump()

        return result
