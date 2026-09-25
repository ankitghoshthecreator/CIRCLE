"""
generator/rl_prompt_builder.py
CIRCLE Part 17: Groq-Evidence → Ollama Prompt Builder

Converts the *specific failure evidence* from a GroqCriticAgent critique
into structured system+user prompt pairs that Ollama can act on directly.

Instead of generic prompts like "generate data for degenerate_repetition",
the prompts say:
  "The model produced this specific mistake: '<evidence>'.
   Generate 5 training examples that teach the correct behaviour."

This gives Ollama the full context from Groq so the synthetic data is
precisely targeted at what the student model is getting wrong *right now*.
"""

import uuid
import logging
from typing import List, Optional
from dataclasses import dataclass, field

from eval.failure_parser import (
    FailureModeReport,
    FailureMode,
    Severity,
    FailureCategory,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class OllamaPromptRequest:
    """A fully formed prompt request ready to send to Ollama."""
    request_id:      str
    stage_id:        int
    failure_category: str
    severity:        str
    system_prompt:   str
    user_prompt:     str
    num_samples:     int            # How many training pairs to generate
    priority:        int            # 1 = highest priority

    def to_dict(self) -> dict:
        return {
            "request_id":       self.request_id,
            "stage_id":         self.stage_id,
            "failure_category": self.failure_category,
            "severity":         self.severity,
            "system_prompt":    self.system_prompt,
            "user_prompt":      self.user_prompt,
            "num_samples":      self.num_samples,
            "priority":         self.priority,
        }


@dataclass
class OllamaPromptBatch:
    """Collection of OllamaPromptRequests for one RL step."""
    stage_id:  int
    rl_step:   int
    requests:  List[OllamaPromptRequest] = field(default_factory=list)

    @property
    def total_samples_requested(self) -> int:
        return sum(r.num_samples for r in self.requests)

    @property
    def is_empty(self) -> bool:
        return len(self.requests) == 0


# ──────────────────────────────────────────────────────────────
# Prompt templates per failure category
# ──────────────────────────────────────────────────────────────

# Maps FailureCategory → (system_template, user_template)
# Placeholders: {evidence}, {stage_name}, {num_samples}
PROMPT_TEMPLATES = {
    FailureCategory.DEGENERATE_REPETITION: (
        "You are an expert NLP dataset creator. Your job is to generate high-quality English "
        "language model training pairs that teach the model to avoid repeating the same words "
        "or phrases. Each training pair must have an 'input_text' and a 'target_text' where "
        "the target demonstrates diverse, non-repetitive language. Return ONLY a JSON array.",
        "The student model produced text with severe token/phrase repetition:\n\n"
        "EVIDENCE: \"{evidence}\"\n\n"
        "Generate {num_samples} training pairs (JSON array of objects with keys "
        "'input_text' and 'target_text') that correct this exact failure. "
        "Each target_text must use varied vocabulary and avoid any repeated phrases."
    ),
    FailureCategory.ABRUPT_CUTOFF: (
        "You are an expert NLP dataset creator. Generate English training pairs where the "
        "target response is always a complete, well-formed sentence or paragraph that ends "
        "with proper terminal punctuation. Return ONLY a JSON array.",
        "The student model cut off its response mid-sentence:\n\n"
        "EVIDENCE: \"{evidence}\"\n\n"
        "Generate {num_samples} training pairs (JSON array of objects with keys "
        "'input_text' and 'target_text') where every target_text is at least 2 complete "
        "sentences ending with a period, exclamation mark, or question mark."
    ),
    FailureCategory.INCOHERENT_CONTINUATION: (
        "You are an expert NLP dataset creator. Generate English training pairs where the "
        "target logically continues the input with clear causal reasoning. Return ONLY a JSON array.",
        "The student model produced an incoherent continuation:\n\n"
        "EVIDENCE: \"{evidence}\"\n\n"
        "Generate {num_samples} training pairs (JSON array of objects with keys "
        "'input_text' and 'target_text') where each target_text demonstrates clear, "
        "logically-chained reasoning from the input."
    ),
    FailureCategory.SPEAKER_DRIFT: (
        "You are an expert NLP dataset creator. Generate English training pairs that require "
        "sustained, consistent point-of-view narration. Return ONLY a JSON array.",
        "The student model switched perspective (speaker drift) during generation:\n\n"
        "EVIDENCE: \"{evidence}\"\n\n"
        "Generate {num_samples} training pairs (JSON array of objects with keys "
        "'input_text' and 'target_text') where the target maintains consistent first-person "
        "OR third-person narration throughout — never switching."
    ),
    FailureCategory.SUBJECT_VERB_MISMATCH: (
        "You are an expert NLP dataset creator. Generate English training pairs that test "
        "and demonstrate correct subject-verb agreement. Return ONLY a JSON array.",
        "The student model produced subject-verb agreement errors:\n\n"
        "EVIDENCE: \"{evidence}\"\n\n"
        "Generate {num_samples} training pairs (JSON array of objects with keys "
        "'input_text' and 'target_text') where each target_text uses perfect "
        "grammatical subject-verb agreement, including tricky plural/singular cases."
    ),
    FailureCategory.RUN_ON_SENTENCE: (
        "You are an expert NLP dataset creator. Generate English training pairs that "
        "demonstrate proper sentence boundaries and punctuation. Return ONLY a JSON array.",
        "The student model produced run-on sentences without proper punctuation:\n\n"
        "EVIDENCE: \"{evidence}\"\n\n"
        "Generate {num_samples} training pairs (JSON array of objects with keys "
        "'input_text' and 'target_text') where each target_text uses clear sentence "
        "boundaries, with periods and correct punctuation throughout."
    ),
    FailureCategory.SEMANTIC_CIRCULARITY: (
        "You are an expert NLP dataset creator. Generate English training pairs where "
        "responses add new information rather than restating the question. Return ONLY a JSON array.",
        "The student model merely restated the prompt without adding new information:\n\n"
        "EVIDENCE: \"{evidence}\"\n\n"
        "Generate {num_samples} training pairs (JSON array of objects with keys "
        "'input_text' and 'target_text') where each target_text introduces new facts, "
        "elaborations, or examples — not a restatement of the input."
    ),
    FailureCategory.ENTITY_DROPPING: (
        "You are an expert NLP dataset creator. Generate English training pairs that "
        "require maintaining entity references across multiple sentences. Return ONLY a JSON array.",
        "The student model dropped entity references mid-response:\n\n"
        "EVIDENCE: \"{evidence}\"\n\n"
        "Generate {num_samples} training pairs (JSON array of objects with keys "
        "'input_text' and 'target_text') where each target_text consistently references "
        "the same entities throughout without dropping or replacing them."
    ),
    FailureCategory.VOCABULARY_POVERTY: (
        "You are an expert NLP dataset creator. Generate English training pairs that "
        "demonstrate rich, varied vocabulary. Return ONLY a JSON array.",
        "The student model used extremely limited vocabulary:\n\n"
        "EVIDENCE: \"{evidence}\"\n\n"
        "Generate {num_samples} training pairs (JSON array of objects with keys "
        "'input_text' and 'target_text') where each target_text uses diverse, "
        "expressive vocabulary with synonyms and domain-appropriate terminology."
    ),
}

# Default template for unmapped categories
_DEFAULT_SYSTEM = (
    "You are an expert NLP dataset creator. Generate high-quality English language model "
    "training pairs that address a specific linguistic failure. Return ONLY a JSON array."
)
_DEFAULT_USER = (
    "The student model exhibited this failure: {failure_category}\n\n"
    "EVIDENCE: \"{evidence}\"\n\n"
    "Generate {num_samples} training pairs (JSON array of objects with keys "
    "'input_text' and 'target_text') that teach the correct behaviour."
)


# How many samples to request per severity level
SAMPLES_BY_SEVERITY = {
    Severity.CRITICAL: 10,
    Severity.HIGH:     8,
    Severity.MEDIUM:   5,
    Severity.LOW:      3,
}


# ──────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────

class RLPromptBuilder:
    """
    Converts a FailureModeReport (from Groq critique) into a batch of
    OllamaPromptRequests — one per detected failure mode that triggers
    data commissioning.

    Usage:
        builder = RLPromptBuilder()
        batch = builder.build_from_report(report, stage_id=1, rl_step=3)
        for req in batch.requests:
            # send req.system_prompt + req.user_prompt to Ollama
    """

    def __init__(self, default_samples: int = 5):
        self.default_samples = default_samples

    def build_from_report(
        self,
        report: FailureModeReport,
        stage_id: int,
        rl_step: int = 1,
        stage_name: str = "",
    ) -> OllamaPromptBatch:
        """
        Build an OllamaPromptBatch from a FailureModeReport.

        Only failure modes with commission_data=True are included.
        Modes are sorted by severity (CRITICAL first).
        """
        batch = OllamaPromptBatch(stage_id=stage_id, rl_step=rl_step)

        commissionable = [
            fm for fm in report.failure_modes if fm.commission_data
        ]

        if not commissionable:
            logger.info(
                f"[RLPromptBuilder] Stage {stage_id} | No commissionable failure modes. "
                "No Ollama prompts generated."
            )
            return batch

        for priority, fm in enumerate(commissionable, start=1):
            req = self._build_request(fm, stage_id, priority, stage_name)
            batch.requests.append(req)
            logger.debug(
                f"[RLPromptBuilder] Built request for [{fm.severity}] {fm.category} "
                f"({req.num_samples} samples, priority {priority})"
            )

        logger.info(
            f"[RLPromptBuilder] Stage {stage_id} | RL step {rl_step} | "
            f"Built {len(batch.requests)} Ollama prompt requests "
            f"({batch.total_samples_requested} total samples requested)."
        )
        return batch

    def build_from_raw_critique(
        self,
        critique_raw: str,
        stage_id: int,
        stage_name: str,
        rl_step: int = 1,
    ) -> OllamaPromptBatch:
        """
        Convenience: parse a raw Groq critique string and build the batch.
        """
        from eval.failure_parser import FailureModeParser
        parser = FailureModeParser()
        report = parser.parse(stage_id, stage_name, critique_raw)
        return self.build_from_report(report, stage_id, rl_step, stage_name)

    # ── Internal ─────────────────────────────────────────────

    def _build_request(
        self,
        fm: FailureMode,
        stage_id: int,
        priority: int,
        stage_name: str,
    ) -> OllamaPromptRequest:
        category = fm.category
        evidence = fm.evidence[:300]  # cap evidence for prompt length
        num_samples = SAMPLES_BY_SEVERITY.get(str(fm.severity), self.default_samples)

        # Get template pair
        sys_tmpl, usr_tmpl = PROMPT_TEMPLATES.get(
            category, (_DEFAULT_SYSTEM, _DEFAULT_USER)
        )

        system_prompt = sys_tmpl
        user_prompt = usr_tmpl.format(
            evidence=evidence,
            num_samples=num_samples,
            stage_name=stage_name,
            failure_category=str(category),
        )

        return OllamaPromptRequest(
            request_id=f"rl_{stage_id}_{str(category)}_{uuid.uuid4().hex[:6]}",
            stage_id=stage_id,
            failure_category=str(category),
            severity=str(fm.severity),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            num_samples=num_samples,
            priority=priority,
        )


# ──────────────────────────────────────────────────────────────
# CLI smoke test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    MOCK_CRITIQUE = (
        "The model shows degenerate repetition, looping the word 'the' dozens of times. "
        "There is severe speaker drift — it starts in first-person then switches to third. "
        "Vocabulary is extremely limited with only common one-syllable words used."
    )

    builder = RLPromptBuilder()
    batch = builder.build_from_raw_critique(
        critique_raw=MOCK_CRITIQUE,
        stage_id=1,
        stage_name="Base English",
        rl_step=1,
    )

    print(f"\n=== OllamaPromptBatch (Stage {batch.stage_id}, RL Step {batch.rl_step}) ===")
    print(f"  Total Requests : {len(batch.requests)}")
    print(f"  Samples Needed : {batch.total_samples_requested}")
    for req in batch.requests:
        print(f"\n  [{req.priority}] {req.failure_category} ({req.severity}) — {req.num_samples} samples")
        print(f"  User prompt preview: {req.user_prompt[:200]}...")
