"""
eval/failure_parser.py
Part 6: Structured Failure Mode Analysis & Parser

Transforms unstructured LLM critique text into machine-readable,
categorized failure mode reports with severity scoring and dataset
commissioning triggers.
"""

import re
import json
import logging
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Failure Mode Taxonomy
# ──────────────────────────────────────────────────────────────

class FailureCategory(str, Enum):
    # Fluency / Generation
    DEGENERATE_REPETITION   = "degenerate_repetition"       # Same token/phrase looped
    ABRUPT_CUTOFF           = "abrupt_cutoff"                # Mid-sentence termination
    INCOHERENT_CONTINUATION = "incoherent_continuation"      # Logically nonsensical follow-on

    # Syntax / Grammar
    SUBJECT_VERB_MISMATCH   = "subject_verb_mismatch"
    RUN_ON_SENTENCE         = "run_on_sentence"
    CLAUSE_BOUNDARY_ERROR   = "clause_boundary_error"
    MISSING_PUNCTUATION     = "missing_punctuation"
    INCORRECT_PREPOSITION   = "incorrect_preposition"

    # Semantics / Discourse
    ENTITY_DROPPING         = "entity_dropping"              # Entity mentioned then lost
    SPEAKER_DRIFT           = "speaker_drift"                # 1st/3rd person confusion
    REGISTER_SHIFT          = "register_shift"               # Formal ↔ casual mid-response
    SEMANTIC_CIRCULARITY    = "semantic_circularity"         # Rephrases prompt without adding info
    TOPIC_DRIFT             = "topic_drift"                  # Wanders off-topic

    # Vocabulary
    LEXICAL_REPETITION      = "lexical_repetition"           # Same word overused in context
    VOCABULARY_POVERTY      = "vocabulary_poverty"           # Very limited word choice
    DOMAIN_VOCABULARY_MISS  = "domain_vocabulary_miss"       # Missing expected domain terms

    # Meta / Unknown
    OTHER                   = "other"


class Severity(str, Enum):
    LOW      = "low"       # Minor, cosmetic — monitor
    MEDIUM   = "medium"    # Affects quality — targeted data helps
    HIGH     = "high"      # Blocks stage advancement — must fix before training next stage
    CRITICAL = "critical"  # Fundamental capability gap — requires curriculum redesign


# Severity lookup per category
CATEGORY_SEVERITY_MAP: Dict[FailureCategory, Severity] = {
    FailureCategory.DEGENERATE_REPETITION:   Severity.HIGH,
    FailureCategory.ABRUPT_CUTOFF:           Severity.HIGH,
    FailureCategory.INCOHERENT_CONTINUATION: Severity.CRITICAL,
    FailureCategory.SUBJECT_VERB_MISMATCH:   Severity.MEDIUM,
    FailureCategory.RUN_ON_SENTENCE:         Severity.MEDIUM,
    FailureCategory.CLAUSE_BOUNDARY_ERROR:   Severity.MEDIUM,
    FailureCategory.MISSING_PUNCTUATION:     Severity.LOW,
    FailureCategory.INCORRECT_PREPOSITION:   Severity.LOW,
    FailureCategory.ENTITY_DROPPING:         Severity.HIGH,
    FailureCategory.SPEAKER_DRIFT:           Severity.HIGH,
    FailureCategory.REGISTER_SHIFT:          Severity.MEDIUM,
    FailureCategory.SEMANTIC_CIRCULARITY:    Severity.HIGH,
    FailureCategory.TOPIC_DRIFT:             Severity.MEDIUM,
    FailureCategory.LEXICAL_REPETITION:      Severity.MEDIUM,
    FailureCategory.VOCABULARY_POVERTY:      Severity.MEDIUM,
    FailureCategory.DOMAIN_VOCABULARY_MISS:  Severity.LOW,
    FailureCategory.OTHER:                   Severity.LOW,
}

# Whether this failure triggers immediate synthetic data commissioning
COMMISSIONING_TRIGGERS: Dict[Severity, bool] = {
    Severity.LOW:      False,
    Severity.MEDIUM:   True,
    Severity.HIGH:     True,
    Severity.CRITICAL: True,
}


# ──────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────

class FailureMode(BaseModel):
    category: FailureCategory
    severity: Severity
    evidence: str = Field(..., description="Quoted or paraphrased evidence from the critique text")
    commission_data: bool = Field(False, description="Whether this failure triggers data synthesis")
    suggested_prompt_focus: Optional[str] = Field(None, description="Hint for the prompt writer (Part 7)")

    model_config = {"use_enum_values": True}


class DataCommissionRequest(BaseModel):
    stage_id: int
    failure_category: str
    severity: str
    suggested_prompt_focus: str
    priority: int = Field(..., description="1=highest, 5=lowest")

    model_config = {"use_enum_values": True}


class FailureModeReport(BaseModel):
    stage_id: int
    stage_name: str
    failure_modes: List[FailureMode]
    overall_severity: Severity
    advance_to_next_stage: bool = Field(False, description="True if no HIGH/CRITICAL failures")
    commission_requests: List[DataCommissionRequest] = Field(default_factory=list)
    raw_critique_length: int = 0

    model_config = {"use_enum_values": True}


# ──────────────────────────────────────────────────────────────
# Keyword → Category Signal Map (for regex-based detection)
# ──────────────────────────────────────────────────────────────

KEYWORD_SIGNALS: List[tuple] = [
    # (regex pattern, FailureCategory)
    (r"repeti(tion|tive|t[eo]d)|repeat(s|ing|ed)|loop(s|ing)|same (word|phrase|token)",
     FailureCategory.DEGENERATE_REPETITION),

    (r"cut(s)? off|abrupt(ly)?|incomplete|truncat|mid.sentence|unfinished",
     FailureCategory.ABRUPT_CUTOFF),

    (r"incoher|nonsens|illogical|does not follow|no logical",
     FailureCategory.INCOHERENT_CONTINUATION),

    (r"subject.verb|agreement error|verb form|tense error",
     FailureCategory.SUBJECT_VERB_MISMATCH),

    (r"run.on|run on sentence|comma splice",
     FailureCategory.RUN_ON_SENTENCE),

    (r"clause bound|sentence bound|clause break|fragment",
     FailureCategory.CLAUSE_BOUNDARY_ERROR),

    (r"missing punct|no (period|comma|question mark)|punctuation error",
     FailureCategory.MISSING_PUNCTUATION),

    (r"wrong preposition|incorrect preposition|preposition error",
     FailureCategory.INCORRECT_PREPOSITION),

    (r"entity drop|lost (the |a )?(entity|subject|topic)|coreference",
     FailureCategory.ENTITY_DROPPING),

    (r"speaker drift|person shift|first.person|third.person switch|pov shift",
     FailureCategory.SPEAKER_DRIFT),

    (r"register (shift|change|drift)|formal.casual|casual.formal|tone shift",
     FailureCategory.REGISTER_SHIFT),

    (r"circular|tautolog|rephrases? the prompt|restates? the question|no new info",
     FailureCategory.SEMANTIC_CIRCULARITY),

    (r"off.topic|topic drift|wanders|tangent|unrelated",
     FailureCategory.TOPIC_DRIFT),

    (r"lexical repetition|same word|word repetition|overuse",
     FailureCategory.LEXICAL_REPETITION),

    (r"limited vocabular|vocabular poverty|lack(s)? variet|monotonous word",
     FailureCategory.VOCABULARY_POVERTY),

    (r"domain vocab|technical term|missing term|expected (term|word)",
     FailureCategory.DOMAIN_VOCABULARY_MISS),
]


# ──────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────

class FailureModeParser:
    """
    Parses raw LLM critique text into a structured FailureModeReport.

    Strategy:
    1. Sliding window regex scan over critique sentences.
    2. Each matched category → FailureMode with evidence snippet.
    3. Deduplicate by category (keep first occurrence).
    4. Compute overall severity and advance/hold decision.
    5. Build DataCommissionRequest list for Part 7 (Prompt Writer).
    """

    def parse(
        self,
        stage_id: int,
        stage_name: str,
        critique_raw: str
    ) -> FailureModeReport:

        if not critique_raw or len(critique_raw.strip()) < 10:
            logger.warning("Critique text is too short to parse meaningfully.")
            return FailureModeReport(
                stage_id=stage_id,
                stage_name=stage_name,
                failure_modes=[],
                overall_severity=Severity.LOW,
                advance_to_next_stage=True,
                commission_requests=[],
                raw_critique_length=len(critique_raw)
            )

        # Strip thinking tags (Qwen outputs <think>...</think> blocks)
        cleaned = re.sub(r"<think>.*?</think>", "", critique_raw, flags=re.DOTALL).strip()

        # Split into sentence-level chunks for evidence extraction
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)

        detected: Dict[FailureCategory, str] = {}

        for sentence in sentences:
            for pattern, category in KEYWORD_SIGNALS:
                if category in detected:
                    continue  # already found, skip duplicate
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Clip evidence to 200 chars
                    evidence = sentence.strip()[:200]
                    detected[category] = evidence

        # Build FailureMode objects
        failure_modes: List[FailureMode] = []
        for category, evidence in detected.items():
            severity = CATEGORY_SEVERITY_MAP.get(category, Severity.LOW)
            commission = COMMISSIONING_TRIGGERS.get(severity, False)

            # Generate suggested prompt focus hint
            prompt_focus = self._suggest_prompt_focus(category)

            failure_modes.append(FailureMode(
                category=category,
                severity=severity,
                evidence=evidence,
                commission_data=commission,
                suggested_prompt_focus=prompt_focus
            ))

        # Sort by severity (CRITICAL → HIGH → MEDIUM → LOW)
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3
        }
        failure_modes.sort(key=lambda fm: severity_order.get(fm.severity, 4))

        # Overall severity = worst individual severity found
        overall = Severity.LOW
        for fm in failure_modes:
            if severity_order.get(fm.severity, 4) < severity_order.get(overall, 4):
                overall = fm.severity

        # Advance only if no HIGH or CRITICAL failures
        blocking = {Severity.HIGH, Severity.CRITICAL}
        advance = not any(fm.severity in blocking for fm in failure_modes)

        # Build data commission requests (for Part 7)
        commissions: List[DataCommissionRequest] = []
        priority = 1
        for fm in failure_modes:
            if fm.commission_data:
                commissions.append(DataCommissionRequest(
                    stage_id=stage_id,
                    failure_category=fm.category,
                    severity=fm.severity,
                    suggested_prompt_focus=fm.suggested_prompt_focus or fm.category,
                    priority=min(priority, 5)
                ))
                priority += 1

        logger.info(
            f"Parsed {len(failure_modes)} failure modes for Stage {stage_id}. "
            f"Overall: {overall}. Advance: {advance}. Commissions: {len(commissions)}."
        )

        return FailureModeReport(
            stage_id=stage_id,
            stage_name=stage_name,
            failure_modes=failure_modes,
            overall_severity=overall,
            advance_to_next_stage=advance,
            commission_requests=commissions,
            raw_critique_length=len(critique_raw)
        )

    def _suggest_prompt_focus(self, category: FailureCategory) -> str:
        FOCUS_HINTS = {
            FailureCategory.DEGENERATE_REPETITION:
                "Generate prompts requiring diverse lexical responses; penalize repeated tokens",
            FailureCategory.ABRUPT_CUTOFF:
                "Generate prompts with expected long-form completions of 3+ sentences",
            FailureCategory.INCOHERENT_CONTINUATION:
                "Generate prompts that require logically chained causal reasoning",
            FailureCategory.SUBJECT_VERB_MISMATCH:
                "Generate sentence completion prompts with plural/singular subject traps",
            FailureCategory.RUN_ON_SENTENCE:
                "Generate prompts that model multi-clause sentences requiring proper stops",
            FailureCategory.CLAUSE_BOUNDARY_ERROR:
                "Generate prompts with embedded relative clauses to test boundary handling",
            FailureCategory.MISSING_PUNCTUATION:
                "Generate dialogue and list-style completions requiring commas and periods",
            FailureCategory.INCORRECT_PREPOSITION:
                "Generate prompts with prepositional phrase completions",
            FailureCategory.ENTITY_DROPPING:
                "Generate multi-sentence prompts requiring consistent entity reference",
            FailureCategory.SPEAKER_DRIFT:
                "Generate prompts requiring sustained first-person or third-person narration",
            FailureCategory.REGISTER_SHIFT:
                "Generate prompts with explicit formal or informal register constraints",
            FailureCategory.SEMANTIC_CIRCULARITY:
                "Generate prompts requiring novel factual elaboration beyond restating the premise",
            FailureCategory.TOPIC_DRIFT:
                "Generate focused prompts with a clear topic constraint and expected on-topic completion",
            FailureCategory.LEXICAL_REPETITION:
                "Generate prompts where vocabulary diversity is demonstrated through synonym usage",
            FailureCategory.VOCABULARY_POVERTY:
                "Generate prompts in specialized domains requiring rich vocabulary",
            FailureCategory.DOMAIN_VOCABULARY_MISS:
                "Generate domain-specific (ML, CS, linguistics) completion prompts",
            FailureCategory.OTHER:
                "Generate general-purpose diverse completion prompts",
        }
        return FOCUS_HINTS.get(category, "Generate targeted completions for this failure mode")


# ──────────────────────────────────────────────────────────────
# Convenience: parse from saved critique JSON report
# ──────────────────────────────────────────────────────────────

def parse_critique_report(report_path: str) -> FailureModeReport:
    """Load a saved stage critique JSON and parse it into a FailureModeReport."""
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parser = FailureModeParser()
    return parser.parse(
        stage_id=data["stage_id"],
        stage_name=data.get("stage_name", f"Stage {data['stage_id']}"),
        critique_raw=data.get("critique_raw", "")
    )


def save_failure_report(report: FailureModeReport, output_dir: str = "./eval/reports") -> str:
    """Serialise a FailureModeReport to JSON and save alongside the critique."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"stage_{report.stage_id}_failures.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
    logger.info(f"Failure mode report saved to '{path}'")
    return path


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    report_path = "./eval/reports/stage_1_critique.json"
    if not __import__("os").path.exists(report_path):
        print(f"ERROR: Could not find '{report_path}'. Run run_stage_eval.py first.")
        sys.exit(1)

    print("\n--- Parsing Stage 1 Critique Report ---")
    failure_report = parse_critique_report(report_path)

    print(f"\nStage          : {failure_report.stage_name}")
    print(f"Overall Severity: {failure_report.overall_severity}")
    print(f"Advance Stage  : {failure_report.advance_to_next_stage}")
    print(f"Failure Modes  : {len(failure_report.failure_modes)}")

    print("\nDetected Failure Modes:")
    for fm in failure_report.failure_modes:
        print(f"  [{fm.severity.upper():8s}] {fm.category}")
        print(f"           Evidence : {fm.evidence[:120]}...")
        print(f"           Commission: {fm.commission_data}  |  Focus: {fm.suggested_prompt_focus[:80]}")

    print(f"\nData Commission Requests ({len(failure_report.commission_requests)}):")
    for cr in failure_report.commission_requests:
        print(f"  [P{cr.priority}] {cr.failure_category} ({cr.severity}) -> {cr.suggested_prompt_focus[:80]}")

    out = save_failure_report(failure_report)
    print(f"\nSaved structured failure report to: {out}")
