"""
generator/validator.py
Part 10: Synthetic Data Post-Processing & Validation Gating

Filters out noise, hallucinations, invalid formatting, and duplicates from
SyntheticDataBatch before samples are admitted into the active training buffer.
Produces a structured ValidationReport and feeds validated samples to DatasetMerger.
"""

import os
import json
import hashlib
import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from generator.generate import SyntheticDataSample, SyntheticDataBatch
from trainer.curriculum.dataset_handler import (
    CurriculumExample,
    load_stage_dataset,
    save_stage_dataset,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Validation Pydantic Schemas
# ──────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    """Per-sample validation outcome."""
    sample_id: str = Field(..., description="ID of the evaluated SyntheticDataSample")
    passed: bool = Field(..., description="True if sample cleared all validation gates")
    rejection_reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable rejection reasons (empty if passed)"
    )
    quality_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Composite quality score 0.0-1.0 (1.0 = fully clean)"
    )

    model_config = {"use_enum_values": True}


class ValidationReport(BaseModel):
    """Aggregate report for an entire SyntheticDataBatch validation run."""
    stage_id: int = Field(..., description="Curriculum stage ID of the batch")
    total_input: int = Field(..., description="Total samples submitted")
    total_passed: int = Field(..., description="Samples that passed all gates")
    total_rejected: int = Field(..., description="Samples rejected by at least one gate")
    pass_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of samples passed")
    results: List[ValidationResult] = Field(default_factory=list)
    validated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    model_config = {"use_enum_values": True}


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

MIN_TEXT_LENGTH: int = 10          # chars — below this is not useful training signal
MAX_TEXT_LENGTH: int = 4096        # chars — above this risks tokenizer truncation artefacts
BIGRAM_REPEAT_THRESHOLD: float = 0.35   # fraction of bigrams repeated = degenerate loop
CIRCULARITY_OVERLAP_THRESHOLD: float = 0.85  # Jaccard word overlap — target is trivial rephrasing


# ──────────────────────────────────────────────────────────────
# Synthetic Data Validator
# ──────────────────────────────────────────────────────────────

class SyntheticDataValidator:
    """
    Multi-layer quality gate for SyntheticDataSample objects.

    Validation Layers (applied in order):
        Layer 1 – Schema Validation    : non-empty input_text and target_text
        Layer 2 – Minimum Length       : >= MIN_TEXT_LENGTH chars each
        Layer 3 – Maximum Length       : <= MAX_TEXT_LENGTH chars each
        Layer 4 – Degenerate Repetition: bigram repeat ratio > BIGRAM_REPEAT_THRESHOLD
        Layer 5 – Identity Check       : input_text == target_text
        Layer 6 – Semantic Circularity : Jaccard overlap > CIRCULARITY_OVERLAP_THRESHOLD
        Layer 7 – Deduplication        : SHA-256 content hash dedup within the batch
    """

    def __init__(
        self,
        min_length: int = MIN_TEXT_LENGTH,
        max_length: int = MAX_TEXT_LENGTH,
        bigram_repeat_threshold: float = BIGRAM_REPEAT_THRESHOLD,
        circularity_threshold: float = CIRCULARITY_OVERLAP_THRESHOLD,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.bigram_repeat_threshold = bigram_repeat_threshold
        self.circularity_threshold = circularity_threshold

    # ── Public API ────────────────────────────────────────────

    def validate_batch(self, batch: SyntheticDataBatch) -> Tuple[ValidationReport, List[CurriculumExample]]:
        """
        Validates every sample in batch.

        Returns:
            report          : ValidationReport with per-sample outcomes
            valid_examples  : List[CurriculumExample] of all passed samples, ready for merger
        """
        seen_hashes: set = set()
        results: List[ValidationResult] = []
        valid_examples: List[CurriculumExample] = []

        for sample in batch.samples:
            result = self._validate_sample(sample, seen_hashes)
            results.append(result)
            if result.passed:
                # Register hash so subsequent identical samples are deduped
                h = _content_hash(sample.input_text, sample.target_text)
                seen_hashes.add(h)
                valid_examples.append(_to_curriculum_example(sample))

        total_input = len(results)
        total_passed = sum(1 for r in results if r.passed)
        total_rejected = total_input - total_passed
        pass_rate = round(total_passed / max(1, total_input), 4)

        report = ValidationReport(
            stage_id=batch.stage_id,
            total_input=total_input,
            total_passed=total_passed,
            total_rejected=total_rejected,
            pass_rate=pass_rate,
            results=results,
        )

        logger.info(
            f"[Validator] Stage {batch.stage_id}: {total_passed}/{total_input} passed "
            f"({pass_rate:.1%}), {total_rejected} rejected."
        )
        return report, valid_examples

    def validate_sample(self, sample: SyntheticDataSample) -> ValidationResult:
        """Validates a single sample in isolation (no deduplication context)."""
        return self._validate_sample(sample, seen_hashes=set())

    # ── Internal Validation Layers ────────────────────────────

    def _validate_sample(
        self, sample: SyntheticDataSample, seen_hashes: set
    ) -> ValidationResult:
        reasons: List[str] = []
        penalty = 0.0

        inp = sample.input_text
        tgt = sample.target_text

        # Layer 1 – Schema Validation
        if not isinstance(inp, str) or not inp.strip():
            reasons.append("L1_SCHEMA: input_text is empty or not a string")
            penalty += 0.5
        if not isinstance(tgt, str) or not tgt.strip():
            reasons.append("L1_SCHEMA: target_text is empty or not a string")
            penalty += 0.5

        # Short-circuit if schema fails entirely
        if len(reasons) >= 2:
            return ValidationResult(
                sample_id=sample.sample_id,
                passed=False,
                rejection_reasons=reasons,
                quality_score=0.0,
            )

        inp = inp.strip()
        tgt = tgt.strip()

        # Layer 2 – Minimum Length
        if len(inp) < self.min_length:
            reasons.append(f"L2_MIN_LEN: input_text too short ({len(inp)} < {self.min_length} chars)")
            penalty += 0.3
        if len(tgt) < self.min_length:
            reasons.append(f"L2_MIN_LEN: target_text too short ({len(tgt)} < {self.min_length} chars)")
            penalty += 0.3

        # Layer 3 – Maximum Length
        if len(inp) > self.max_length:
            reasons.append(f"L3_MAX_LEN: input_text too long ({len(inp)} > {self.max_length} chars)")
            penalty += 0.2
        if len(tgt) > self.max_length:
            reasons.append(f"L3_MAX_LEN: target_text too long ({len(tgt)} > {self.max_length} chars)")
            penalty += 0.2

        # Layer 4 – Degenerate Repetition (applied to target_text)
        rep_ratio = _bigram_repeat_ratio(tgt)
        if rep_ratio > self.bigram_repeat_threshold:
            reasons.append(
                f"L4_DEGENERATE_REPETITION: target bigram repeat ratio {rep_ratio:.2f} "
                f"> threshold {self.bigram_repeat_threshold}"
            )
            penalty += 0.4

        # Layer 5 – Identity Check (input == target)
        if inp.lower().strip() == tgt.lower().strip():
            reasons.append("L5_IDENTITY: input_text and target_text are identical")
            penalty += 0.5

        # Layer 6 – Semantic Circularity (Jaccard overlap)
        jaccard = _jaccard_word_overlap(inp, tgt)
        if jaccard > self.circularity_threshold and inp.lower().strip() != tgt.lower().strip():
            reasons.append(
                f"L6_CIRCULARITY: input/target Jaccard overlap {jaccard:.2f} "
                f"> threshold {self.circularity_threshold} (target is a trivial rephrasing)"
            )
            penalty += 0.35

        # Layer 7 – Deduplication
        h = _content_hash(inp, tgt)
        if h in seen_hashes:
            reasons.append("L7_DUPLICATE: identical (input, target) pair already seen in this batch")
            penalty += 1.0  # Hard reject

        quality_score = max(0.0, round(1.0 - min(penalty, 1.0), 3))
        passed = len(reasons) == 0

        return ValidationResult(
            sample_id=sample.sample_id,
            passed=passed,
            rejection_reasons=reasons,
            quality_score=quality_score,
        )


# ──────────────────────────────────────────────────────────────
# Dataset Merger
# ──────────────────────────────────────────────────────────────

class DatasetMerger:
    """
    Appends validated CurriculumExample objects into the active stage training dataset.
    Deduplicates against existing examples before appending.
    """

    def merge_into_stage_dataset(
        self,
        validated_examples: List[CurriculumExample],
        stage_id: int,
        data_dir: str = "./data",
    ) -> int:
        """
        Loads existing stage dataset, deduplicates against new examples,
        appends, and saves.

        Returns:
            int: Number of net-new examples added to the dataset
        """
        if not validated_examples:
            logger.info(f"[Merger] No validated examples to merge for Stage {stage_id}.")
            return 0

        # Load existing dataset (returns empty list if none exists)
        existing = load_stage_dataset(stage_id, data_dir=data_dir, use_seed_fallback=False)

        # Build existing fingerprint set
        existing_hashes = {
            _content_hash(ex.input_text, ex.target_text) for ex in existing
        }

        # Filter: keep only examples not already in existing dataset
        new_examples = [
            ex for ex in validated_examples
            if _content_hash(ex.input_text, ex.target_text) not in existing_hashes
        ]

        if not new_examples:
            logger.info(
                f"[Merger] Stage {stage_id}: All {len(validated_examples)} validated examples "
                f"are already in dataset — nothing to merge."
            )
            return 0

        merged = existing + new_examples
        save_stage_dataset(stage_id, merged, data_dir=data_dir)

        logger.info(
            f"[Merger] Stage {stage_id}: Appended {len(new_examples)} new examples "
            f"(total now: {len(merged)})."
        )
        return len(new_examples)


# ──────────────────────────────────────────────────────────────
# Private Utilities
# ──────────────────────────────────────────────────────────────

def _content_hash(input_text: str, target_text: str) -> str:
    """Returns a SHA-256 hex digest of the normalised (input, target) pair."""
    combined = f"{input_text.strip().lower()}|||{target_text.strip().lower()}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for heuristic checks."""
    return re.findall(r"[a-zA-Z0-9\u00C0-\u024F\u4e00-\u9fff]+", text.lower())


def _bigram_repeat_ratio(text: str) -> float:
    """
    Computes the fraction of bigrams in `text` that are repeated.
    High values indicate degenerate repetition loops.
    """
    tokens = _tokenize(text)
    if len(tokens) < 4:
        return 0.0
    bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    if not bigrams:
        return 0.0
    unique_bigrams = set(bigrams)
    # Repeated bigrams = total - unique
    repeated = len(bigrams) - len(unique_bigrams)
    return repeated / len(bigrams)


def _jaccard_word_overlap(text_a: str, text_b: str) -> float:
    """
    Computes Jaccard similarity between word-sets of two texts.
    High values (close to 1.0) indicate the texts share nearly all vocabulary.
    """
    words_a = set(_tokenize(text_a))
    words_b = set(_tokenize(text_b))
    union = words_a | words_b
    if not union:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / len(union)


def _to_curriculum_example(sample: SyntheticDataSample) -> CurriculumExample:
    """Converts a validated SyntheticDataSample to a CurriculumExample."""
    return CurriculumExample(
        input_text=sample.input_text,
        target_text=sample.target_text,
        stage_id=sample.stage_id,
        source="synthetic",
        metadata={
            "sample_id": sample.sample_id,
            "category": sample.category,
            "prompt_spec_id": sample.prompt_spec_id,
            "generated_at": sample.generated_at,
            "validated": True,
        },
    )
