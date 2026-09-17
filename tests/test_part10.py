"""
tests/test_part10.py
CIRCLE - Part 10 Unit Tests: Synthetic Data Post-Processing & Validation Gating

Tests SyntheticDataValidator (7 layers) and DatasetMerger.
"""

import os
import sys
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

PASS, FAIL = "PASS", "FAIL"
results = []

def report(tid, name, status, detail=""):
    icon = "OK" if status == PASS else "XX"
    print(f"  [{icon}] Test {tid:02d}: {name} - {status}" + (f"\n         Detail: {detail}" if detail else ""))
    results.append((tid, name, status, detail))


print("\n" + "="*65)
print("  CIRCLE - Part 10 Unit Tests (Synthetic Data Validator)")
print("="*65 + "\n")


# ── T01: Valid sample passes all 7 layers ────────────────────
try:
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    sample = SyntheticDataSample(
        sample_id="t01_s1", stage_id=1, category="speaker_drift",
        input_text="Write a first-person narrative about the ocean voyage.",
        target_text="I watched the waves crest as the ship sailed into the horizon.",
        prompt_spec_id="r1"
    )
    result = validator.validate_sample(sample)
    assert result.passed, f"Expected PASS, got: {result.rejection_reasons}"
    assert result.quality_score == 1.0
    report(1, "High-quality sample passes all 7 validation layers", PASS)
except Exception as e:
    report(1, "High-quality sample passes all 7 validation layers", FAIL, str(e))


# ── T02: Empty input_text rejected at Layer 1 ────────────────
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    sample = SyntheticDataSample(
        sample_id="t02_s1", stage_id=1, category="abrupt_cutoff",
        input_text="", target_text="This is a valid target.",
        prompt_spec_id="r1"
    )
    result = validator.validate_sample(sample)
    assert not result.passed
    assert any("L1_SCHEMA" in r for r in result.rejection_reasons)
    report(2, "Empty input_text rejected at Layer 1 (Schema Validation)", PASS,
           f"Reasons: {result.rejection_reasons}")
except Exception as e:
    report(2, "Empty input_text rejected at Layer 1 (Schema Validation)", FAIL, str(e))


# ── T03: Short target_text (< 10 chars) rejected at Layer 2 ──
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    sample = SyntheticDataSample(
        sample_id="t03_s1", stage_id=1, category="degenerate_repetition",
        input_text="Explain why ML models need training data.",
        target_text="Yes.",   # 4 chars — too short
        prompt_spec_id="r1"
    )
    result = validator.validate_sample(sample)
    assert not result.passed
    assert any("L2_MIN_LEN" in r for r in result.rejection_reasons)
    report(3, "Short target_text (< 10 chars) rejected at Layer 2 (Min Length)", PASS,
           f"target len=4, reasons: {result.rejection_reasons}")
except Exception as e:
    report(3, "Short target_text (< 10 chars) rejected at Layer 2 (Min Length)", FAIL, str(e))


# ── T04: Degenerate repetition loop rejected at Layer 4 ──────
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    # Simulate a degenerate token loop
    looped_text = "the model is the model is the model is the model is the model is the model is the model is"
    sample = SyntheticDataSample(
        sample_id="t04_s1", stage_id=1, category="degenerate_repetition",
        input_text="Describe the model behaviour.",
        target_text=looped_text,
        prompt_spec_id="r1"
    )
    result = validator.validate_sample(sample)
    assert not result.passed
    assert any("L4_DEGENERATE" in r for r in result.rejection_reasons)
    report(4, "Degenerate repetition loop rejected at Layer 4", PASS,
           f"Bigram ratio detected, reasons: {result.rejection_reasons}")
except Exception as e:
    report(4, "Degenerate repetition loop rejected at Layer 4", FAIL, str(e))


# ── T05: input == target rejected at Layer 5 (Identity) ──────
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    identical_text = "This is a complete sentence about machine learning."
    sample = SyntheticDataSample(
        sample_id="t05_s1", stage_id=1, category="semantic_circularity",
        input_text=identical_text,
        target_text=identical_text,
        prompt_spec_id="r1"
    )
    result = validator.validate_sample(sample)
    assert not result.passed
    assert any("L5_IDENTITY" in r for r in result.rejection_reasons)
    report(5, "input_text == target_text rejected at Layer 5 (Identity Check)", PASS)
except Exception as e:
    report(5, "input_text == target_text rejected at Layer 5 (Identity Check)", FAIL, str(e))


# ── T06: Duplicate pair rejected at Layer 7 (Deduplication) ──
try:
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    inp = "What is transfer learning in deep neural networks?"
    tgt = "Transfer learning reuses pretrained model weights as a starting point for new tasks."

    samples = [
        SyntheticDataSample(sample_id=f"t06_s{i}", stage_id=1, category="topic_drift",
                            input_text=inp, target_text=tgt, prompt_spec_id="r1")
        for i in range(3)
    ]
    batch = SyntheticDataBatch(stage_id=1, total_samples=3, samples=samples)
    report_obj, valid_examples = validator.validate_batch(batch)

    # Only 1 should pass; the other 2 are duplicates
    assert report_obj.total_passed == 1, f"Expected 1 passed, got {report_obj.total_passed}"
    assert report_obj.total_rejected == 2
    dup_reasons = [r for r in report_obj.results if not r.passed]
    assert all(any("L7_DUPLICATE" in rr for rr in r.rejection_reasons) for r in dup_reasons)
    report(6, "Duplicate (input, target) pairs rejected at Layer 7 (SHA-256 Deduplication)", PASS,
           f"1 passed, 2 duplicates rejected")
except Exception as e:
    report(6, "Duplicate (input, target) pairs rejected at Layer 7 (SHA-256 Deduplication)", FAIL, str(e))


# ── T07: ValidationReport Pydantic serialization roundtrip ───
try:
    import json
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    sample = SyntheticDataSample(
        sample_id="t07_s1", stage_id=2, category="entity_dropping",
        input_text="Alice met Bob at the conference. They discussed neural architectures.",
        target_text="Alice and Bob discussed neural architecture design at the conference.",
        prompt_spec_id="r2"
    )
    batch = SyntheticDataBatch(stage_id=2, total_samples=1, samples=[sample])
    report_obj, _ = validator.validate_batch(batch)

    dumped = json.dumps(report_obj.model_dump())
    reloaded = json.loads(dumped)
    assert reloaded["stage_id"] == 2
    assert reloaded["total_passed"] == 1
    assert len(reloaded["results"]) == 1
    report(7, "ValidationReport Pydantic serialization roundtrip succeeds", PASS)
except Exception as e:
    report(7, "ValidationReport Pydantic serialization roundtrip succeeds", FAIL, str(e))


# ── T08: DatasetMerger appends validated examples to stage dataset ─
try:
    import tempfile
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator, DatasetMerger

    with tempfile.TemporaryDirectory() as tmpdir:
        validator = SyntheticDataValidator()
        merger = DatasetMerger()

        samples = [
            SyntheticDataSample(
                sample_id=f"t08_s{i}", stage_id=1, category="lexical_repetition",
                input_text=f"Describe technique {i} used in curriculum learning pipelines.",
                target_text=f"Technique {i} involves adaptive replay scheduling during stage transitions.",
                prompt_spec_id="r1"
            )
            for i in range(5)
        ]
        batch = SyntheticDataBatch(stage_id=1, total_samples=5, samples=samples)
        report_obj, valid_examples = validator.validate_batch(batch)

        added = merger.merge_into_stage_dataset(valid_examples, stage_id=1, data_dir=tmpdir)
        assert added == 5, f"Expected 5 new examples added, got {added}"

        # Verify saved dataset
        saved_path = os.path.join(tmpdir, "stage_1", "dataset.json")
        assert os.path.exists(saved_path)
        with open(saved_path) as f:
            saved = json.load(f)
        assert len(saved) == 5

        report(8, "DatasetMerger appends validated examples to stage dataset on disk", PASS,
               f"Added: {added} examples, dataset file verified")
except Exception as e:
    report(8, "DatasetMerger appends validated examples to stage dataset on disk", FAIL, str(e))


# ─── SUMMARY ───
print("\n" + "="*65)
passed = sum(1 for _, _, s, _ in results if s == PASS)
failed = sum(1 for _, _, s, _ in results if s == FAIL)
print(f"  RESULTS: {passed} PASSED  |  {failed} FAILED  |  {len(results)} TOTAL")
print("="*65 + "\n")
if failed > 0:
    print("  FAILED TESTS:")
    for tid, name, status, detail in results:
        if status == FAIL:
            print(f"    Test {tid:02d}: {name}")
            print(f"    -> {detail}")
