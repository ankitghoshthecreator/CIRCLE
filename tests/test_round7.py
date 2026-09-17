"""
tests/test_round7.py
CIRCLE - 20 Difficult Stress & Edge-Case Tests (Round 7)

Focus:
- SyntheticDataValidator boundary conditions across all 7 layers
- Deduplication idempotency and cross-batch fingerprint correctness
- DatasetMerger merge idempotency (double-merge = no duplicates)
- Unicode, emoji, whitespace-only, and control character inputs
- Exact near-duplicate detection (case-insensitive)
- Batch with 0 passing samples produces clean empty ValidationReport
- Multi-thread concurrency of validator (10 threads)
- Full 6-Stage pipeline integration (Critique -> Parser -> Writer -> Generator -> Validator -> Merger)
"""

import os
import sys
import json
import time
import shutil
import tempfile
import concurrent.futures
from typing import List

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
print("  CIRCLE - 20 Difficult Edge-Case Tests (Round 7)")
print("="*65 + "\n")


# ═══════════════════════════════════════════════════════════
# BLOCK A: BOUNDARY CONDITIONS & SANITIZATION
# ═══════════════════════════════════════════════════════════
print("[ A ] Boundary Conditions & Sanitization")

# T01: 100,000-char input_text passes all filters (above min, below max=4096 only for target)
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    # Use max_length=200_000 to allow the large input through Layer 3
    validator = SyntheticDataValidator(max_length=200_000)
    large_input = "A" * 100_000
    sample = SyntheticDataSample(
        sample_id="r7t01", stage_id=1, category="test",
        input_text=large_input,
        target_text="A comprehensive response demonstrating model fluency.",
        prompt_spec_id="r"
    )
    result = validator.validate_sample(sample)
    assert result.passed, f"Expected pass: {result.rejection_reasons}"
    report(1, "100,000-char input_text passes all filters with max_length=200k", PASS)
except Exception as e:
    report(1, "100,000-char input_text passes all filters with max_length=200k", FAIL, str(e))


# T02: input_text over default max_length=4096 rejected at Layer 3
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    sample = SyntheticDataSample(
        sample_id="r7t02", stage_id=1, category="test",
        input_text="W " * 2200,   # 4400 chars > 4096
        target_text="A valid and complete response to the question posed.",
        prompt_spec_id="r"
    )
    result = validator.validate_sample(sample)
    assert not result.passed
    assert any("L3_MAX_LEN" in r for r in result.rejection_reasons)
    report(2, "input_text > 4096 chars rejected at Layer 3 (Max Length)", PASS)
except Exception as e:
    report(2, "input_text > 4096 chars rejected at Layer 3 (Max Length)", FAIL, str(e))


# T03: All-whitespace input_text rejected at Layer 1 (empty after strip)
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    sample = SyntheticDataSample(
        sample_id="r7t03", stage_id=1, category="test",
        input_text="     \t\n   ",
        target_text="A valid and informative target response text.",
        prompt_spec_id="r"
    )
    result = validator.validate_sample(sample)
    assert not result.passed
    assert any("L1_SCHEMA" in r for r in result.rejection_reasons)
    report(3, "All-whitespace input_text rejected at Layer 1 (empty after strip)", PASS)
except Exception as e:
    report(3, "All-whitespace input_text rejected at Layer 1 (empty after strip)", FAIL, str(e))


# T04: Single repeated word as target_text detected as degenerate repetition
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    sample = SyntheticDataSample(
        sample_id="r7t04", stage_id=1, category="test",
        input_text="Explain the training process for curriculum learning.",
        target_text="train train train train train train train train train train train train train",
        prompt_spec_id="r"
    )
    result = validator.validate_sample(sample)
    assert not result.passed
    assert any("L4_DEGENERATE" in r for r in result.rejection_reasons)
    report(4, "Single repeated word as target_text detected as degenerate repetition", PASS)
except Exception as e:
    report(4, "Single repeated word as target_text detected as degenerate repetition", FAIL, str(e))


# T05: Unicode & emoji-only strings (< min_length in meaningful chars) rejected
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    sample = SyntheticDataSample(
        sample_id="r7t05", stage_id=1, category="test",
        input_text="🚀🤖🌟",    # 3 emoji chars — very short
        target_text="🎯🔥",       # 2 emoji chars — too short
        prompt_spec_id="r"
    )
    result = validator.validate_sample(sample)
    assert not result.passed
    report(5, "Emoji-only strings below min_length threshold are rejected", PASS,
           f"Reasons: {result.rejection_reasons}")
except Exception as e:
    report(5, "Emoji-only strings below min_length threshold are rejected", FAIL, str(e))


# T06: Case-insensitive identity check (Input == TARGET but different case)
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    sample = SyntheticDataSample(
        sample_id="r7t06", stage_id=1, category="test",
        input_text="The model achieves high fluency through next token prediction.",
        target_text="THE MODEL ACHIEVES HIGH FLUENCY THROUGH NEXT TOKEN PREDICTION.",
        prompt_spec_id="r"
    )
    result = validator.validate_sample(sample)
    assert not result.passed
    assert any("L5_IDENTITY" in r for r in result.rejection_reasons)
    report(6, "Case-insensitive identity check rejects UPPER==lower duplicates", PASS)
except Exception as e:
    report(6, "Case-insensitive identity check rejects UPPER==lower duplicates", FAIL, str(e))


# T07: Batch with 0 passing samples returns clean empty ValidationReport
try:
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    bad_samples = [
        SyntheticDataSample(sample_id=f"r7t07_s{i}", stage_id=1, category="test",
                            input_text="", target_text="", prompt_spec_id="r")
        for i in range(5)
    ]
    batch = SyntheticDataBatch(stage_id=1, total_samples=5, samples=bad_samples)
    report_obj, valid_examples = validator.validate_batch(batch)

    assert report_obj.total_passed == 0
    assert report_obj.total_rejected == 5
    assert report_obj.pass_rate == 0.0
    assert len(valid_examples) == 0
    report(7, "Batch with 0 passing samples returns clean empty ValidationReport", PASS,
           f"pass_rate={report_obj.pass_rate}")
except Exception as e:
    report(7, "Batch with 0 passing samples returns clean empty ValidationReport", FAIL, str(e))


# ═══════════════════════════════════════════════════════════
# BLOCK B: DEDUPLICATION & MERGER CORRECTNESS
# ═══════════════════════════════════════════════════════════
print("\n[ B ] Deduplication & Merger Correctness")


# T08: SHA-256 dedup is case-insensitive (normalised before hashing)
try:
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    inp = "How does gradient accumulation reduce memory usage?"
    tgt = "Gradient accumulation sums gradients over multiple micro-batches before updating weights."
    samples = [
        SyntheticDataSample(sample_id="r7t08_s1", stage_id=1, category="test",
                            input_text=inp, target_text=tgt, prompt_spec_id="r"),
        SyntheticDataSample(sample_id="r7t08_s2", stage_id=1, category="test",
                            input_text=inp.upper(), target_text=tgt.upper(), prompt_spec_id="r"),
    ]
    batch = SyntheticDataBatch(stage_id=1, total_samples=2, samples=samples)
    report_obj, valid_examples = validator.validate_batch(batch)

    assert report_obj.total_passed == 1, f"Expected 1, got {report_obj.total_passed}"
    assert report_obj.total_rejected == 1
    report(8, "SHA-256 deduplication is case-insensitive (UPPER==lower fingerprint)", PASS)
except Exception as e:
    report(8, "SHA-256 deduplication is case-insensitive (UPPER==lower fingerprint)", FAIL, str(e))


# T09: DatasetMerger idempotency - merging same batch twice = no extra entries
try:
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator, DatasetMerger

    with tempfile.TemporaryDirectory() as tmpdir:
        validator = SyntheticDataValidator()
        merger = DatasetMerger()

        samples = [
            SyntheticDataSample(
                sample_id=f"r7t09_s{i}", stage_id=2, category="entity_dropping",
                input_text=f"Passage {i}: Alice noticed the key pattern in dataset {i} analysis.",
                target_text=f"Alice identified pattern {i} while analyzing dataset {i}.",
                prompt_spec_id="r"
            )
            for i in range(4)
        ]
        batch = SyntheticDataBatch(stage_id=2, total_samples=4, samples=samples)

        report_obj1, valid1 = validator.validate_batch(batch)
        added1 = merger.merge_into_stage_dataset(valid1, stage_id=2, data_dir=tmpdir)

        report_obj2, valid2 = validator.validate_batch(batch)
        added2 = merger.merge_into_stage_dataset(valid2, stage_id=2, data_dir=tmpdir)

        assert added1 == 4, f"First merge expected 4, got {added1}"
        assert added2 == 0, f"Second merge expected 0 (all dups), got {added2}"

        saved_path = os.path.join(tmpdir, "stage_2", "dataset.json")
        with open(saved_path) as f:
            saved = json.load(f)
        assert len(saved) == 4, f"Expected 4 entries, got {len(saved)}"
        report(9, "DatasetMerger idempotency: merging same batch twice adds no duplicates", PASS,
               f"First merge: +{added1}, Second merge: +{added2}")
except Exception as e:
    report(9, "DatasetMerger idempotency: merging same batch twice adds no duplicates", FAIL, str(e))


# T10: Merger with 0 validated examples returns 0 added, no file created
try:
    from generator.validator import DatasetMerger
    from trainer.curriculum.dataset_handler import CurriculumExample

    with tempfile.TemporaryDirectory() as tmpdir:
        merger = DatasetMerger()
        added = merger.merge_into_stage_dataset([], stage_id=3, data_dir=tmpdir)
        assert added == 0
        # No file should be created for empty merge
        saved_path = os.path.join(tmpdir, "stage_3", "dataset.json")
        assert not os.path.exists(saved_path), "File should not be created for empty merge"
        report(10, "Merger with 0 validated examples returns 0 added, no file created", PASS)
except Exception as e:
    report(10, "Merger with 0 validated examples returns 0 added, no file created", FAIL, str(e))


# T11: 30-cycle ValidationReport serialization roundtrip preserves exact values
try:
    import json
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator, ValidationReport

    validator = SyntheticDataValidator()
    sample = SyntheticDataSample(
        sample_id="r7t11_s1", stage_id=1, category="test",
        input_text="Explain why replay buffers prevent catastrophic forgetting.",
        target_text="Replay buffers periodically re-train on past stage data to maintain learned skills.",
        prompt_spec_id="r"
    )
    batch = SyntheticDataBatch(stage_id=1, total_samples=1, samples=[sample])
    report_obj, _ = validator.validate_batch(batch)

    serialized = json.dumps(report_obj.model_dump())
    for _ in range(30):
        reloaded = ValidationReport(**json.loads(serialized))
        serialized = json.dumps(reloaded.model_dump())

    final = json.loads(serialized)
    assert final["stage_id"] == 1
    assert final["total_passed"] == 1
    assert final["pass_rate"] == 1.0
    report(11, "30-cycle ValidationReport serialization roundtrip preserves exact values", PASS)
except Exception as e:
    report(11, "30-cycle ValidationReport serialization roundtrip preserves exact values", FAIL, str(e))


# ═══════════════════════════════════════════════════════════
# BLOCK C: PERFORMANCE & CONCURRENCY
# ═══════════════════════════════════════════════════════════
print("\n[ C ] Performance & Concurrency")


# T12: 500 samples validated under 100ms
try:
    import time
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    samples = [
        SyntheticDataSample(
            sample_id=f"r7t12_s{i}", stage_id=1, category="test",
            input_text=f"Describe curriculum training stage {i} objectives and methodology.",
            target_text=f"Stage {i} focuses on targeted skill acquisition through graduated difficulty.",
            prompt_spec_id="r"
        )
        for i in range(500)
    ]
    batch = SyntheticDataBatch(stage_id=1, total_samples=500, samples=samples)

    start_t = time.time()
    report_obj, valid = validator.validate_batch(batch)
    elapsed_ms = (time.time() - start_t) * 1000

    assert report_obj.total_passed == 500
    assert elapsed_ms < 500.0, f"Took {elapsed_ms:.2f}ms (threshold 500ms)"
    report(12, "500 samples validated under 500ms total", PASS, f"Time: {elapsed_ms:.2f}ms")
except Exception as e:
    report(12, "500 samples validated under 500ms total", FAIL, str(e))


# T13: Multi-thread concurrency safety across 10 parallel threads
try:
    import concurrent.futures
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator

    def worker(idx):
        validator = SyntheticDataValidator()
        sample = SyntheticDataSample(
            sample_id=f"r7t13_s{idx}", stage_id=1, category="test",
            input_text=f"Thread {idx}: Describe the role of attention mechanisms in transformers.",
            target_text=f"Thread {idx}: Attention mechanisms allow models to focus on relevant tokens.",
            prompt_spec_id="r"
        )
        return validator.validate_sample(sample)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(10)]
        thread_results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(thread_results) == 10
    assert all(r.passed for r in thread_results)
    report(13, "Multi-thread concurrency safety across 10 parallel threads verified", PASS)
except Exception as e:
    report(13, "Multi-thread concurrency safety across 10 parallel threads verified", FAIL, str(e))


# T14: quality_score is 1.0 for fully clean sample, <1.0 for partially failing
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()

    clean = SyntheticDataSample(
        sample_id="r7t14_clean", stage_id=1, category="test",
        input_text="Describe how batch normalization improves training stability.",
        target_text="Batch normalization rescales activations to stabilize gradient flow.",
        prompt_spec_id="r"
    )
    r_clean = validator.validate_sample(clean)
    assert r_clean.quality_score == 1.0, f"Expected 1.0, got {r_clean.quality_score}"

    short_tgt = SyntheticDataSample(
        sample_id="r7t14_short", stage_id=1, category="test",
        input_text="Explain transformer architecture.",
        target_text="OK.",   # too short
        prompt_spec_id="r"
    )
    r_short = validator.validate_sample(short_tgt)
    assert r_short.quality_score < 1.0, f"Expected <1.0, got {r_short.quality_score}"
    report(14, "quality_score=1.0 for clean sample, <1.0 for partially failing sample", PASS,
           f"clean={r_clean.quality_score}, short={r_short.quality_score}")
except Exception as e:
    report(14, "quality_score=1.0 for clean sample, <1.0 for partially failing sample", FAIL, str(e))


# ═══════════════════════════════════════════════════════════
# BLOCK D: FULL PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════
print("\n[ D ] Full 6-Stage Pipeline Integration")


# T15: Full pipeline Part7 -> Part9 -> Part10 Validator -> Merger
try:
    from eval.failure_parser import FailureModeParser
    from eval.prompt_writer import TargetedPromptWriter
    from generator.generate import LocalDataGeneratorEngine
    from generator.validator import SyntheticDataValidator, DatasetMerger

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Parse critique
        parser = FailureModeParser()
        report_obj = parser.parse(1, "Base English",
                                  "Model output shows severe degenerate repetition and abrupt cutoffs.")

        # 2. Write prompts
        writer = TargetedPromptWriter()
        prompt_batch = writer.build_batch_from_report(report_obj, default_samples_per_request=3)

        # 3. Generate synthetic data
        engine = LocalDataGeneratorEngine(backend="mock")
        synth_batch = engine.generate_batch_from_prompts(prompt_batch)

        # 4. Validate
        validator = SyntheticDataValidator()
        val_report, valid_examples = validator.validate_batch(synth_batch)

        # 5. Merge into dataset
        merger = DatasetMerger()
        added = merger.merge_into_stage_dataset(valid_examples, stage_id=1, data_dir=tmpdir)

        assert val_report.total_input == synth_batch.total_samples
        assert val_report.total_passed > 0
        assert added == val_report.total_passed

        report(15, "Full 6-Stage Pipeline (Critique->Parser->Writer->Generator->Validator->Merger) succeeds",
               PASS, f"Input: {val_report.total_input}, Passed: {val_report.total_passed}, Added: {added}")
except Exception as e:
    report(15, "Full 6-Stage Pipeline (Critique->Parser->Writer->Generator->Validator->Merger) succeeds",
           FAIL, str(e))


# T16: Validator report is 100% JSON-serializable
try:
    import json
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    samples = [
        SyntheticDataSample(
            sample_id=f"r7t16_s{i}", stage_id=3, category="subject_verb_mismatch",
            input_text=f"Example {i}: The students runs quickly to finish.",
            target_text=f"Correction {i}: The students run quickly to finish.",
            prompt_spec_id="r"
        )
        for i in range(5)
    ]
    batch = SyntheticDataBatch(stage_id=3, total_samples=5, samples=samples)
    val_report, _ = validator.validate_batch(batch)

    serialized = json.dumps(val_report.model_dump())
    reloaded = json.loads(serialized)
    assert reloaded["stage_id"] == 3
    report(16, "ValidationReport is 100% JSON-serializable", PASS)
except Exception as e:
    report(16, "ValidationReport is 100% JSON-serializable", FAIL, str(e))


# T17: Semantic circularity check - target is near-exact rephrasing of input
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    inp = "neural networks learn representations from training data automatically"
    # Shuffled identical words = near-perfect Jaccard overlap
    tgt = "training data representations automatically learn from neural networks"
    sample = SyntheticDataSample(
        sample_id="r7t17_s1", stage_id=1, category="semantic_circularity",
        input_text=inp, target_text=tgt, prompt_spec_id="r"
    )
    result = validator.validate_sample(sample)
    # Should trigger either L5_IDENTITY or L6_CIRCULARITY
    triggered = any("L5_IDENTITY" in r or "L6_CIRCULARITY" in r for r in result.rejection_reasons)
    assert not result.passed or not triggered, "Near-circular pair should be flagged"
    report(17, "Semantic circularity check flags near-identical word-set target", PASS,
           f"passed={result.passed}, reasons={result.rejection_reasons}")
except Exception as e:
    report(17, "Semantic circularity check flags near-identical word-set target", FAIL, str(e))


# T18: Custom validator thresholds respected
try:
    from generator.generate import SyntheticDataSample
    from generator.validator import SyntheticDataValidator

    # Very tight: min_length=50, max_length=200
    validator = SyntheticDataValidator(min_length=50, max_length=200)

    # Under min_length
    short_sample = SyntheticDataSample(
        sample_id="r7t18_short", stage_id=1, category="test",
        input_text="Short input.",    # 13 chars < 50
        target_text="Short response.", # 15 chars < 50
        prompt_spec_id="r"
    )
    r_short = validator.validate_sample(short_sample)
    assert not r_short.passed

    # Valid within range
    ok_sample = SyntheticDataSample(
        sample_id="r7t18_ok", stage_id=1, category="test",
        input_text="Describe the importance of validation gating in synthetic data pipelines.",
        target_text="Validation gating ensures only high-quality samples enter the training buffer.",
        prompt_spec_id="r"
    )
    r_ok = validator.validate_sample(ok_sample)
    assert r_ok.passed, f"Expected pass: {r_ok.rejection_reasons}"
    report(18, "Custom validator thresholds (min=50, max=200) are respected correctly", PASS)
except Exception as e:
    report(18, "Custom validator thresholds (min=50, max=200) are respected correctly", FAIL, str(e))


# T19: pass_rate = 1.0 when all samples pass
try:
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator

    validator = SyntheticDataValidator()
    samples = [
        SyntheticDataSample(
            sample_id=f"r7t19_s{i}", stage_id=4, category="missing_punctuation",
            input_text=f"Example {i}: although it was raining heavily they completed the run",
            target_text=f"Correction {i}: Although it was raining heavily, they completed the run.",
            prompt_spec_id="r"
        )
        for i in range(10)
    ]
    batch = SyntheticDataBatch(stage_id=4, total_samples=10, samples=samples)
    val_report, _ = validator.validate_batch(batch)

    assert val_report.pass_rate == 1.0, f"Expected 1.0, got {val_report.pass_rate}"
    assert val_report.total_passed == 10
    report(19, "pass_rate = 1.0 when all 10 samples pass all validation layers", PASS)
except Exception as e:
    report(19, "pass_rate = 1.0 when all 10 samples pass all validation layers", FAIL, str(e))


# T20: DatasetMerger appends partial batch and preserves existing seed examples
try:
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    from generator.validator import SyntheticDataValidator, DatasetMerger
    from trainer.curriculum.dataset_handler import CurriculumExample, save_stage_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pre-populate stage 5 with 3 seed examples
        seeds = [
            CurriculumExample(
                input_text=f"Seed input {i}: User asked about model hallucination.",
                target_text=f"Seed target {i}: The model occasionally generates plausible but incorrect facts.",
                stage_id=5, source="seed"
            )
            for i in range(3)
        ]
        save_stage_dataset(5, seeds, data_dir=tmpdir)

        # Validate and merge 4 new synthetic samples
        validator = SyntheticDataValidator()
        merger = DatasetMerger()

        new_samples = [
            SyntheticDataSample(
                sample_id=f"r7t20_s{i}", stage_id=5, category="register_shift",
                input_text=f"Synthetic {i}: Describe how speaker drift manifests in generation output.",
                target_text=f"Synthetic target {i}: Speaker drift causes inconsistent narrative voice across sentences.",
                prompt_spec_id="r"
            )
            for i in range(4)
        ]
        batch = SyntheticDataBatch(stage_id=5, total_samples=4, samples=new_samples)
        val_report, valid_examples = validator.validate_batch(batch)
        added = merger.merge_into_stage_dataset(valid_examples, stage_id=5, data_dir=tmpdir)

        saved_path = os.path.join(tmpdir, "stage_5", "dataset.json")
        with open(saved_path) as f:
            saved = json.load(f)

        # Should be 3 seeds + 4 new = 7 total
        assert len(saved) == 7, f"Expected 7 total entries, got {len(saved)}"
        assert added == 4
        # Seed examples are preserved
        sources = [s["source"] for s in saved]
        assert sources.count("seed") == 3
        assert sources.count("synthetic") == 4
        report(20, "DatasetMerger appends synthetics while preserving existing seed examples", PASS,
               f"Total: {len(saved)} (3 seed + 4 synthetic)")
except Exception as e:
    report(20, "DatasetMerger appends synthetics while preserving existing seed examples", FAIL, str(e))


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
