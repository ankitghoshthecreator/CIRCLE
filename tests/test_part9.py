"""
tests/test_part9.py
Part 9 Unit Test Suite: Local DeepSeek Data Generator Engine

Tests synthetic data sample creation, batch execution from Part 7 prompt specs,
JSON serialization roundtrips, offline mock fallbacks, and CurriculumExample conversions.
"""

import os
import sys
import json
import shutil
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

PASS, FAIL = "PASS", "FAIL"
results = []

def report(tid, name, status, detail=""):
    icon = "OK" if status == PASS else "XX"
    print(f"  [{icon}] Test {tid:02d}: {name} - {status}" + (f"\n         Detail: {detail}" if detail else ""))
    results.append((tid, name, status, detail))

print("\n" + "="*65)
print("  CIRCLE - Part 9 Unit Tests (Local Data Generator Engine)")
print("="*65 + "\n")

# ─── TEST 1: SyntheticDataSample and SyntheticDataBatch model validation ───
try:
    from generator.generate import SyntheticDataSample, SyntheticDataBatch
    sample = SyntheticDataSample(
        sample_id="synth_s1_001",
        stage_id=1,
        category="speaker_drift",
        input_text="Sample input prompt",
        target_text="Sample target completion text",
        prompt_spec_id="req_01"
    )
    batch = SyntheticDataBatch(stage_id=1, total_samples=1, samples=[sample], generation_time_sec=0.1)
    assert batch.total_samples == 1
    assert batch.samples[0].category == "speaker_drift"
    report(1, "SyntheticDataSample & SyntheticDataBatch Pydantic validation succeeds", PASS)
except Exception as e:
    report(1, "SyntheticDataSample & SyntheticDataBatch Pydantic validation succeeds", FAIL, str(e))

# ─── TEST 2: LocalDataGeneratorEngine endpoint availability check ───
try:
    from generator.generate import LocalDataGeneratorEngine
    engine = LocalDataGeneratorEngine(endpoint_url="http://localhost:59999")  # Unused port
    is_avail = engine.is_endpoint_available()
    assert is_avail is False
    report(2, "is_endpoint_available returns False cleanly for offline endpoint", PASS)
except Exception as e:
    report(2, "is_endpoint_available returns False cleanly for offline endpoint", FAIL, str(e))

# ─── TEST 3: generate_samples_for_spec creates valid samples ───
try:
    from generator.generate import LocalDataGeneratorEngine
    from eval.prompt_writer import TargetedPromptSpec
    spec = TargetedPromptSpec(
        request_id="stage_1_req_01_degenerate_repetition",
        stage_id=1,
        category="degenerate_repetition",
        severity="high",
        priority=1,
        meta_prompt="Generate clean text without n-gram loops.",
        suggested_prompt_focus="Avoid repetition",
        target_sample_count=3
    )
    engine = LocalDataGeneratorEngine()
    samples = engine.generate_samples_for_spec(spec, num_samples=3)
    assert len(samples) == 3
    assert all(s.category == "degenerate_repetition" for s in samples)
    assert all(len(s.input_text) > 0 and len(s.target_text) > 0 for s in samples)
    report(3, "generate_samples_for_spec produces valid paired synthetic samples", PASS,
           f"Generated {len(samples)} samples")
except Exception as e:
    report(3, "generate_samples_for_spec produces valid paired synthetic samples", FAIL, str(e))

# ─── TEST 4: generate_batch_from_prompts creates SyntheticDataBatch from TargetedPromptBatch ───
try:
    from generator.generate import LocalDataGeneratorEngine
    from eval.prompt_writer import TargetedPromptSpec, TargetedPromptBatch
    spec1 = TargetedPromptSpec(request_id="r1", stage_id=2, category="abrupt_cutoff", severity="high", priority=1, meta_prompt="p1", suggested_prompt_focus="f1", target_sample_count=2)
    spec2 = TargetedPromptSpec(request_id="r2", stage_id=2, category="speaker_drift", severity="high", priority=2, meta_prompt="p2", suggested_prompt_focus="f2", target_sample_count=2)
    prompt_batch = TargetedPromptBatch(stage_id=2, stage_name="Summarization", total_requests=2, total_target_samples=4, prompt_specs=[spec1, spec2])

    engine = LocalDataGeneratorEngine()
    synth_batch = engine.generate_batch_from_prompts(prompt_batch)
    assert synth_batch.stage_id == 2
    assert synth_batch.total_samples == 4
    assert len(synth_batch.samples) == 4
    report(4, "generate_batch_from_prompts creates SyntheticDataBatch from prompt specs", PASS,
           f"Total samples: {synth_batch.total_samples}, time: {synth_batch.generation_time_sec}s")
except Exception as e:
    report(4, "generate_batch_from_prompts creates SyntheticDataBatch from prompt specs", FAIL, str(e))

# ─── TEST 5: save_synthetic_batch and load_synthetic_batch JSON roundtrip ───
try:
    from generator.generate import LocalDataGeneratorEngine, SyntheticDataSample, SyntheticDataBatch
    engine = LocalDataGeneratorEngine()
    sample = SyntheticDataSample(sample_id="s1", stage_id=1, category="test", input_text="inp", target_text="tgt", prompt_spec_id="r1")
    batch = SyntheticDataBatch(stage_id=1, total_samples=1, samples=[sample], generation_time_sec=0.05)

    test_path = "./data/synthetic_test_roundtrip.json"
    engine.save_synthetic_batch(batch, test_path)
    assert os.path.exists(test_path)

    loaded = engine.load_synthetic_batch(test_path)
    assert loaded.stage_id == 1
    assert loaded.total_samples == 1
    assert loaded.samples[0].input_text == "inp"

    shutil.rmtree("./data/synthetic_test_roundtrip.json", ignore_errors=True)
    if os.path.exists(test_path):
        os.remove(test_path)
    report(5, "save_synthetic_batch & load_synthetic_batch JSON roundtrip succeeds", PASS)
except Exception as e:
    if os.path.exists("./data/synthetic_test_roundtrip.json"):
        os.remove("./data/synthetic_test_roundtrip.json")
    report(5, "save_synthetic_batch & load_synthetic_batch JSON roundtrip succeeds", FAIL, str(e))

# ─── TEST 6: convert_to_curriculum_examples creates CurriculumExample list ───
try:
    from generator.generate import LocalDataGeneratorEngine, SyntheticDataSample, SyntheticDataBatch
    engine = LocalDataGeneratorEngine()
    sample = SyntheticDataSample(sample_id="s1", stage_id=1, category="speaker_drift", input_text="I spoke", target_text="I replied", prompt_spec_id="r1")
    batch = SyntheticDataBatch(stage_id=1, total_samples=1, samples=[sample])

    curr_examples = engine.convert_to_curriculum_examples(batch)
    assert len(curr_examples) == 1
    ex = curr_examples[0]
    assert ex.input_text == "I spoke"
    assert ex.target_text == "I replied"
    assert ex.stage_id == 1
    assert ex.source == "synthetic"
    assert ex.metadata["category"] == "speaker_drift"
    report(6, "convert_to_curriculum_examples produces valid CurriculumExample list with source='synthetic'", PASS)
except Exception as e:
    report(6, "convert_to_curriculum_examples produces valid CurriculumExample list with source='synthetic'", FAIL, str(e))

# ─── TEST 7: Full integration Part 7 Prompt Batch -> Part 9 Data Generation -> Curriculum Examples ───
try:
    from eval.prompt_writer import TargetedPromptWriter
    from eval.failure_parser import FailureModeReport, DataCommissionRequest, Severity
    from generator.generate import LocalDataGeneratorEngine

    writer = TargetedPromptWriter()
    report_obj = FailureModeReport(
        stage_id=1, stage_name="Base English", failure_modes=[], overall_severity=Severity.HIGH, advance_to_next_stage=False,
        commission_requests=[DataCommissionRequest(stage_id=1, failure_category="abrupt_cutoff", severity="high", suggested_prompt_focus="cutoff", priority=1)]
    )
    prompt_batch = writer.build_batch_from_report(report_obj, default_samples_per_request=2)

    engine = LocalDataGeneratorEngine()
    synth_batch = engine.generate_batch_from_prompts(prompt_batch)
    curr_examples = engine.convert_to_curriculum_examples(synth_batch)

    assert len(curr_examples) > 0
    assert curr_examples[0].source == "synthetic"
    report(7, "Full Part 7 -> Part 9 -> CurriculumExample integration pipeline succeeds", PASS,
           f"Generated {len(curr_examples)} CurriculumExamples")
except Exception as e:
    report(7, "Full Part 7 -> Part 9 -> CurriculumExample integration pipeline succeeds", FAIL, str(e))

# ─── TEST 8: Mock fallback generation for all failure categories ───
try:
    from generator.generate import LocalDataGeneratorEngine
    from eval.failure_parser import FailureCategory
    from eval.prompt_writer import TargetedPromptSpec
    engine = LocalDataGeneratorEngine()

    categories = list(FailureCategory)
    all_valid = True
    for cat in categories:
        spec = TargetedPromptSpec(request_id=f"req_{cat.value}", stage_id=1, category=cat.value, severity="medium", priority=2, meta_prompt="m", suggested_prompt_focus="f", target_sample_count=1)
        samples = engine.generate_samples_for_spec(spec, num_samples=1)
        if len(samples) != 1 or len(samples[0].input_text) == 0 or len(samples[0].target_text) == 0:
            all_valid = False
            break

    assert all_valid, "Mock fallback generation failed for some failure categories!"
    report(8, "Mock fallback generator produces valid pairs across all 17 FailureCategory values", PASS,
           f"Tested {len(categories)} categories")
except Exception as e:
    report(8, "Mock fallback generator produces valid pairs across all 17 FailureCategory values", FAIL, str(e))

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
