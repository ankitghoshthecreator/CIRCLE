"""
tests/test_part7.py
Part 7 Unit Test Suite: Targeted Prompt Writer

Tests meta-prompt template coverage, prompt spec generation, batching,
priority sorting, Pydantic JSON serialization, and real critique integration.
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
print("  CIRCLE - Part 7 Unit Tests (Targeted Prompt Writer)")
print("="*65 + "\n")

# ─── TEST 1: All FailureCategory values have an entry in META_PROMPT_TEMPLATES ───
try:
    from eval.failure_parser import FailureCategory
    from eval.prompt_writer import META_PROMPT_TEMPLATES
    missing = [c.value for c in FailureCategory if c.value not in META_PROMPT_TEMPLATES]
    assert not missing, f"Categories missing meta-prompt templates: {missing}"
    report(1, "All FailureCategory values have meta-prompt templates", PASS,
           f"{len(META_PROMPT_TEMPLATES)} templates registered")
except Exception as e:
    report(1, "All FailureCategory values have meta-prompt templates", FAIL, str(e))

# ─── TEST 2: get_template returns category template or falls back to OTHER ───
try:
    from eval.prompt_writer import TargetedPromptWriter, META_PROMPT_TEMPLATES, FailureCategory
    writer = TargetedPromptWriter()
    tmpl_rep = writer.get_template(FailureCategory.DEGENERATE_REPETITION.value)
    assert "repeating identical n-grams" in tmpl_rep

    tmpl_unknown = writer.get_template("non_existent_category_xyz")
    assert tmpl_unknown == META_PROMPT_TEMPLATES[FailureCategory.OTHER.value]
    report(2, "get_template returns correct template and falls back to OTHER", PASS)
except Exception as e:
    report(2, "get_template returns correct template and falls back to OTHER", FAIL, str(e))

# ─── TEST 3: generate_prompt_spec creates valid TargetedPromptSpec ───
try:
    from eval.failure_parser import DataCommissionRequest
    from eval.prompt_writer import TargetedPromptWriter
    writer = TargetedPromptWriter()
    req = DataCommissionRequest(
        stage_id=1,
        failure_category="degenerate_repetition",
        severity="high",
        suggested_prompt_focus="Avoid repeating token loops in long generations",
        priority=1
    )
    spec = writer.generate_prompt_spec(req, idx=1, num_samples=15)
    assert spec.request_id == "stage_1_req_01_degenerate_repetition"
    assert spec.priority == 1
    assert spec.target_sample_count == 15
    assert "Avoid repeating token loops" in spec.meta_prompt
    report(3, "generate_prompt_spec constructs valid TargetedPromptSpec", PASS,
           f"Meta prompt: '{spec.meta_prompt[:60]}...'")
except Exception as e:
    report(3, "generate_prompt_spec constructs valid TargetedPromptSpec", FAIL, str(e))

# ─── TEST 4: build_batch_from_report builds TargetedPromptBatch sorted by priority ───
try:
    from eval.failure_parser import FailureModeReport, DataCommissionRequest, Severity
    from eval.prompt_writer import TargetedPromptWriter
    writer = TargetedPromptWriter()
    report_obj = FailureModeReport(
        stage_id=2,
        stage_name="Summarization",
        failure_modes=[],
        overall_severity=Severity.HIGH,
        advance_to_next_stage=False,
        commission_requests=[
            DataCommissionRequest(
                stage_id=2,
                failure_category="missing_punctuation",
                severity="low",
                suggested_prompt_focus="Punctuation details",
                priority=5
            ),
            DataCommissionRequest(
                stage_id=2,
                failure_category="incoherent_continuation",
                severity="critical",
                suggested_prompt_focus="Logic and coherence",
                priority=1
            ),
            DataCommissionRequest(
                stage_id=2,
                failure_category="speaker_drift",
                severity="high",
                suggested_prompt_focus="Speaker tracking",
                priority=2
            )
        ]
    )
    batch = writer.build_batch_from_report(report_obj, default_samples_per_request=10)
    assert batch.total_requests == 3
    priorities = [s.priority for s in batch.prompt_specs]
    assert priorities == sorted(priorities), f"Specs not sorted by priority: {priorities}"
    assert batch.prompt_specs[0].priority == 1
    assert batch.prompt_specs[0].category == "incoherent_continuation"
    report(4, "build_batch_from_report sorts specs by priority (1=highest)", PASS,
           f"Priorities order: {priorities}")
except Exception as e:
    report(4, "build_batch_from_report sorts specs by priority (1=highest)", FAIL, str(e))

# ─── TEST 5: Priority multiplier scales target sample count ───
try:
    from eval.failure_parser import FailureModeReport, DataCommissionRequest, Severity
    from eval.prompt_writer import TargetedPromptWriter
    writer = TargetedPromptWriter()
    report_obj = FailureModeReport(
        stage_id=1,
        stage_name="Base English",
        failure_modes=[],
        overall_severity=Severity.HIGH,
        advance_to_next_stage=False,
        commission_requests=[
            DataCommissionRequest(stage_id=1, failure_category="abrupt_cutoff", severity="high", suggested_prompt_focus="cutoff", priority=1),
            DataCommissionRequest(stage_id=1, failure_category="missing_punctuation", severity="low", suggested_prompt_focus="punc", priority=5)
        ]
    )
    batch = writer.build_batch_from_report(report_obj, default_samples_per_request=10)
    high_priority_spec = next(s for s in batch.prompt_specs if s.priority == 1)
    low_priority_spec = next(s for s in batch.prompt_specs if s.priority == 5)
    assert high_priority_spec.target_sample_count > low_priority_spec.target_sample_count, \
        f"High priority ({high_priority_spec.target_sample_count}) should have more samples than low priority ({low_priority_spec.target_sample_count})"
    report(5, "Priority multiplier allocates more samples to higher priority requests", PASS,
           f"P1: {high_priority_spec.target_sample_count} vs P5: {low_priority_spec.target_sample_count}")
except Exception as e:
    report(5, "Priority multiplier allocates more samples to higher priority requests", FAIL, str(e))

# ─── TEST 6: save_prompt_batch and load_prompt_batch JSON I/O roundtrip ───
try:
    from eval.failure_parser import DataCommissionRequest, FailureModeReport, Severity
    from eval.prompt_writer import TargetedPromptWriter
    writer = TargetedPromptWriter()
    report_obj = FailureModeReport(
        stage_id=1,
        stage_name="Base English",
        failure_modes=[],
        overall_severity=Severity.HIGH,
        advance_to_next_stage=False,
        commission_requests=[
            DataCommissionRequest(stage_id=1, failure_category="entity_dropping", severity="high", suggested_prompt_focus="Entity tracking", priority=2)
        ]
    )
    batch = writer.build_batch_from_report(report_obj)
    test_dir = "./eval/reports_test_p7"
    out_path = os.path.join(test_dir, "targeted_prompts_test.json")
    saved_path = writer.save_prompt_batch(batch, out_path)
    assert os.path.exists(saved_path)

    loaded_batch = writer.load_prompt_batch(saved_path)
    assert loaded_batch.stage_id == batch.stage_id
    assert loaded_batch.total_requests == batch.total_requests
    assert loaded_batch.prompt_specs[0].category == "entity_dropping"

    shutil.rmtree(test_dir, ignore_errors=True)
    report(6, "save_prompt_batch and load_prompt_batch JSON I/O roundtrip succeeds", PASS)
except Exception as e:
    shutil.rmtree("./eval/reports_test_p7", ignore_errors=True)
    report(6, "save_prompt_batch and load_prompt_batch JSON I/O roundtrip succeeds", FAIL, str(e))

# ─── TEST 7: End-to-end integration with real stage_1_critique.json ───
try:
    from eval.failure_parser import parse_critique_report
    from eval.prompt_writer import TargetedPromptWriter
    critique_path = "./eval/reports/stage_1_critique.json"
    if not os.path.exists(critique_path):
        raise FileNotFoundError(f"Missing {critique_path}")

    report_obj = parse_critique_report(critique_path)
    writer = TargetedPromptWriter()
    batch = writer.build_batch_from_report(report_obj, default_samples_per_request=10)

    out_path = "./eval/reports/targeted_prompts_stage_1.json"
    writer.save_prompt_batch(batch, out_path)
    assert os.path.exists(out_path)
    report(7, "Synthesizes targeted_prompts_stage_1.json from real critique report", PASS,
           f"Specs: {batch.total_requests}, Total target samples: {batch.total_target_samples}")
except Exception as e:
    report(7, "Synthesizes targeted_prompts_stage_1.json from real critique report", FAIL, str(e))

# ─── TEST 8: Custom template override in TargetedPromptWriter constructor ───
try:
    from eval.failure_parser import DataCommissionRequest
    from eval.prompt_writer import TargetedPromptWriter
    custom = {"degenerate_repetition": "CUSTOM TEMPLATE: {focus_hint}"}
    writer = TargetedPromptWriter(custom_templates=custom)
    req = DataCommissionRequest(stage_id=1, failure_category="degenerate_repetition", severity="high", suggested_prompt_focus="Custom focus", priority=1)
    spec = writer.generate_prompt_spec(req)
    assert spec.meta_prompt == "CUSTOM TEMPLATE: Custom focus"
    report(8, "Custom template overrides work cleanly in TargetedPromptWriter", PASS)
except Exception as e:
    report(8, "Custom template overrides work cleanly in TargetedPromptWriter", FAIL, str(e))

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
