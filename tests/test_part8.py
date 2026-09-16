"""
tests/test_part8.py
Part 8 Unit Test Suite: Lightweight Student Distillation Evaluator

Tests sub-5ms heuristic evaluation, fluency/syntax/semantics/vocabulary metrics,
risk prediction, Pydantic report serialization, and TieredEvaluator routing.
"""

import os
import sys
import json
import time
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
print("  CIRCLE - Part 8 Unit Tests (Distilled Student Evaluator)")
print("="*65 + "\n")

# ─── TEST 1: High-quality text receives high score and low risk ───
try:
    from eval.distilled_eval import LightweightStudentEvaluator
    evaluator = LightweightStudentEvaluator()
    prompt = "Summarize the primary benefits of renewable solar energy."
    good_text = (
        "Solar energy provides a clean, sustainable source of power that significantly "
        "reduces greenhouse gas emissions. Photovoltaic systems convert sunlight directly into "
        "electricity, lowering long-term utility costs for homes and businesses."
    )
    score = evaluator.evaluate_text(prompt, good_text)
    assert score.overall_score >= 0.70, f"Expected overall score >= 0.70, got {score.overall_score}"
    assert score.predicted_risk == "low", f"Expected low risk, got {score.predicted_risk}"
    report(1, "High-quality text receives high overall score and low risk", PASS,
           f"Score: {score.overall_score}, Risk: {score.predicted_risk}")
except Exception as e:
    report(1, "High-quality text receives high overall score and low risk", FAIL, str(e))

# ─── TEST 2: Severe token loops trigger severe_token_loop flag and critical risk ───
try:
    from eval.distilled_eval import LightweightStudentEvaluator
    evaluator = LightweightStudentEvaluator()
    prompt = "Explain quantum computing."
    loop_text = "Quantum computing is fast quantum computing is fast quantum computing is fast quantum computing is fast."
    score = evaluator.evaluate_text(prompt, loop_text)
    assert "severe_token_loop" in score.detected_flags or "high_repetition" in score.detected_flags
    assert score.predicted_risk in ("high", "critical"), f"Expected high/critical risk, got {score.predicted_risk}"
    report(2, "Severe token loops trigger repetition flag and high/critical risk", PASS,
           f"Flags: {score.detected_flags}, Risk: {score.predicted_risk}")
except Exception as e:
    report(2, "Severe token loops trigger repetition flag and high/critical risk", FAIL, str(e))

# ─── TEST 3: Abrupt cutoff detected when terminal punctuation is missing ───
try:
    from eval.distilled_eval import LightweightStudentEvaluator
    evaluator = LightweightStudentEvaluator()
    prompt = "Write a sentence about climate change."
    cutoff_text = "The rising global temperatures are causing ice caps to melt rapidly in the polar"
    score = evaluator.evaluate_text(prompt, cutoff_text)
    assert "abrupt_cutoff" in score.detected_flags, f"Expected abrupt_cutoff flag, got {score.detected_flags}"
    report(3, "Missing terminal punctuation triggers abrupt_cutoff flag", PASS)
except Exception as e:
    report(3, "Missing terminal punctuation triggers abrupt_cutoff flag", FAIL, str(e))

# ─── TEST 4: Semantic circularity detected when output restates prompt ───
try:
    from eval.distilled_eval import LightweightStudentEvaluator
    evaluator = LightweightStudentEvaluator()
    prompt = "What is artificial intelligence and deep learning?"
    circular_text = "Artificial intelligence and deep learning is artificial intelligence and deep learning."
    score = evaluator.evaluate_text(prompt, circular_text)
    assert "semantic_circularity" in score.detected_flags or "severe_token_loop" in score.detected_flags
    report(4, "Circular restatement of prompt triggers semantic flag", PASS)
except Exception as e:
    report(4, "Circular restatement of prompt triggers semantic flag", FAIL, str(e))

# ─── TEST 5: Pydantic serialization / deserialization roundtrip for DistilledEvalReport ───
try:
    from eval.distilled_eval import LightweightStudentEvaluator, DistilledEvalReport
    evaluator = LightweightStudentEvaluator()
    probes = [
        {"prompt": "Prompt 1", "output": "Clear concise output statement."},
        {"prompt": "Prompt 2", "output": "Repetitive output statement statement statement statement."}
    ]
    report_obj = evaluator.evaluate_batch(1, "Base English", probes)
    dumped = report_obj.model_dump()
    serialized = json.dumps(dumped)
    reloaded_dict = json.loads(serialized)
    reloaded_report = DistilledEvalReport(**reloaded_dict)
    assert reloaded_report.stage_id == 1
    assert reloaded_report.total_probes == 2
    report(5, "DistilledEvalReport Pydantic model serialization roundtrip succeeds", PASS)
except Exception as e:
    report(5, "DistilledEvalReport Pydantic model serialization roundtrip succeeds", FAIL, str(e))

# ─── TEST 6: TieredEvaluator fast pass (bypasses teacher for good outputs) ───
try:
    from eval.distilled_eval import TieredEvaluator
    tiered = TieredEvaluator(escalation_threshold=0.70)
    good_probes = [
        {"prompt": "Describe machine learning.", "output": "Machine learning enables systems to learn from data patterns automatically."},
        {"prompt": "Define neural networks.", "output": "Neural networks are computing systems inspired by biological neural networks."}
    ]
    res = tiered.evaluate_stage(1, "Base English", ["fluency"], good_probes)
    assert res["escalated_to_teacher"] is False
    assert res["tiered_eval_mode"] == "student_fast_pass"
    report(6, "TieredEvaluator fast pass bypasses teacher for high-quality outputs", PASS,
           f"Mode: {res['tiered_eval_mode']}")
except Exception as e:
    report(6, "TieredEvaluator fast pass bypasses teacher for high-quality outputs", FAIL, str(e))

# ─── TEST 7: TieredEvaluator escalation trigger (escalates low quality to teacher) ───
try:
    from eval.distilled_eval import TieredEvaluator
    tiered = TieredEvaluator(escalation_threshold=0.70)
    bad_probes = [
        {"prompt": "Describe machine learning.", "output": "loop loop loop loop loop loop loop loop loop loop."}
    ]
    res = tiered.evaluate_stage(1, "Base English", ["fluency"], bad_probes)
    assert res["escalated_to_teacher"] is True
    assert res["tiered_eval_mode"] == "teacher_deep_critique"
    assert res["failure_mode_report"] is not None
    report(7, "TieredEvaluator escalates low-quality outputs to teacher critique", PASS,
           f"Mode: {res['tiered_eval_mode']}")
except Exception as e:
    report(7, "TieredEvaluator escalates low-quality outputs to teacher critique", FAIL, str(e))

# ─── TEST 8: Sub-5ms latency benchmark across 50 probes ───
try:
    from eval.distilled_eval import LightweightStudentEvaluator
    evaluator = LightweightStudentEvaluator()
    batch_probes = [
        {"prompt": f"Prompt {i}", "output": f"Sample response sentence number {i} with varied words and clear concluding punctuation."}
        for i in range(50)
    ]
    start_t = time.time()
    batch_report = evaluator.evaluate_batch(1, "Benchmark", batch_probes)
    elapsed_ms = (time.time() - start_t) * 1000

    assert elapsed_ms < 50, f"Batch evaluation of 50 probes took {elapsed_ms:.2f}ms (threshold 50ms)"
    per_probe_ms = elapsed_ms / 50
    assert per_probe_ms < 5.0, f"Per-probe evaluation took {per_probe_ms:.2f}ms (threshold 5.0ms)"
    report(8, "Student evaluator latency benchmark: sub-5ms per probe", PASS,
           f"Total 50 probes: {elapsed_ms:.2f}ms ({per_probe_ms:.3f}ms/probe)")
except Exception as e:
    report(8, "Student evaluator latency benchmark: sub-5ms per probe", FAIL, str(e))

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
