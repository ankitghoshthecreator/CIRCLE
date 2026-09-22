"""
tests/test_round10.py
CIRCLE - 15 Hard-Mode Edge-Case & Invariant Tests (Round 10)

Focus:
- Probe task harness evaluation rules (Punctuation, Dialogue, General)
- ProbeExecutionReport schema integrity, JSON persistence, and score bound invariants
- Stage 4-5 dialogue turn tracking and vocabulary repetition heuristics
- Kubernetes RBAC, HPA, PVC, and orchestrator state machine invariants
"""

import os
import sys
import json
import inspect
import tempfile
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = "PASS", "FAIL"
results = []


def report(tid, name, status, detail=""):
    icon = "OK" if status == PASS else "XX"
    print(f"  [{icon}] Test {tid:02d}: {name} - {status}" + (f"\n         Detail: {detail}" if detail else ""))
    results.append((tid, name, status, detail))


print("\n" + "=" * 65)
print("  CIRCLE - 15 Hard-Mode Edge-Case Tests (Round 10)")
print("=" * 65 + "\n")


# ═══════════════════════════════════════════════════════════
# BLOCK A: PROBE HARNESS ENGINE & RULE EVALUATION (T01 - T05)
# ═══════════════════════════════════════════════════════════
print("[ A ] Probe Harness Engine & Rule Evaluation")

# T01: Stage 4 punctuation evaluator scores terminal_punctuation 1.0 for valid endings and <1.0 for missing
try:
    from eval.probe_harness import evaluate_stage_4_punctuation
    rules_valid = evaluate_stage_4_punctuation("Prompt", "This sentence is punctuated correctly.")
    rules_invalid = evaluate_stage_4_punctuation("Prompt", "This sentence has no terminal mark")
    
    score_v = next(r.score for r in rules_valid if r.rule_name == "terminal_punctuation")
    score_inv = next(r.score for r in rules_invalid if r.rule_name == "terminal_punctuation")
    
    assert score_v == 1.0, f"Expected 1.0 for valid ending, got {score_v}"
    assert score_inv < 1.0, f"Expected <1.0 for missing terminal mark, got {score_inv}"
    report(1, "Stage 4 punctuation evaluator validates terminal_punctuation bounds correctly", PASS)
except Exception as e:
    report(1, "Stage 4 punctuation evaluator validates terminal_punctuation bounds correctly", FAIL, str(e))


# T02: Stage 4 punctuation evaluator flags odd quotation marks (unbalanced quotes)
try:
    from eval.probe_harness import evaluate_stage_4_punctuation
    rules_unbalanced = evaluate_stage_4_punctuation("Prompt", 'She said "hello and left.')
    quote_rule = next(r for r in rules_unbalanced if r.rule_name == "quote_balance")
    assert not quote_rule.passed, "Unbalanced quotation mark was improperly marked as passed"
    assert quote_rule.score <= 0.3, f"Expected score <= 0.3 for unbalanced quotes, got {quote_rule.score}"
    report(2, "Stage 4 evaluator flags odd/unbalanced quotation mark syntax", PASS)
except Exception as e:
    report(2, "Stage 4 evaluator flags odd/unbalanced quotation mark syntax", FAIL, str(e))


# T03: Stage 5 dialogue evaluator flags 'User:' prefix hijacking to enforce Assistant persona
try:
    from eval.probe_harness import evaluate_stage_5_dialogue
    rules_hijack = evaluate_stage_5_dialogue("User: Hi\nAssistant:", "User: I will answer myself now.")
    turn_rule = next(r for r in rules_hijack if r.rule_name == "speaker_turn_adherence")
    assert not turn_rule.passed, "Speaker turn hijack was improperly marked as passed"
    assert turn_rule.score <= 0.3, f"Expected score <= 0.3 for turn hijack, got {turn_rule.score}"
    report(3, "Stage 5 dialogue evaluator detects and penalizes User: speaker turn hijacking", PASS)
except Exception as e:
    report(3, "Stage 5 dialogue evaluator detects and penalizes User: speaker turn hijacking", FAIL, str(e))


# T04: Stage 5 dialogue evaluator detects repetitive vocabulary looping
try:
    from eval.probe_harness import evaluate_stage_5_dialogue
    looping_text = "word " * 30  # Highly repetitive text
    rules_loop = evaluate_stage_5_dialogue("User: Help\nAssistant:", looping_text)
    rep_rule = next(r for r in rules_loop if r.rule_name == "no_repetitive_looping")
    assert not rep_rule.passed, "Repetitive looping text was improperly marked as passed"
    assert rep_rule.score <= 0.2, f"Expected score <= 0.2 for vocabulary loop, got {rep_rule.score}"
    report(4, "Stage 5 evaluator flags low-diversity vocabulary looping patterns", PASS)
except Exception as e:
    report(4, "Stage 5 evaluator flags low-diversity vocabulary looping patterns", FAIL, str(e))


# T05: ProbeTaskHarness sets advance_threshold_met=False when target score is unachievable
try:
    from eval.probe_harness import ProbeTaskHarness
    harness = ProbeTaskHarness(output_dir=tempfile.gettempdir())
    report_obj = harness.run_stage_probes(stage_id=4, mock_mode=True, target_score_override=1.05)
    assert not report_obj.advance_threshold_met, "advance_threshold_met should be False for target > 1.0"
    report(5, "ProbeTaskHarness evaluates advance_threshold_met=False for unachievable target scores", PASS)
except Exception as e:
    report(5, "ProbeTaskHarness evaluates advance_threshold_met=False for unachievable target scores", FAIL, str(e))


# ═══════════════════════════════════════════════════════════
# BLOCK B: REPORT SERIALIZATION & SCHEMA INVARIANTS (T06 - T10)
# ═══════════════════════════════════════════════════════════
print("\n[ B ] Report Serialization & Schema Invariants")

# T06: ProbeExecutionReport contains non-empty ISO format timestamp
try:
    from eval.probe_harness import ProbeTaskHarness
    harness = ProbeTaskHarness(output_dir=tempfile.gettempdir())
    rep = harness.run_stage_probes(stage_id=4, mock_mode=True)
    assert rep.timestamp, "Report timestamp is empty"
    assert "T" in rep.timestamp, f"Timestamp does not look like ISO format: {rep.timestamp}"
    report(6, "ProbeExecutionReport generates valid ISO timestamp format", PASS)
except Exception as e:
    report(6, "ProbeExecutionReport generates valid ISO timestamp format", FAIL, str(e))


# T07: ProbeExecutionReport.summary_by_rule keys match all unique rule names in detailed_results
try:
    from eval.probe_harness import ProbeTaskHarness
    harness = ProbeTaskHarness(output_dir=tempfile.gettempdir())
    rep = harness.run_stage_probes(stage_id=4, mock_mode=True)
    all_rule_names = {r.rule_name for item in rep.detailed_results for r in item.rule_scores}
    summary_rule_names = set(rep.summary_by_rule.keys())
    assert all_rule_names == summary_rule_names, f"Rule names mismatch: {all_rule_names} vs {summary_rule_names}"
    report(7, "ProbeExecutionReport summary_by_rule keys strictly match all detailed rule names", PASS)
except Exception as e:
    report(7, "ProbeExecutionReport summary_by_rule keys strictly match all detailed rule names", FAIL, str(e))


# T08: All RuleScoreDetail scores across all probes strictly bounded in [0.0, 1.0]
try:
    from eval.probe_harness import ProbeTaskHarness
    harness = ProbeTaskHarness(output_dir=tempfile.gettempdir())
    rep = harness.run_stage_probes(stage_id=5, mock_mode=True)
    for item in rep.detailed_results:
        for r in item.rule_scores:
            assert 0.0 <= r.score <= 1.0, f"Rule score out of bounds: {r.score} for rule '{r.rule_name}'"
    report(8, "All RuleScoreDetail scores strictly bounded within [0.0, 1.0]", PASS)
except Exception as e:
    report(8, "All RuleScoreDetail scores strictly bounded within [0.0, 1.0]", FAIL, str(e))


# T09: load_stage_dataset returns CurriculumExample objects with default source 'seed'
try:
    from trainer.curriculum.dataset_handler import load_stage_dataset
    examples = load_stage_dataset(4, data_dir=os.path.join(PROJECT_ROOT, "data"))
    assert len(examples) > 0, "Stage 4 dataset is empty"
    for ex in examples:
        assert ex.source == "seed", f"Expected source 'seed', got '{ex.source}'"
    report(9, "load_stage_dataset returns CurriculumExample objects with default source 'seed'", PASS)
except Exception as e:
    report(9, "load_stage_dataset returns CurriculumExample objects with default source 'seed'", FAIL, str(e))


# T10: STAGES dictionary target competence scores bounded in [0.70, 0.95] for all 5 stages
try:
    from trainer.curriculum.stage_config import STAGES
    for stage_id, stage_obj in STAGES.items():
        score = stage_obj.target_competence_score
        assert 0.70 <= score <= 0.95, f"Stage {stage_id} target competence score out of bounds: {score}"
    report(10, "STAGES target_competence_score values bounded within [0.70, 0.95] across all 5 stages", PASS)
except Exception as e:
    report(10, "STAGES target_competence_score values bounded within [0.70, 0.95] across all 5 stages", FAIL, str(e))


# ═══════════════════════════════════════════════════════════
# BLOCK C: ORCHESTRATION & KUBERNETES INVARIANTS (T11 - T15)
# ═══════════════════════════════════════════════════════════
print("\n[ C ] Orchestration & Kubernetes Invariants")

# T11: orchestrator/k8s/rbac.yaml grants batch/jobs API access for orchestrator Job control
try:
    import yaml
    rbac_path = os.path.join(PROJECT_ROOT, "orchestrator", "k8s", "rbac.yaml")
    with open(rbac_path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    role_doc = next(d for d in docs if d and d.get("kind") in ["ClusterRole", "Role"])
    rules = role_doc["rules"]
    batch_rule = next(r for r in rules if "batch" in r.get("apiGroups", []))
    assert "jobs" in batch_rule["resources"], "RBAC role missing 'jobs' resource access in batch apiGroup"
    report(11, "RBAC manifest grants batch/jobs API resource access for orchestrator", PASS)
except Exception as e:
    report(11, "RBAC manifest grants batch/jobs API resource access for orchestrator", FAIL, str(e))


# T12: orchestrator/k8s/hpa.yaml targets generator service with minReplicas 1 and maxReplicas 4
try:
    import yaml
    hpa_path = os.path.join(PROJECT_ROOT, "orchestrator", "k8s", "hpa.yaml")
    with open(hpa_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["spec"]["scaleTargetRef"]["name"] == "circle-generator"
    assert data["spec"]["minReplicas"] == 1
    assert data["spec"]["maxReplicas"] == 4
    report(12, "HPA manifest scales generator service between 1 and 4 replicas", PASS)
except Exception as e:
    report(12, "HPA manifest scales generator service between 1 and 4 replicas", FAIL, str(e))


# T13: orchestrator StatePersistenceManager creates state JSON with required keys
try:
    from orchestrator.state_machine import StatePersistenceManager, PipelineState, LoopState
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = StatePersistenceManager(state_dir=tmpdir)
        state = PipelineState(run_id="test_run_10", current_stage=2, current_state=LoopState.TRAINING)
        mgr.save(state)
        loaded = mgr.load("test_run_10")
        assert loaded is not None, "Failed to load saved state"
        assert loaded.run_id == "test_run_10"
        assert loaded.current_stage == 2
        assert loaded.current_state == LoopState.TRAINING
    report(13, "StatePersistenceManager saves and restores PipelineState JSON faithfully", PASS)
except Exception as e:
    report(13, "StatePersistenceManager saves and restores PipelineState JSON faithfully", FAIL, str(e))


# T14: eval/probe_harness.py raises FileNotFoundError for non-existent stage config
try:
    from eval.probe_harness import ProbeTaskHarness
    harness = ProbeTaskHarness(output_dir=tempfile.gettempdir())
    try:
        harness.run_stage_probes(stage_id=99, mock_mode=True)
        report(14, "ProbeTaskHarness raises FileNotFoundError for non-existent stage ID 99", FAIL, "Expected FileNotFoundError")
    except FileNotFoundError:
        report(14, "ProbeTaskHarness raises FileNotFoundError for non-existent stage ID 99", PASS)
except Exception as e:
    report(14, "ProbeTaskHarness raises FileNotFoundError for non-existent stage ID 99", FAIL, str(e))


# T15: Subprocess execution of test_part15.py passes cleanly with 0 return code
try:
    script_path = os.path.join(PROJECT_ROOT, "tests", "test_part15.py")
    res = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True)
    assert res.returncode == 0, f"test_part15.py failed with return code {res.returncode}"
    report(15, "test_part15.py executes as standalone test suite with 0 return code", PASS)
except Exception as e:
    report(15, "test_part15.py executes as standalone test suite with 0 return code", FAIL, str(e))


# ─── SUMMARY ───
print("\n" + "=" * 65)
passed = sum(1 for _, _, s, _ in results if s == PASS)
failed = sum(1 for _, _, s, _ in results if s == FAIL)
print(f"  RESULTS: {passed} PASSED  |  {failed} FAILED  |  {len(results)} TOTAL")
print("=" * 65 + "\n")
if failed > 0:
    sys.exit(1)
