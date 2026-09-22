"""
tests/test_part13.py
CIRCLE Part 13: Orchestrator Control Loop & State Engine — Test Suite

Covers:
  Block A — LoopState enum & PipelineState model (T01–T05)
  Block B — StatePersistenceManager save/load/resume (T06–T10)
  Block C — ProbeRunner stage probes & mock output (T11–T15)
  Block D — ClosedLoopOrchestrator mock execution & API surface (T16–T20)
"""

import os
import sys
import json
import uuid
import shutil
import tempfile

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
print("  CIRCLE Part 13 — Orchestrator State Engine Test Suite")
print("="*65 + "\n")

# ────────────────────────────────────────────────────────────────
# BLOCK A: LoopState Enum & PipelineState Model
# ────────────────────────────────────────────────────────────────
print("[ A ] LoopState Enum & PipelineState Model")

# T01: LoopState enum has all 8 required values
try:
    from orchestrator.state_machine import LoopState
    required = {"idle", "training", "evaluating", "generating", "merging", "advancing", "complete", "failed"}
    actual = {s.value for s in LoopState}
    assert required == actual, f"LoopState values mismatch. Missing: {required - actual}"
    report(1, "LoopState enum has all 8 required state values", PASS)
except Exception as e:
    report(1, "LoopState enum has all 8 required state values", FAIL, str(e))


# T02: PipelineState initialises with correct defaults
try:
    from orchestrator.state_machine import PipelineState, LoopState
    state = PipelineState(run_id="test-t02")
    assert state.current_stage == 1
    assert state.current_iteration == 1
    assert state.max_iterations_per_stage == 3
    assert state.total_stages == 5
    assert state.current_state == LoopState.IDLE
    assert state.stage_advance_threshold == 0.85
    assert state.stage_results == []
    assert state.error_log == []
    report(2, "PipelineState initialises with correct defaults (stage=1, iter=1, state=idle)", PASS)
except Exception as e:
    report(2, "PipelineState initialises with correct defaults", FAIL, str(e))


# T03: PipelineState.transition() changes state and stamps last_updated_at
try:
    from orchestrator.state_machine import PipelineState, LoopState
    state = PipelineState(run_id="test-t03")
    ts_before = state.last_updated_at
    state.transition(LoopState.TRAINING)
    assert state.current_state == LoopState.TRAINING
    # timestamp should be updated (or at worst equal if sub-second)
    assert state.last_updated_at >= ts_before
    report(3, "PipelineState.transition() updates current_state and last_updated_at", PASS)
except Exception as e:
    report(3, "PipelineState.transition() updates state correctly", FAIL, str(e))


# T04: PipelineState.log_error() appends timestamped entry to error_log
try:
    from orchestrator.state_machine import PipelineState
    state = PipelineState(run_id="test-t04")
    state.log_error("Something went wrong")
    assert len(state.error_log) == 1
    assert "Something went wrong" in state.error_log[0]
    assert "T" in state.error_log[0]  # ISO timestamp check
    report(4, "PipelineState.log_error() appends timestamped error entry", PASS)
except Exception as e:
    report(4, "PipelineState.log_error() appends error entry", FAIL, str(e))


# T05: PipelineState.is_complete and is_failed properties work correctly
try:
    from orchestrator.state_machine import PipelineState, LoopState
    state = PipelineState(run_id="test-t05")
    assert not state.is_complete
    assert not state.is_failed
    state.transition(LoopState.COMPLETE)
    assert state.is_complete
    assert not state.is_failed
    state.transition(LoopState.FAILED)
    assert state.is_failed
    assert not state.is_complete
    report(5, "PipelineState.is_complete / is_failed properties toggle correctly", PASS)
except Exception as e:
    report(5, "PipelineState.is_complete / is_failed properties", FAIL, str(e))


# ────────────────────────────────────────────────────────────────
# BLOCK B: StatePersistenceManager Save / Load / Resume
# ────────────────────────────────────────────────────────────────
print("\n[ B ] StatePersistenceManager Save / Load / Resume")

TMP_DIR = tempfile.mkdtemp(prefix="circle_test_state_")


# T06: save() persists state to a JSON file with correct run_id filename
try:
    from orchestrator.state_machine import PipelineState, StatePersistenceManager
    mgr = StatePersistenceManager(state_dir=TMP_DIR)
    state = PipelineState(run_id="run-t06")
    path = mgr.save(state)
    assert os.path.exists(path), f"State file not found: {path}"
    assert "run-t06" in path
    report(6, "StatePersistenceManager.save() writes JSON file named by run_id", PASS)
except Exception as e:
    report(6, "StatePersistenceManager.save() writes JSON file", FAIL, str(e))


# T07: saved JSON is valid and contains all required top-level fields
try:
    from orchestrator.state_machine import PipelineState, StatePersistenceManager
    mgr = StatePersistenceManager(state_dir=TMP_DIR)
    state = PipelineState(run_id="run-t07", current_stage=3, current_iteration=2)
    path = mgr.save(state)
    with open(path) as f:
        data = json.load(f)
    for key in ["run_id", "current_stage", "current_iteration", "current_state",
                "stage_advance_threshold", "stage_results", "error_log",
                "started_at", "last_updated_at"]:
        assert key in data, f"Missing key in saved JSON: {key}"
    assert data["current_stage"] == 3
    assert data["current_iteration"] == 2
    report(7, "Saved state JSON contains all required PipelineState fields with correct values", PASS)
except Exception as e:
    report(7, "Saved state JSON contains all required fields", FAIL, str(e))


# T08: load() restores a saved PipelineState with identical values
try:
    from orchestrator.state_machine import PipelineState, StatePersistenceManager, LoopState
    mgr = StatePersistenceManager(state_dir=TMP_DIR)
    state = PipelineState(run_id="run-t08", current_stage=2, current_iteration=3)
    state.transition(LoopState.EVALUATING)
    mgr.save(state)

    loaded = mgr.load("run-t08")
    assert loaded is not None, "load() returned None for existing run"
    assert loaded.run_id == "run-t08"
    assert loaded.current_stage == 2
    assert loaded.current_iteration == 3
    assert loaded.current_state == LoopState.EVALUATING
    report(8, "StatePersistenceManager.load() restores PipelineState with identical values", PASS)
except Exception as e:
    report(8, "StatePersistenceManager.load() restores state correctly", FAIL, str(e))


# T09: load() returns None for unknown run_id (no crash)
try:
    from orchestrator.state_machine import StatePersistenceManager
    mgr = StatePersistenceManager(state_dir=TMP_DIR)
    result = mgr.load("nonexistent-run-xyz-123")
    assert result is None, f"Expected None for missing run, got {result}"
    report(9, "StatePersistenceManager.load() returns None for unknown run_id (no crash)", PASS)
except Exception as e:
    report(9, "StatePersistenceManager.load() returns None for unknown run", FAIL, str(e))


# T10: delete() removes the state file and returns True
try:
    from orchestrator.state_machine import PipelineState, StatePersistenceManager
    mgr = StatePersistenceManager(state_dir=TMP_DIR)
    state = PipelineState(run_id="run-t10-delete")
    path = mgr.save(state)
    assert os.path.exists(path)
    ok = mgr.delete("run-t10-delete")
    assert ok is True
    assert not os.path.exists(path), "State file still exists after delete()"
    report(10, "StatePersistenceManager.delete() removes state file and returns True", PASS)
except Exception as e:
    report(10, "StatePersistenceManager.delete() removes state file", FAIL, str(e))

# Cleanup temp dir
shutil.rmtree(TMP_DIR, ignore_errors=True)


# ────────────────────────────────────────────────────────────────
# BLOCK C: ProbeRunner Stage Probes & Mock Outputs
# ────────────────────────────────────────────────────────────────
print("\n[ C ] ProbeRunner Stage Probes & Mock Outputs")


# T11: STAGE_PROBES contains entries for all 5 stages
try:
    from orchestrator.probe_runner import STAGE_PROBES
    for sid in range(1, 6):
        assert sid in STAGE_PROBES, f"STAGE_PROBES missing stage {sid}"
        assert len(STAGE_PROBES[sid]) >= 2, f"Stage {sid} has fewer than 2 probes"
    report(11, "STAGE_PROBES contains probe entries for all 5 curriculum stages", PASS)
except Exception as e:
    report(11, "STAGE_PROBES entries for all 5 stages", FAIL, str(e))


# T12: Each probe entry has 'prompt' and 'expected_theme' keys
try:
    from orchestrator.probe_runner import STAGE_PROBES
    for sid, probes in STAGE_PROBES.items():
        for p in probes:
            assert "prompt" in p, f"Stage {sid} probe missing 'prompt' key"
            assert "expected_theme" in p, f"Stage {sid} probe missing 'expected_theme' key"
    report(12, "All STAGE_PROBES entries have 'prompt' and 'expected_theme' keys", PASS)
except Exception as e:
    report(12, "STAGE_PROBES entries have required keys", FAIL, str(e))


# T13: ProbeRunner.run_probes() returns correct number of {prompt, output} dicts
try:
    from orchestrator.probe_runner import ProbeRunner, STAGE_PROBES
    runner = ProbeRunner(use_mock=True)
    for sid in range(1, 6):
        results_list = runner.run_probes(stage_id=sid)
        expected_count = len(STAGE_PROBES[sid])
        assert len(results_list) == expected_count, \
            f"Stage {sid}: expected {expected_count} probe results, got {len(results_list)}"
    report(13, "ProbeRunner.run_probes() returns exactly len(STAGE_PROBES[sid]) result dicts", PASS)
except Exception as e:
    report(13, "ProbeRunner.run_probes() returns correct count", FAIL, str(e))


# T14: Each probe result dict has 'prompt' and 'output' keys with non-empty string values
try:
    from orchestrator.probe_runner import ProbeRunner
    runner = ProbeRunner(use_mock=True)
    probe_results = runner.run_probes(stage_id=1)
    for r in probe_results:
        assert "prompt" in r and isinstance(r["prompt"], str) and r["prompt"].strip()
        assert "output" in r and isinstance(r["output"], str) and r["output"].strip()
    report(14, "All probe results have non-empty 'prompt' and 'output' string fields", PASS)
except Exception as e:
    report(14, "Probe results have required 'prompt' and 'output' fields", FAIL, str(e))


# T15: ProbeRunner respects custom_probes override parameter
try:
    from orchestrator.probe_runner import ProbeRunner
    runner = ProbeRunner(use_mock=True)
    custom = [
        {"prompt": "Custom probe 1", "expected_theme": "test"},
        {"prompt": "Custom probe 2", "expected_theme": "test"},
        {"prompt": "Custom probe 3", "expected_theme": "test"},
    ]
    results_list = runner.run_probes(stage_id=1, custom_probes=custom)
    assert len(results_list) == 3, f"Expected 3 custom probe results, got {len(results_list)}"
    assert results_list[0]["prompt"] == "Custom probe 1"
    report(15, "ProbeRunner respects custom_probes override parameter", PASS)
except Exception as e:
    report(15, "ProbeRunner respects custom_probes override", FAIL, str(e))


# ────────────────────────────────────────────────────────────────
# BLOCK D: ClosedLoopOrchestrator Mock Execution & API Surface
# ────────────────────────────────────────────────────────────────
print("\n[ D ] ClosedLoopOrchestrator Mock Execution & API Surface")


TMP_LOOP_DIR = tempfile.mkdtemp(prefix="circle_test_loop_")


# T16: ClosedLoopOrchestrator constructs without error in mock mode
try:
    from orchestrator.loop_controller import ClosedLoopOrchestrator
    orch = ClosedLoopOrchestrator(
        start_stage=1, max_iterations=1,
        stage_advance_threshold=0.0,
        log_dir=TMP_LOOP_DIR, mock_mode=True,
        run_id="test-t16"
    )
    assert orch.mock_mode is True
    assert orch.run_id == "test-t16"
    report(16, "ClosedLoopOrchestrator constructs without error in mock mode", PASS)
except Exception as e:
    report(16, "ClosedLoopOrchestrator construction in mock mode", FAIL, str(e))


# T17: execute_loop() completes all 5 stages and returns PipelineState with state=complete
try:
    from orchestrator.loop_controller import ClosedLoopOrchestrator
    from orchestrator.state_machine import LoopState
    orch = ClosedLoopOrchestrator(
        start_stage=1, max_iterations=1,
        stage_advance_threshold=0.0,   # force advance every iter
        log_dir=TMP_LOOP_DIR, mock_mode=True,
        run_id=f"test-t17-{uuid.uuid4().hex[:6]}"
    )
    final = orch.execute_loop()
    assert final.current_state == LoopState.COMPLETE, \
        f"Expected COMPLETE, got {final.current_state}"
    assert final.stages_completed == 5, \
        f"Expected 5 stages completed, got {final.stages_completed}"
    report(17, "execute_loop() completes all 5 stages and returns state=COMPLETE", PASS)
except Exception as e:
    report(17, "execute_loop() completes all 5 stages", FAIL, str(e))


# T18: execute_loop() produces exactly 5 StageResult records
try:
    from orchestrator.loop_controller import ClosedLoopOrchestrator
    orch = ClosedLoopOrchestrator(
        start_stage=1, max_iterations=1,
        stage_advance_threshold=0.0,
        log_dir=TMP_LOOP_DIR, mock_mode=True,
        run_id=f"test-t18-{uuid.uuid4().hex[:6]}"
    )
    final = orch.execute_loop()
    assert len(final.stage_results) == 5, \
        f"Expected 5 StageResults, got {len(final.stage_results)}"
    # Each result should have its stage_id set correctly
    for i, r in enumerate(final.stage_results):
        assert r.stage_id == i + 1, f"Result {i} has stage_id {r.stage_id}, expected {i+1}"
    report(18, "execute_loop() produces 5 StageResult records with correct stage_ids", PASS)
except Exception as e:
    report(18, "execute_loop() produces correct StageResult records", FAIL, str(e))


# T19: State file persisted to log_dir after loop completes
try:
    from orchestrator.loop_controller import ClosedLoopOrchestrator
    run_id = f"test-t19-{uuid.uuid4().hex[:6]}"
    orch = ClosedLoopOrchestrator(
        start_stage=1, max_iterations=1,
        stage_advance_threshold=0.0,
        log_dir=TMP_LOOP_DIR, mock_mode=True,
        run_id=run_id
    )
    orch.execute_loop()
    expected_path = os.path.join(TMP_LOOP_DIR, f"pipeline_state_{run_id}.json")
    assert os.path.exists(expected_path), f"State file not found: {expected_path}"
    with open(expected_path) as f:
        data = json.load(f)
    assert data["run_id"] == run_id
    assert data["current_state"] == "complete"
    report(19, "State JSON persisted to log_dir after loop completes with state='complete'", PASS)
except Exception as e:
    report(19, "State file persisted correctly after loop", FAIL, str(e))


# T20: max_iterations circuit-breaker prevents infinite loops when threshold not met
try:
    from orchestrator.loop_controller import ClosedLoopOrchestrator
    from orchestrator.state_machine import LoopState
    # threshold=0.99 is never satisfied by mock eval (~0.72) and max_iterations=1
    # → circuit-breaker fires after 1 iteration per stage and forces advance through all 5
    orch = ClosedLoopOrchestrator(
        start_stage=1, max_iterations=1,
        stage_advance_threshold=0.99,   # mock score ~0.72 → never satisfies → circuit-break
        log_dir=TMP_LOOP_DIR, mock_mode=True,
        run_id=f"test-t20-{uuid.uuid4().hex[:6]}"
    )
    final = orch.execute_loop()
    # Circuit breaker should force advance through all 5 stages → COMPLETE
    assert final.current_state == LoopState.COMPLETE, \
        f"Expected COMPLETE after circuit-break, got {final.current_state}"
    # All 5 stage results should have advance_to_next_stage=True (forced)
    assert len(final.stage_results) == 5, \
        f"Expected 5 StageResults, got {len(final.stage_results)}"
    report(20, "Max-iterations circuit-breaker forces advance after max_iterations (no infinite loop)", PASS)
except Exception as e:
    report(20, "Max-iterations circuit-breaker prevents infinite loops", FAIL, str(e))


shutil.rmtree(TMP_LOOP_DIR, ignore_errors=True)


# ─── SUMMARY ───
print("\n" + "="*65)
passed = sum(1 for _, _, s, _ in results if s == PASS)
failed = sum(1 for _, _, s, _ in results if s == FAIL)
print(f"  RESULTS: {passed} PASSED  |  {failed} FAILED  |  {len(results)} TOTAL")
print("="*65 + "\n")
if failed > 0:
    sys.exit(1)
