"""
CIRCLE - 10 Difficult Test Cases
Tests edge cases and failure modes across all Parts 1-5 components.
"""
import os
import sys
import json
import torch
import shutil
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

PASS = "PASS"
FAIL = "FAIL"
results = []

def report(test_id, name, status, detail=""):
    icon = "OK" if status == PASS else "XX"
    print(f"  [{icon}] Test {test_id:02d}: {name} - {status}" + (f"\n         Detail: {detail}" if detail else ""))
    results.append((test_id, name, status, detail))

print("\n" + "="*65)
print("  CIRCLE - 10 Difficult Test Cases")
print("="*65 + "\n")

# ─── TEST 1: Stage 1 has replay_ratio=0 → Blending must return only current data unchanged ───
print("[ REPLAY BUFFER ]")
try:
    from trainer.replay_buffer import ReplayBufferManager
    from trainer.curriculum.dataset_handler import CurriculumExample
    mgr = ReplayBufferManager(data_dir="./data")
    current = [CurriculumExample(input_text="Stage 1 text", target_text="", stage_id=1)]
    blended = mgr.get_blended_dataset(1, current, replay_ratio=0.0)
    assert blended == current, "Replay blending Stage 1 should return current data unchanged"
    report(1, "Stage 1 replay_ratio=0.0 → no blending", PASS)
except Exception as e:
    report(1, "Stage 1 replay_ratio=0.0 → no blending", FAIL, str(e))

# ─── TEST 2: Large replay ratio (5.0) with tiny dataset → should not crash, cap at available ───
try:
    mgr = ReplayBufferManager(data_dir="./data")
    current = [CurriculumExample(input_text="Stage 3 test item", target_text="", stage_id=3)]
    blended = mgr.get_blended_dataset(3, current, replay_ratio=5.0)
    assert len(blended) >= len(current), "Must have at least current examples"
    report(2, "Extreme replay_ratio=5.0 doesn't crash", PASS, f"Blended size: {len(blended)}")
except Exception as e:
    report(2, "Extreme replay_ratio=5.0 doesn't crash", FAIL, str(e))

# ─── TEST 3: Replay buffer with no prior stage data → returns current unchanged ───
try:
    mgr = ReplayBufferManager(data_dir="./data_nonexistent_test_dir")
    current = [CurriculumExample(input_text="Orphan stage text", target_text="", stage_id=2)]
    blended = mgr.get_blended_dataset(2, current, replay_ratio=0.3)
    assert len(blended) == len(current), "No prior data → should return current unchanged"
    report(3, "Replay buffer with no prior data → returns current unchanged", PASS)
except Exception as e:
    report(3, "Replay buffer with no prior data → returns current unchanged", FAIL, str(e))

# ─── TEST 4: CurriculumDataset with empty input_text and target_text → must not crash ───
print("\n[ CURRICULUM DATASET HANDLER ]")
try:
    from trainer.curriculum.dataset_handler import CurriculumDataset, CurriculumExample
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    empty_example = CurriculumExample(input_text="", target_text="", stage_id=1)
    ds = CurriculumDataset([empty_example], tokenizer)
    item = ds[0]
    assert "input_ids" in item
    report(4, "CurriculumDataset handles empty input_text/target_text", PASS)
except Exception as e:
    report(4, "CurriculumDataset handles empty input_text/target_text", FAIL, str(e))

# ─── TEST 5: CurriculumDataset with very long input (truncation must be applied) ───
try:
    from trainer.curriculum.dataset_handler import CurriculumDataset, CurriculumExample
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    long_text = "The quick brown fox jumps over the lazy dog. " * 200  # ~8000 tokens
    example = CurriculumExample(input_text=long_text, target_text="", stage_id=1)
    ds = CurriculumDataset([example], tokenizer, max_length=128)
    item = ds[0]
    assert item["input_ids"].shape[0] == 128, f"Expected 128 tokens, got {item['input_ids'].shape[0]}"
    report(5, "CurriculumDataset truncates long inputs to max_length=128", PASS)
except Exception as e:
    report(5, "CurriculumDataset truncates long inputs to max_length=128", FAIL, str(e))

# ─── TEST 6: Load corrupted/missing stage JSON config → must raise FileNotFoundError ───
print("\n[ STAGE CONFIGS ]")
try:
    from trainer.curriculum.dataset_handler import load_stage_config
    try:
        load_stage_config(99)  # Stage 99 does not exist
        report(6, "Loading non-existent stage config (stage 99) raises FileNotFoundError", FAIL, "No exception raised!")
    except FileNotFoundError:
        report(6, "Loading non-existent stage config (stage 99) raises FileNotFoundError", PASS)
except Exception as e:
    report(6, "Loading non-existent stage config (stage 99) raises FileNotFoundError", FAIL, str(e))

# ─── TEST 7: load_stage_dataset with completely empty JSON file → should handle gracefully ───
try:
    from trainer.curriculum.dataset_handler import load_stage_dataset, save_stage_dataset
    test_data_dir = "./data_test_empty"
    os.makedirs(os.path.join(test_data_dir, "stage_1"), exist_ok=True)
    empty_file = os.path.join(test_data_dir, "stage_1", "dataset.json")
    with open(empty_file, "w") as f:
        json.dump([], f)  # empty list
    result = load_stage_dataset(1, data_dir=test_data_dir)
    # If file exists but is empty, it returns [], which triggers fallback to seed examples
    shutil.rmtree(test_data_dir, ignore_errors=True)
    report(7, "load_stage_dataset handles empty dataset.json gracefully", PASS, f"Returned {len(result)} examples")
except Exception as e:
    shutil.rmtree("./data_test_empty", ignore_errors=True)
    report(7, "load_stage_dataset handles empty dataset.json gracefully", FAIL, str(e))

# ─── TEST 8: GroqCriticAgent with empty probe results list → must not crash ───
print("\n[ GROQ CRITIC AGENT ]")
try:
    from eval.critique import GroqCriticAgent
    agent = GroqCriticAgent()
    result = agent.analyze_stage_outputs(1, "Base English", ["fluency"], [])
    assert "stage_id" in result
    report(8, "GroqCriticAgent handles empty probe_results list", PASS)
except Exception as e:
    report(8, "GroqCriticAgent handles empty probe_results list", FAIL, str(e))

# ─── TEST 9: GroqCriticAgent with intentionally bad API key → must fail gracefully ───
try:
    from eval.critique import GroqCriticAgent
    bad_agent = GroqCriticAgent(api_key="bad_key_xyz")
    result = bad_agent.analyze_stage_outputs(1, "Base English", ["fluency"], [
        {"prompt": "test", "output": "test output"}
    ])
    assert "error" in result or "critique_raw" in result, "Should return error dict"
    report(9, "GroqCriticAgent fails gracefully with invalid API key", PASS, f"Got: {list(result.keys())}")
except Exception as e:
    report(9, "GroqCriticAgent fails gracefully with invalid API key", FAIL, str(e))

# ─── TEST 10: run_stage_eval with missing checkpoint → evaluates base model without crashing ───
print("\n[ PROBE EVALUATION HARNESS ]")
try:
    from eval.run_stage_eval import evaluate_stage_checkpoint
    report_dir = "./eval/reports_test"
    result = evaluate_stage_checkpoint(
        stage_id=1,
        checkpoint_dir="./trainer/checkpoints_nonexistent",  # Force fallback to base model
        output_dir=report_dir
    )
    assert "stage_id" in result
    assert os.path.exists(os.path.join(report_dir, "stage_1_critique.json")), "Report file must be saved"
    shutil.rmtree(report_dir, ignore_errors=True)
    report(10, "run_stage_eval falls back to base model if adapter checkpoint missing", PASS)
except Exception as e:
    shutil.rmtree("./eval/reports_test", ignore_errors=True)
    report(10, "run_stage_eval falls back to base model if adapter checkpoint missing", FAIL, str(e))

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
            print(f"    → {detail}")
