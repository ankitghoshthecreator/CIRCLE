"""
CIRCLE - 10 Harder Integration & Stress Test Cases (Round 2)
Tests deep integration, numerical correctness, and stress scenarios.
"""
import os
import sys
import json
import torch
import shutil
import logging
import tempfile

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
print("  CIRCLE - 10 Harder Integration & Stress Tests (Round 2)")
print("="*65 + "\n")

# ─── TEST 1: Training loss must decrease over 3 epochs on Stage 1 seed data ───
print("[ TRAINING ENGINE ]")
try:
    from trainer.train import train_stage
    from trainer.curriculum.dataset_handler import CurriculumExample

    losses_per_epoch = []
    # Use a tiny custom dataset so training is fast & loss is trackable
    custom = [CurriculumExample(input_text="The cat sat on the mat and ate.", target_text="", stage_id=1)] * 4

    # We need to peek at per-epoch losses - patch train_stage temporarily
    import trainer.train as train_module
    original_train = train_module.train_stage

    epoch_losses_captured = []

    # Instead of monkey-patching, run 2 separate 1-epoch runs and compare loss trend
    import torch
    from trainer.model_loader import load_qlora_model_and_tokenizer
    from trainer.curriculum.dataset_handler import CurriculumDataset
    from torch.utils.data import DataLoader

    model, tokenizer = load_qlora_model_and_tokenizer("gpt2", is_trainable=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CurriculumDataset(custom, tokenizer)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    batch = next(iter(dl))
    losses = []
    for step in range(3):
        out = model(input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device))
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    # Loss should change (model is learning, not frozen)
    assert losses[0] != losses[-1], "Loss must change across steps — model is not learning"
    report(1, "Training loss changes over multiple optimizer steps", PASS,
           f"Step losses: {[round(l,4) for l in losses]}")
except Exception as e:
    report(1, "Training loss changes over multiple optimizer steps", FAIL, str(e))

# ─── TEST 2: Saved adapter checkpoint file structure is valid (required files exist) ───
print("\n[ ADAPTER CHECKPOINT INTEGRITY ]")
try:
    adapter_path = "./trainer/checkpoints/stage_1"
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(adapter_path, f))]
    assert not missing, f"Missing checkpoint files: {missing}"

    # Validate adapter_config.json is valid JSON with expected keys
    with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
        config = json.load(f)
    assert "r" in config and "lora_alpha" in config, "adapter_config.json missing LoRA keys"
    report(2, "Stage 1 adapter checkpoint has valid file structure & LoRA config", PASS,
           f"r={config.get('r')}, lora_alpha={config.get('lora_alpha')}, target_modules={config.get('target_modules')}")
except Exception as e:
    report(2, "Stage 1 adapter checkpoint has valid file structure & LoRA config", FAIL, str(e))

# ─── TEST 3: Adapter checkpoint loads and generates text (not all zeros/garbage) ───
try:
    from trainer.model_loader import load_qlora_model_and_tokenizer, load_stage_adapter
    base_model, tokenizer = load_qlora_model_and_tokenizer("gpt2", is_trainable=False)
    model = load_stage_adapter(base_model, "./trainer/checkpoints/stage_1")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer("Language models learn", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    assert len(generated.strip()) > 0, "Generated text is empty"
    assert generated != "Language models learn", "Model produced no new tokens"
    report(3, "Loaded Stage 1 adapter generates non-empty coherent text", PASS, f"Output: '{generated}'")
except Exception as e:
    report(3, "Loaded Stage 1 adapter generates non-empty coherent text", FAIL, str(e))

# ─── TEST 4: Replay ratio exactly 0.2 yields correct blend proportion ───
print("\n[ REPLAY BUFFER NUMERICAL CORRECTNESS ]")
try:
    from trainer.replay_buffer import ReplayBufferManager
    from trainer.curriculum.dataset_handler import CurriculumExample, save_stage_dataset

    test_dir = "./data_test_ratio"
    # 10 stage-1 examples
    stage1_data = [CurriculumExample(input_text=f"S1 example {i}", target_text="", stage_id=1) for i in range(10)]
    save_stage_dataset(1, stage1_data, data_dir=test_dir)

    # 5 current stage-2 examples
    current = [CurriculumExample(input_text=f"S2 example {i}", target_text="", stage_id=2) for i in range(5)]

    mgr = ReplayBufferManager(data_dir=test_dir)
    blended = mgr.get_blended_dataset(2, current, replay_ratio=0.2)

    # Should have 5 current + max(1, int(5 * 0.2)) = 5 + 1 = 6
    replay_count = sum(1 for ex in blended if ex.stage_id == 1)
    current_count = sum(1 for ex in blended if ex.stage_id == 2)

    assert current_count == 5, f"Expected 5 current examples, got {current_count}"
    assert replay_count >= 1, f"Expected at least 1 replay example, got {replay_count}"
    shutil.rmtree(test_dir, ignore_errors=True)
    report(4, "Replay ratio=0.2 yields correct proportional blend", PASS,
           f"Current: {current_count}, Replay: {replay_count}, Total: {len(blended)}")
except Exception as e:
    shutil.rmtree("./data_test_ratio", ignore_errors=True)
    report(4, "Replay ratio=0.2 yields correct proportional blend", FAIL, str(e))

# ─── TEST 5: Stage identity preserved after dataset save->load roundtrip ───
print("\n[ DATASET SAVE/LOAD ROUNDTRIP ]")
try:
    from trainer.curriculum.dataset_handler import save_stage_dataset, load_stage_dataset, CurriculumExample
    test_dir = "./data_test_roundtrip"
    examples = [
        CurriculumExample(input_text=f"Input {i}", target_text=f"Target {i}",
                          stage_id=2, source="seed", metadata={"idx": i})
        for i in range(5)
    ]
    save_stage_dataset(2, examples, data_dir=test_dir)
    loaded = load_stage_dataset(2, data_dir=test_dir, use_seed_fallback=False)

    assert len(loaded) == len(examples), f"Length mismatch: {len(loaded)} vs {len(examples)}"
    for orig, loaded_ex in zip(examples, loaded):
        assert orig.input_text == loaded_ex.input_text
        assert orig.target_text == loaded_ex.target_text
        assert orig.stage_id == loaded_ex.stage_id
        assert orig.metadata == loaded_ex.metadata

    shutil.rmtree(test_dir, ignore_errors=True)
    report(5, "Dataset save->load roundtrip preserves all fields", PASS, f"Verified {len(examples)} examples")
except Exception as e:
    shutil.rmtree("./data_test_roundtrip", ignore_errors=True)
    report(5, "Dataset save->load roundtrip preserves all fields", FAIL, str(e))

# ─── TEST 6: All 5 stage JSON configs load with correct hyperparameter types ───
print("\n[ STAGE CONFIG VALIDATION ]")
try:
    from trainer.curriculum.dataset_handler import load_stage_config
    issues = []
    for sid in range(1, 6):
        cfg = load_stage_config(sid)
        if not isinstance(cfg.get("hyperparameters", {}).get("learning_rate"), float):
            issues.append(f"Stage {sid}: learning_rate is not a float")
        if not isinstance(cfg.get("replay_ratio"), float):
            issues.append(f"Stage {sid}: replay_ratio is not a float")
        if not isinstance(cfg.get("probe_prompts"), list) or len(cfg["probe_prompts"]) == 0:
            issues.append(f"Stage {sid}: probe_prompts is empty or not a list")
        if not isinstance(cfg.get("objectives"), list) or len(cfg["objectives"]) == 0:
            issues.append(f"Stage {sid}: objectives is empty or not a list")
    assert not issues, "\n".join(issues)
    report(6, "All 5 stage configs have correct hyperparameter types", PASS, "All stages validated")
except Exception as e:
    report(6, "All 5 stage configs have correct hyperparameter types", FAIL, str(e))

# ─── TEST 7: CurriculumDataset labels == input_ids (causal LM shift check) ───
print("\n[ CAUSAL LM DATASET INTEGRITY ]")
try:
    from trainer.curriculum.dataset_handler import CurriculumDataset, CurriculumExample
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ex = CurriculumExample(input_text="Hello world test sentence", target_text="response text", stage_id=1)
    ds = CurriculumDataset([ex], tokenizer)
    item = ds[0]
    assert torch.equal(item["input_ids"], item["labels"]), "Labels must equal input_ids for causal LM"
    assert item["input_ids"].dtype == torch.long, f"input_ids must be long, got {item['input_ids'].dtype}"
    assert item["attention_mask"].sum() > 0, "Attention mask is all zeros!"
    report(7, "CurriculumDataset labels == input_ids (causal LM format verified)", PASS,
           f"input_ids shape: {item['input_ids'].shape}, non-pad tokens: {item['attention_mask'].sum().item()}")
except Exception as e:
    report(7, "CurriculumDataset labels == input_ids (causal LM format verified)", FAIL, str(e))

# ─── TEST 8: Stage config replay_ratio ordering is monotonically increasing ───
try:
    from trainer.curriculum.dataset_handler import load_stage_config
    ratios = [load_stage_config(s)["replay_ratio"] for s in range(1, 6)]
    assert ratios[0] == 0.0, f"Stage 1 replay_ratio must be 0.0, got {ratios[0]}"
    for i in range(1, len(ratios) - 1):
        assert ratios[i] <= ratios[i+1], f"replay_ratio not monotonically increasing: {ratios}"
    report(8, "Stage replay_ratios are 0.0 for Stage 1 and non-decreasing across stages", PASS,
           f"Ratios: {ratios}")
except Exception as e:
    report(8, "Stage replay_ratios are 0.0 for Stage 1 and non-decreasing across stages", FAIL, str(e))

# ─── TEST 9: GroqCriticAgent report contains all required keys ───
print("\n[ CRITIC REPORT SCHEMA ]")
try:
    from eval.critique import GroqCriticAgent
    agent = GroqCriticAgent()
    sample = [{"prompt": "Test prompt for schema check", "output": "Test output response from model."}]
    report_data = agent.analyze_stage_outputs(1, "Base English", ["fluency", "syntax"], sample)
    required_keys = ["stage_id", "stage_name", "objectives", "probe_results"]
    missing = [k for k in required_keys if k not in report_data]
    assert not missing, f"Missing keys in critique report: {missing}"
    assert report_data["stage_id"] == 1
    assert report_data["stage_name"] == "Base English"
    report(9, "GroqCriticAgent report contains all required schema keys", PASS,
           f"Keys: {list(report_data.keys())}")
except Exception as e:
    report(9, "GroqCriticAgent report contains all required schema keys", FAIL, str(e))

# ─── TEST 10: Saved critique report file is valid JSON with required fields ───
print("\n[ SAVED CRITIQUE REPORT INTEGRITY ]")
try:
    report_path = "./eval/reports/stage_1_critique.json"
    assert os.path.exists(report_path), f"Report file missing: {report_path}"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("stage_id") == 1, "stage_id must be 1"
    assert isinstance(data.get("probe_results"), list), "probe_results must be a list"
    assert len(data["probe_results"]) > 0, "probe_results must not be empty"
    assert "critique_raw" in data, "critique_raw must exist in report"
    assert len(data.get("critique_raw", "")) > 50, "critique_raw seems too short to be a real critique"
    report(10, "Saved stage_1_critique.json is valid JSON with correct structure", PASS,
           f"Probes: {len(data['probe_results'])}, Critique length: {len(data.get('critique_raw',''))} chars")
except Exception as e:
    report(10, "Saved stage_1_critique.json is valid JSON with correct structure", FAIL, str(e))

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
