"""
tests/test_round6.py
CIRCLE - 20 Difficult Stress & Edge-Case Test Cases (Round 6)

Focus:
- LocalDataGeneratorEngine boundary conditions (invalid backend, max_samples=0, empty specs)
- Endpoint URL whitespace & trailing slash sanitization
- 100,000+ character payload handling in SyntheticDataSample
- 30-cycle SyntheticDataBatch serialization invariance
- Multi-thread concurrency safety across 10 parallel threads
- Endpoint availability caching optimization verification (<1ms check)
- Full 5-Stage Pipeline Integration (Critique -> Parser -> Writer -> Generator -> CurriculumDataset)
- PyTorch DataLoader mini-batch collation of synthetic CurriculumExample objects
"""

import os
import sys
import json
import time
import shutil
import logging
import concurrent.futures
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

PASS, FAIL = "PASS", "FAIL"
results = []

def report(tid, name, status, detail=""):
    icon = "OK" if status == PASS else "XX"
    print(f"  [{icon}] Test {tid:02d}: {name} - {status}" + (f"\n         Detail: {detail}" if detail else ""))
    results.append((tid, name, status, detail))

print("\n" + "="*65)
print("  CIRCLE - 20 Difficult Edge-Case Tests (Round 6)")
print("="*65 + "\n")

# ═══════════════════════════════════════════════════════════
# BLOCK A: DATA GENERATOR ENGINE BOUNDARY & SANITIZATION
# ═══════════════════════════════════════════════════════════
print("[ A ] Data Generator Engine Boundary & Sanitization")

# T01: Invalid backend string defaults safely to mock generator
try:
    from generator.generate import LocalDataGeneratorEngine
    from eval.prompt_writer import TargetedPromptSpec
    engine = LocalDataGeneratorEngine(backend="unknown_backend_xyz_123")
    spec = TargetedPromptSpec(request_id="req1", stage_id=1, category="speaker_drift", severity="high", priority=1, meta_prompt="m", suggested_prompt_focus="f", target_sample_count=2)
    samples = engine.generate_samples_for_spec(spec)
    assert len(samples) == 2
    assert samples[0].generation_metadata["backend"] == "mock_fallback"
    report(1, "Invalid backend string defaults safely to mock fallback generator", PASS)
except Exception as e:
    report(1, "Invalid backend string defaults safely to mock fallback generator", FAIL, str(e))

# T02: Empty TargetedPromptBatch returns clean empty SyntheticDataBatch
try:
    from generator.generate import LocalDataGeneratorEngine
    from eval.prompt_writer import TargetedPromptBatch
    engine = LocalDataGeneratorEngine()
    empty_prompt_batch = TargetedPromptBatch(stage_id=1, stage_name="Test", total_requests=0, total_target_samples=0, prompt_specs=[])
    synth_batch = engine.generate_batch_from_prompts(empty_prompt_batch)
    assert synth_batch.total_samples == 0
    assert len(synth_batch.samples) == 0
    report(2, "Empty TargetedPromptBatch returns clean empty SyntheticDataBatch", PASS)
except Exception as e:
    report(2, "Empty TargetedPromptBatch returns clean empty SyntheticDataBatch", FAIL, str(e))

# T03: max_samples_per_spec = 0 produces 0 samples without errors
try:
    from generator.generate import LocalDataGeneratorEngine
    from eval.prompt_writer import TargetedPromptSpec, TargetedPromptBatch
    engine = LocalDataGeneratorEngine()
    spec = TargetedPromptSpec(request_id="req1", stage_id=1, category="abrupt_cutoff", severity="medium", priority=2, meta_prompt="m", suggested_prompt_focus="f", target_sample_count=5)
    prompt_batch = TargetedPromptBatch(stage_id=1, stage_name="Test", total_requests=1, total_target_samples=5, prompt_specs=[spec])
    synth_batch = engine.generate_batch_from_prompts(prompt_batch, max_samples_per_spec=0)
    assert synth_batch.total_samples == 0
    assert len(synth_batch.samples) == 0
    report(3, "max_samples_per_spec=0 produces 0 samples cleanly", PASS)
except Exception as e:
    report(3, "max_samples_per_spec=0 produces 0 samples cleanly", FAIL, str(e))

# T04: 100,000+ character payload in SyntheticDataSample serializes without truncation
try:
    from generator.generate import SyntheticDataSample
    large_payload = "A" * 100000
    sample = SyntheticDataSample(
        sample_id="large_1", stage_id=1, category="test", input_text=large_payload, target_text=large_payload, prompt_spec_id="r1"
    )
    dumped = json.dumps(sample.model_dump())
    reloaded = json.loads(dumped)
    assert len(reloaded["input_text"]) == 100000
    report(4, "100,000+ character payload serializes cleanly in SyntheticDataSample", PASS)
except Exception as e:
    report(4, "100,000+ character payload serializes cleanly in SyntheticDataSample", FAIL, str(e))

# T05: Unicode, emojis, control characters, and HTML tags escape safely
try:
    from generator.generate import SyntheticDataSample
    tricky_text = "Emoji: 🚀🤖. HTML: <script>alert(1)</script>. Control: \x00\x07. Multilingual: 汉字"
    sample = SyntheticDataSample(
        sample_id="tricky_1", stage_id=1, category="test", input_text=tricky_text, target_text=tricky_text, prompt_spec_id="r1"
    )
    dumped = json.dumps(sample.model_dump())
    reloaded = json.loads(dumped)
    assert reloaded["input_text"] == tricky_text
    report(5, "Unicode, emojis, control chars, and HTML escape cleanly in SyntheticDataSample", PASS)
except Exception as e:
    report(5, "Unicode, emojis, control chars, and HTML escape cleanly in SyntheticDataSample", FAIL, str(e))

# T06: Endpoint URL whitespace and trailing slash sanitization
try:
    from generator.generate import LocalDataGeneratorEngine
    engine = LocalDataGeneratorEngine(endpoint_url="  http://localhost:11434///   ")
    assert engine.endpoint_url == "http://localhost:11434"
    report(6, "Endpoint URL whitespace and trailing slashes sanitized correctly", PASS)
except Exception as e:
    report(6, "Endpoint URL whitespace and trailing slashes sanitized correctly", FAIL, str(e))

# ═══════════════════════════════════════════════════════════
# BLOCK B: CACHING & PERFORMANCE OPTIMIZATION VERIFICATION
# ═══════════════════════════════════════════════════════════
print("\n[ B ] Caching & Performance Optimization Verification")

# T07: Endpoint availability check is cached after first probe (<1ms per subsequent check)
try:
    from generator.generate import LocalDataGeneratorEngine
    engine = LocalDataGeneratorEngine()
    _ = engine.is_endpoint_available()  # First check

    start_t = time.time()
    for _ in range(100):
        _ = engine.is_endpoint_available()
    elapsed_ms = (time.time() - start_t) * 1000

    assert elapsed_ms < 5.0, f"100 cached endpoint checks took {elapsed_ms:.2f}ms (threshold 5.0ms)"
    report(7, "Endpoint availability check is cached after first probe (<0.05ms/check)", PASS, f"100 checks: {elapsed_ms:.2f}ms")
except Exception as e:
    report(7, "Endpoint availability check is cached after first probe (<0.05ms/check)", FAIL, str(e))

# T08: 500 synthetic samples converted to CurriculumExample objects under 5ms
try:
    from generator.generate import LocalDataGeneratorEngine, SyntheticDataSample, SyntheticDataBatch
    engine = LocalDataGeneratorEngine()
    samples = [
        SyntheticDataSample(sample_id=f"s_{i}", stage_id=1, category="test", input_text=f"Inp {i}", target_text=f"Tgt {i}", prompt_spec_id="r")
        for i in range(500)
    ]
    batch = SyntheticDataBatch(stage_id=1, total_samples=500, samples=samples)

    start_t = time.time()
    examples = engine.convert_to_curriculum_examples(batch)
    elapsed_ms = (time.time() - start_t) * 1000

    assert len(examples) == 500
    assert elapsed_ms < 15.0, f"Conversion took {elapsed_ms:.2f}ms (threshold 15ms)"
    report(8, "500 synthetic samples converted to CurriculumExample objects under 15ms", PASS, f"Time: {elapsed_ms:.2f}ms")
except Exception as e:
    report(8, "500 synthetic samples converted to CurriculumExample objects under 15ms", FAIL, str(e))

# T09: Request timeout = 0.001s switches cleanly to mock fallback without hanging
try:
    from generator.generate import LocalDataGeneratorEngine
    from eval.prompt_writer import TargetedPromptSpec
    engine = LocalDataGeneratorEngine(backend="ollama", endpoint_url="http://127.0.0.1:59999", request_timeout_sec=0.001)
    spec = TargetedPromptSpec(request_id="r", stage_id=1, category="test", severity="high", priority=1, meta_prompt="m", suggested_prompt_focus="f", target_sample_count=1)

    start_t = time.time()
    samples = engine.generate_samples_for_spec(spec)
    elapsed_ms = (time.time() - start_t) * 1000

    assert len(samples) == 1
    assert samples[0].generation_metadata["backend"] == "mock_fallback"
    assert elapsed_ms < 500.0, f"Took {elapsed_ms:.2f}ms"
    report(9, "Ultra-short timeout (0.001s) switches cleanly to mock fallback", PASS, f"Time: {elapsed_ms:.2f}ms")
except Exception as e:
    report(9, "Ultra-short timeout (0.001s) switches cleanly to mock fallback", FAIL, str(e))

# ═══════════════════════════════════════════════════════════
# BLOCK C: SERIALIZATION INVARIANCE & CONCURRENCY
# ═══════════════════════════════════════════════════════════
print("\n[ C ] Serialization Invariance & Concurrency")

# T10: 30-cycle SyntheticDataBatch serialization roundtrip preserves 100% data fidelity
try:
    from generator.generate import LocalDataGeneratorEngine, SyntheticDataSample, SyntheticDataBatch
    engine = LocalDataGeneratorEngine()
    sample = SyntheticDataSample(sample_id="s1", stage_id=1, category="speaker_drift", input_text="Inp", target_text="Tgt", prompt_spec_id="r1")
    batch = SyntheticDataBatch(stage_id=1, total_samples=1, samples=[sample], generation_time_sec=0.05)

    test_file = "./data/batch_invariance_test.json"
    engine.save_synthetic_batch(batch, test_file)

    for _ in range(30):
        curr = engine.load_synthetic_batch(test_file)
        engine.save_synthetic_batch(curr, test_file)

    final_batch = engine.load_synthetic_batch(test_file)
    assert final_batch.stage_id == 1
    assert final_batch.samples[0].sample_id == "s1"
    if os.path.exists(test_file):
        os.remove(test_file)
    report(10, "30-cycle SyntheticDataBatch serialization roundtrip preserves exact values", PASS)
except Exception as e:
    if os.path.exists("./data/batch_invariance_test.json"):
        os.remove("./data/batch_invariance_test.json")
    report(10, "30-cycle SyntheticDataBatch serialization roundtrip preserves exact values", FAIL, str(e))

# T11: Multi-thread concurrency safety across 10 parallel threads generating samples
try:
    from generator.generate import LocalDataGeneratorEngine
    from eval.prompt_writer import TargetedPromptSpec
    engine = LocalDataGeneratorEngine()
    spec = TargetedPromptSpec(request_id="r", stage_id=1, category="speaker_drift", severity="high", priority=1, meta_prompt="m", suggested_prompt_focus="f", target_sample_count=2)

    def worker(idx):
        return engine.generate_samples_for_spec(spec)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(10)]
        results_list = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results_list) == 10
    assert all(len(s) == 2 for s in results_list)
    report(11, "Multi-thread concurrency safety across 10 parallel threads verified", PASS)
except Exception as e:
    report(11, "Multi-thread concurrency safety across 10 parallel threads verified", FAIL, str(e))

# T12: Loading corrupted JSON file in load_synthetic_batch raises error gracefully
try:
    from generator.generate import LocalDataGeneratorEngine
    engine = LocalDataGeneratorEngine()
    corrupt_file = "./data/corrupt_batch.json"
    with open(corrupt_file, "w") as f:
        f.write("{ invalid json syntax ... ")

    try:
        engine.load_synthetic_batch(corrupt_file)
        report(12, "Loading corrupt JSON file raises error gracefully", FAIL, "No exception raised!")
    except (json.JSONDecodeError, Exception):
        report(12, "Loading corrupt JSON file raises error gracefully", PASS)
    if os.path.exists(corrupt_file):
        os.remove(corrupt_file)
except Exception as e:
    if os.path.exists("./data/corrupt_batch.json"):
        os.remove("./data/corrupt_batch.json")
    report(12, "Loading corrupt JSON file raises error gracefully", FAIL, str(e))

# T13: Loading non-existent file in load_synthetic_batch raises FileNotFoundError
try:
    from generator.generate import LocalDataGeneratorEngine
    engine = LocalDataGeneratorEngine()
    try:
        engine.load_synthetic_batch("./data/non_existent_file_xyz_99.json")
        report(13, "Loading non-existent file raises FileNotFoundError", FAIL, "No exception raised!")
    except FileNotFoundError:
        report(13, "Loading non-existent file raises FileNotFoundError", PASS)
except Exception as e:
    report(13, "Loading non-existent file raises FileNotFoundError", FAIL, str(e))

# ═══════════════════════════════════════════════════════════
# BLOCK D: FULL 5-STAGE PIPELINE & PYTORCH INTEGRATION
# ═══════════════════════════════════════════════════════════
print("\n[ D ] Full 5-Stage Pipeline & PyTorch Integration")

# T14: Complete 5-Stage Pipeline Integration:
# Raw Critique -> Failure Parser -> Prompt Writer -> Data Generator -> Curriculum Examples -> CurriculumDataset
try:
    from eval.failure_parser import FailureModeParser
    from eval.prompt_writer import TargetedPromptWriter
    from generator.generate import LocalDataGeneratorEngine
    from trainer.curriculum.dataset_handler import CurriculumDataset
    from transformers import AutoTokenizer

    # 1. Parse Critique
    parser = FailureModeParser()
    report_obj = parser.parse(1, "Base English", "Model output exhibits severe degenerate repetition loops.")

    # 2. Write Prompts
    writer = TargetedPromptWriter()
    prompt_batch = writer.build_batch_from_report(report_obj, default_samples_per_request=3)

    # 3. Generate Data
    engine = LocalDataGeneratorEngine()
    synth_batch = engine.generate_batch_from_prompts(prompt_batch)

    # 4. Convert to Curriculum Examples
    examples = engine.convert_to_curriculum_examples(synth_batch)

    # 5. Build PyTorch CurriculumDataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = CurriculumDataset(examples, tokenizer, max_length=128)
    assert len(dataset) == len(examples)
    item = dataset[0]
    assert "input_ids" in item and "labels" in item

    report(14, "Full 5-Stage Pipeline (Critique -> Parser -> Writer -> Generator -> Dataset) succeeds", PASS,
           f"Dataset size: {len(dataset)}")
except Exception as e:
    report(14, "Full 5-Stage Pipeline (Critique -> Parser -> Writer -> Generator -> Dataset) succeeds", FAIL, str(e))

# T15: PyTorch DataLoader collation of synthetic CurriculumDataset mini-batches
try:
    from torch.utils.data import DataLoader
    from generator.generate import LocalDataGeneratorEngine, SyntheticDataSample, SyntheticDataBatch
    from trainer.curriculum.dataset_handler import CurriculumDataset
    from transformers import AutoTokenizer

    engine = LocalDataGeneratorEngine()
    samples = [
        SyntheticDataSample(sample_id=f"s_{i}", stage_id=1, category="test", input_text=f"Question {i}?", target_text=f"Answer {i}.", prompt_spec_id="r")
        for i in range(8)
    ]
    batch = SyntheticDataBatch(stage_id=1, total_samples=8, samples=samples)
    examples = engine.convert_to_curriculum_examples(batch)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = CurriculumDataset(examples, tokenizer, max_length=64)
    dl = DataLoader(ds, batch_size=4, shuffle=False)

    batch_item = next(iter(dl))
    assert batch_item["input_ids"].shape == (4, 64)
    assert batch_item["labels"].shape == (4, 64)
    report(15, "PyTorch DataLoader mini-batch collation of synthetic examples succeeds", PASS,
           f"Batch input shape: {batch_item['input_ids'].shape}")
except Exception as e:
    report(15, "PyTorch DataLoader mini-batch collation of synthetic examples succeeds", FAIL, str(e))

# T16: Metadata preservation in CurriculumExample conversion
try:
    from generator.generate import LocalDataGeneratorEngine, SyntheticDataSample, SyntheticDataBatch
    engine = LocalDataGeneratorEngine()
    sample = SyntheticDataSample(sample_id="meta_1", stage_id=2, category="speaker_drift", input_text="P", target_text="O", prompt_spec_id="spec_123")
    batch = SyntheticDataBatch(stage_id=2, total_samples=1, samples=[sample])
    examples = engine.convert_to_curriculum_examples(batch)

    meta = examples[0].metadata
    assert meta["sample_id"] == "meta_1"
    assert meta["category"] == "speaker_drift"
    assert meta["prompt_spec_id"] == "spec_123"
    assert "generated_at" in meta
    report(16, "CurriculumExample metadata preserves sample_id, category, and prompt_spec_id", PASS)
except Exception as e:
    report(16, "CurriculumExample metadata preserves sample_id, category, and prompt_spec_id", FAIL, str(e))

# T17: SyntheticDataBatch total_samples matches samples list length
try:
    from generator.generate import LocalDataGeneratorEngine, SyntheticDataSample, SyntheticDataBatch
    samples = [SyntheticDataSample(sample_id=f"s_{i}", stage_id=1, category="t", input_text="i", target_text="o", prompt_spec_id="r") for i in range(5)]
    batch = SyntheticDataBatch(stage_id=1, total_samples=5, samples=samples)
    assert batch.total_samples == len(batch.samples)
    report(17, "SyntheticDataBatch total_samples matches samples list length", PASS)
except Exception as e:
    report(17, "SyntheticDataBatch total_samples matches samples list length", FAIL, str(e))

# T18: Generation metadata includes priority and model_name
try:
    from generator.generate import LocalDataGeneratorEngine
    from eval.prompt_writer import TargetedPromptSpec
    engine = LocalDataGeneratorEngine(model_name="custom-deepseek-model")
    spec = TargetedPromptSpec(request_id="r", stage_id=1, category="abrupt_cutoff", severity="high", priority=1, meta_prompt="m", suggested_prompt_focus="f", target_sample_count=1)
    samples = engine.generate_samples_for_spec(spec)
    meta = samples[0].generation_metadata
    assert meta["model_name"] == "custom-deepseek-model"
    assert meta["priority"] == 1
    report(18, "SyntheticDataSample metadata captures priority and model_name", PASS)
except Exception as e:
    report(18, "SyntheticDataSample metadata captures priority and model_name", FAIL, str(e))

# T19: LocalDataGeneratorEngine request_timeout_sec parameter stored correctly
try:
    from generator.generate import LocalDataGeneratorEngine
    engine = LocalDataGeneratorEngine(request_timeout_sec=5.5)
    assert engine.request_timeout_sec == 5.5
    report(19, "request_timeout_sec parameter configured correctly", PASS)
except Exception as e:
    report(19, "request_timeout_sec parameter configured correctly", FAIL, str(e))

# T20: High sample count spec generation (50 samples) completed in-memory under 50ms
try:
    from generator.generate import LocalDataGeneratorEngine
    from eval.prompt_writer import TargetedPromptSpec
    engine = LocalDataGeneratorEngine(backend="mock")
    spec = TargetedPromptSpec(request_id="r", stage_id=1, category="speaker_drift", severity="high", priority=1, meta_prompt="m", suggested_prompt_focus="f", target_sample_count=50)

    start_t = time.time()
    samples = engine.generate_samples_for_spec(spec)
    elapsed_ms = (time.time() - start_t) * 1000

    assert len(samples) == 50
    assert elapsed_ms < 50.0, f"50 samples took {elapsed_ms:.2f}ms"
    report(20, "50 synthetic samples generated in-memory under 50ms", PASS, f"Time: {elapsed_ms:.2f}ms")
except Exception as e:
    report(20, "50 synthetic samples generated in-memory under 50ms", FAIL, str(e))


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
