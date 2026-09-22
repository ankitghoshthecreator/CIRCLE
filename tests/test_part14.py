import os
import json
import sys
import unittest
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.curriculum.dataset_handler import (
    load_stage_config,
    load_stage_dataset,
    CurriculumExample,
    CurriculumDataset
)
from trainer.curriculum.stage_config import STAGES, CurriculumStage


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainer", "curriculum", "configs")


class MockTokenizer:
    """Mock HuggingFace-style tokenizer for testing PyTorch dataset wrapper."""
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text, truncation=True, max_length=128, padding="max_length", return_tensors="pt"):
        try:
            import torch
            tokens = [101] + [ord(c) % 1000 for c in text[:max_length - 2]] + [102]
            if len(tokens) < max_length:
                tokens = tokens + [0] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
            
            input_ids = torch.tensor([tokens], dtype=torch.long)
            mask = torch.tensor([[1 if t != 0 else 0 for t in tokens]], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": mask}
        except ImportError:
            # Fallback if torch is not installed on CPU host
            return {"input_ids": [101, 102], "attention_mask": [1, 1]}


class TestPart14CurriculumExpansion(unittest.TestCase):

    def test_01_stage_configs_exist_all_5_stages(self):
        """Test that stage_1.json through stage_5.json exist in configs dir."""
        for stage_id in range(1, 6):
            config_file = os.path.join(CONFIG_DIR, f"stage_{stage_id}.json")
            self.assertTrue(os.path.exists(config_file), f"Config file for stage {stage_id} missing: {config_file}")

    def test_02_stage_config_schema_validation(self):
        """Test required schema fields in all 5 JSON configs."""
        required_keys = [
            "stage_id", "name", "description", "objectives",
            "target_competence_score", "hyperparameters", "replay_ratio", "probe_prompts"
        ]
        for stage_id in range(1, 6):
            cfg = load_stage_config(stage_id)
            for k in required_keys:
                self.assertIn(k, cfg, f"Stage {stage_id} config missing required key: '{k}'")
            self.assertEqual(cfg["stage_id"], stage_id)
            self.assertIsInstance(cfg["objectives"], list)
            self.assertGreater(len(cfg["objectives"]), 0)

    def test_03_stage_config_probe_counts(self):
        """Test that each stage config contains at least 8 probe prompts."""
        for stage_id in range(1, 6):
            cfg = load_stage_config(stage_id)
            probes = cfg.get("probe_prompts", [])
            self.assertGreaterEqual(
                len(probes), 8,
                f"Stage {stage_id} has {len(probes)} probes, expected >= 8"
            )

    def test_04_stage_config_advance_criteria(self):
        """Test advance_criteria structure across all 5 configs."""
        for stage_id in range(1, 6):
            cfg = load_stage_config(stage_id)
            self.assertIn("advance_criteria", cfg, f"Stage {stage_id} missing advance_criteria")
            ac = cfg["advance_criteria"]
            self.assertIn("min_mean_score", ac)
            self.assertIn("max_severity_allowed", ac)
            self.assertIn("min_probe_pass_rate", ac)
            self.assertGreaterEqual(ac["min_mean_score"], 0.70)

    def test_05_stage_config_expected_behaviors(self):
        """Test probe_expected_behaviors list across all 5 configs."""
        for stage_id in range(1, 6):
            cfg = load_stage_config(stage_id)
            self.assertIn("probe_expected_behaviors", cfg, f"Stage {stage_id} missing probe_expected_behaviors")
            eb = cfg["probe_expected_behaviors"]
            self.assertIsInstance(eb, list)
            self.assertGreaterEqual(len(eb), 3, f"Stage {stage_id} expected behaviors < 3")

    def test_06_stage_datasets_exist_all_5_stages(self):
        """Test that dataset.json exists for stages 1 through 5 in data_dir."""
        for stage_id in range(1, 6):
            data_file = os.path.join(DATA_DIR, f"stage_{stage_id}", "dataset.json")
            self.assertTrue(os.path.exists(data_file), f"Dataset file for stage {stage_id} missing: {data_file}")

    def test_07_dataset_json_validity(self):
        """Test that all stage dataset files parse as valid JSON arrays."""
        for stage_id in range(1, 6):
            data_file = os.path.join(DATA_DIR, f"stage_{stage_id}", "dataset.json")
            with open(data_file, "r", encoding="utf-8") as f:
                content = json.load(f)
            self.assertIsInstance(content, list, f"Stage {stage_id} dataset is not a JSON list")

    def test_08_dataset_minimum_sample_counts(self):
        """Test that each stage dataset contains at least 30 seed examples."""
        for stage_id in range(1, 6):
            data_file = os.path.join(DATA_DIR, f"stage_{stage_id}", "dataset.json")
            with open(data_file, "r", encoding="utf-8") as f:
                content = json.load(f)
            self.assertGreaterEqual(
                len(content), 30,
                f"Stage {stage_id} has {len(content)} dataset samples, expected >= 30"
            )

    def test_09_dataset_pydantic_validation(self):
        """Test that every example in all 5 datasets validates with CurriculumExample."""
        for stage_id in range(1, 6):
            data_file = os.path.join(DATA_DIR, f"stage_{stage_id}", "dataset.json")
            with open(data_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            for idx, item in enumerate(items):
                try:
                    ex = CurriculumExample(**item)
                    self.assertIsInstance(ex, CurriculumExample)
                except Exception as ve:
                    self.fail(f"Stage {stage_id} item #{idx} failed Pydantic validation: {ve}")

    def test_10_dataset_stage_id_consistency(self):
        """Test that stage_id in each dataset item matches its directory stage ID."""
        for stage_id in range(1, 6):
            examples = load_stage_dataset(stage_id, data_dir=DATA_DIR)
            for ex in examples:
                self.assertEqual(
                    ex.stage_id, stage_id,
                    f"Example stage_id {ex.stage_id} does not match directory stage {stage_id}"
                )

    def test_11_dataset_non_empty_text(self):
        """Test that input_text is non-empty for all examples across all 5 datasets."""
        for stage_id in range(1, 6):
            examples = load_stage_dataset(stage_id, data_dir=DATA_DIR)
            for idx, ex in enumerate(examples):
                self.assertTrue(
                    len(ex.input_text.strip()) > 0,
                    f"Stage {stage_id} item #{idx} has empty input_text"
                )

    def test_12_dataset_target_text_presence(self):
        """Test target_text presence for stages 2..5."""
        for stage_id in range(2, 6):
            examples = load_stage_dataset(stage_id, data_dir=DATA_DIR)
            for idx, ex in enumerate(examples):
                self.assertTrue(
                    len(ex.target_text.strip()) > 0,
                    f"Stage {stage_id} item #{idx} missing target_text completion"
                )

    def test_13_dataset_metadata_schema(self):
        """Test metadata is dict type for all items across datasets."""
        for stage_id in range(1, 6):
            examples = load_stage_dataset(stage_id, data_dir=DATA_DIR)
            for ex in examples:
                self.assertIsInstance(ex.metadata, dict)

    def test_14_curriculum_loader_load_stage_config(self):
        """Test load_stage_config for all 5 stages."""
        for stage_id in range(1, 6):
            cfg = load_stage_config(stage_id)
            self.assertEqual(cfg["stage_id"], stage_id)

    def test_15_curriculum_loader_load_stage_dataset(self):
        """Test load_stage_dataset helper returns List[CurriculumExample]."""
        for stage_id in range(1, 6):
            dataset = load_stage_dataset(stage_id, data_dir=DATA_DIR, use_seed_fallback=False)
            self.assertIsInstance(dataset, list)
            self.assertGreater(len(dataset), 0)

    def test_16_curriculum_stages_dict_populates_all_5(self):
        """Test STAGES dictionary in stage_config populates stages 1 to 5."""
        self.assertEqual(len(STAGES), 5)
        for stage_id in range(1, 6):
            self.assertIn(stage_id, STAGES)
            stage_obj = STAGES[stage_id]
            self.assertIsInstance(stage_obj, CurriculumStage)
            self.assertEqual(stage_obj.stage_id, stage_id)

    def test_17_pytorch_curriculum_dataset_formatting(self):
        """Test PyTorch CurriculumDataset dataset formatting and tensors."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed on host environment")

        tokenizer = MockTokenizer()
        examples = load_stage_dataset(1, data_dir=DATA_DIR)[:5]
        ds = CurriculumDataset(examples, tokenizer, max_length=64)
        self.assertEqual(len(ds), 5)
        item = ds[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        self.assertEqual(item["input_ids"].shape[0], 64)
        self.assertEqual(item["attention_mask"].shape[0], 64)

    def test_18_stage_difficulty_metadata_coverage(self):
        """Test that datasets include rich metadata keys (category/topic/difficulty)."""
        for stage_id in range(1, 6):
            examples = load_stage_dataset(stage_id, data_dir=DATA_DIR)
            has_meta_keys = any(len(ex.metadata) > 0 for ex in examples)
            self.assertTrue(
                has_meta_keys,
                f"Stage {stage_id} dataset missing metadata key annotations"
            )

    def test_19_replay_ratio_range(self):
        """Test replay_ratio is valid float between 0.0 and 0.5."""
        for stage_id in range(1, 6):
            cfg = load_stage_config(stage_id)
            ratio = cfg.get("replay_ratio", 0.0)
            self.assertGreaterEqual(ratio, 0.0)
            self.assertLessEqual(ratio, 0.5)

    def test_20_hyperparameters_completeness(self):
        """Test hyperparameter completeness across all 5 stage configs."""
        required_hp = ["default_epochs", "batch_size", "grad_accum_steps", "learning_rate"]
        for stage_id in range(1, 6):
            cfg = load_stage_config(stage_id)
            hp = cfg.get("hyperparameters", {})
            for key in required_hp:
                self.assertIn(key, hp, f"Stage {stage_id} missing hyperparameter '{key}'")


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  CIRCLE Part 14 — Curriculum Pipeline & Seed Datasets Suite")
    print("="*65 + "\n")
    unittest.main()
