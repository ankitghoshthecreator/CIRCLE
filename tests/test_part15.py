import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.probe_harness import (
    ProbeTaskHarness,
    ProbeTaskResult,
    ProbeExecutionReport,
    RuleScoreDetail,
    evaluate_stage_4_punctuation,
    evaluate_stage_5_dialogue,
    evaluate_general_probe
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestPart15ProbeHarness(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.harness = ProbeTaskHarness(output_dir=self.temp_dir)

    def test_01_probe_harness_instantiation(self):
        """Test ProbeTaskHarness initialization and directory creation."""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertEqual(self.harness.output_dir, self.temp_dir)

    def test_02_stage_4_rule_evaluator_punctuation(self):
        """Test Stage 4 punctuation rules on clean vs bad completions."""
        # Clean completion
        clean_rules = evaluate_stage_4_punctuation(
            "Add punctuation to: although it was raining we went out",
            "Although it was raining, we went out."
        )
        scores = {r.rule_name: r.score for r in clean_rules}
        self.assertEqual(scores["non_empty_completion"], 1.0)
        self.assertEqual(scores["terminal_punctuation"], 1.0)
        self.assertEqual(scores["sentence_capitalization"], 1.0)
        self.assertEqual(scores["quote_balance"], 1.0)

        # Bad completion (missing terminal punct, uncapitalized, unbalanced quote)
        bad_rules = evaluate_stage_4_punctuation(
            "Punctuate",
            'although "it was raining we went out'
        )
        bad_scores = {r.rule_name: r.score for r in bad_rules}
        self.assertLess(bad_scores["terminal_punctuation"], 1.0)
        self.assertLess(bad_scores["sentence_capitalization"], 1.0)
        self.assertLess(bad_scores["quote_balance"], 1.0)

    def test_03_stage_5_rule_evaluator_dialogue(self):
        """Test Stage 5 dialogue rules on clean vs bad completions."""
        # Clean dialogue completion
        clean_rules = evaluate_stage_5_dialogue(
            "User: Explain CIRCLE.\nAssistant:",
            "CIRCLE is a closed-loop curriculum training framework for language models."
        )
        scores = {r.rule_name: r.score for r in clean_rules}
        self.assertEqual(scores["response_substance"], 1.0)
        self.assertEqual(scores["speaker_turn_adherence"], 1.0)
        self.assertEqual(scores["helpfulness_relevance"], 1.0)
        self.assertEqual(scores["no_repetitive_looping"], 1.0)

        # Bad dialogue completion (user hijack, repetitive refusal)
        bad_rules = evaluate_stage_5_dialogue(
            "User: Help\nAssistant:",
            "User: I cannot help I cannot help I cannot help"
        )
        bad_scores = {r.rule_name: r.score for r in bad_rules}
        self.assertLess(bad_scores["speaker_turn_adherence"], 1.0)
        self.assertLess(bad_scores["helpfulness_relevance"], 1.0)

    def test_04_general_rule_evaluator(self):
        """Test general fallback probe evaluator."""
        rules = evaluate_general_probe("Test prompt", "Valid completion text", stage_id=1)
        self.assertGreaterEqual(len(rules), 2)
        self.assertTrue(all(r.score > 0.0 for r in rules))

    def test_05_mock_completion_generator(self):
        """Test mock completion generator for all 5 stages."""
        for stage_id in range(1, 6):
            comp = self.harness.generate_mock_completion(stage_id, "Sample prompt")
            self.assertIsInstance(comp, str)
            self.assertGreater(len(comp), 5)

    def test_06_evaluate_probe_single_item(self):
        """Test single probe scoring via evaluate_probe method."""
        res = self.harness.evaluate_probe(
            probe_id=1,
            stage_id=4,
            prompt="Punctuate: hello world",
            completion="Hello world.",
            target_score=0.80
        )
        self.assertIsInstance(res, ProbeTaskResult)
        self.assertEqual(res.probe_id, 1)
        self.assertEqual(res.stage_id, 4)
        self.assertTrue(res.passed)
        self.assertGreaterEqual(res.overall_score, 0.80)

    def test_07_run_stage_probes_stage_4(self):
        """Test full stage probe suite execution for Stage 4."""
        report = self.harness.run_stage_probes(stage_id=4, mock_mode=True)
        self.assertIsInstance(report, ProbeExecutionReport)
        self.assertEqual(report.stage_id, 4)
        self.assertEqual(report.total_probes, 10)
        self.assertGreaterEqual(report.passed_probes, 8)
        self.assertTrue(report.advance_threshold_met)

    def test_08_run_stage_probes_stage_5(self):
        """Test full stage probe suite execution for Stage 5."""
        report = self.harness.run_stage_probes(stage_id=5, mock_mode=True)
        self.assertIsInstance(report, ProbeExecutionReport)
        self.assertEqual(report.stage_id, 5)
        self.assertEqual(report.total_probes, 10)
        self.assertTrue(report.advance_threshold_met)

    def test_09_probe_execution_report_pydantic_schema(self):
        """Test Pydantic dump and schema validation for ProbeExecutionReport."""
        report = self.harness.run_stage_probes(stage_id=4, mock_mode=True)
        dumped = report.model_dump()
        self.assertIn("stage_id", dumped)
        self.assertIn("detailed_results", dumped)
        self.assertIn("summary_by_rule", dumped)
        reloaded = ProbeExecutionReport(**dumped)
        self.assertEqual(reloaded.stage_id, 4)

    def test_10_save_report_json_persisted(self):
        """Test save_report persists JSON artifact to disk correctly."""
        report = self.harness.run_stage_probes(stage_id=4, mock_mode=True)
        path = self.harness.save_report(report, filename="test_report.json")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["stage_id"], 4)
        self.assertEqual(data["total_probes"], 10)

    def test_11_probes_override_custom(self):
        """Test custom probes_override list parameter in run_stage_probes."""
        custom_probes = [
            "Punctuate: custom test prompt one",
            "Punctuate: custom test prompt two"
        ]
        report = self.harness.run_stage_probes(
            stage_id=4,
            mock_mode=True,
            probes_override=custom_probes
        )
        self.assertEqual(report.total_probes, 2)
        self.assertEqual(len(report.detailed_results), 2)

    def test_12_advance_threshold_met_logic(self):
        """Test advance_threshold_met evaluates to False when target score is unachievable."""
        report = self.harness.run_stage_probes(
            stage_id=4,
            mock_mode=True,
            target_score_override=1.05  # Impossible target score
        )
        self.assertFalse(report.advance_threshold_met)

    def test_13_summary_by_rule_aggregation(self):
        """Test summary_by_rule aggregates rule scores across probes."""
        report = self.harness.run_stage_probes(stage_id=4, mock_mode=True)
        summary = report.summary_by_rule
        self.assertIn("terminal_punctuation", summary)
        self.assertIn("sentence_capitalization", summary)
        self.assertGreaterEqual(summary["terminal_punctuation"], 0.0)

    def test_14_stage_4_dataset_and_probe_alignment(self):
        """Test Stage 4 probe prompts match Stage 4 curriculum config."""
        from trainer.curriculum.dataset_handler import load_stage_config
        cfg = load_stage_config(4)
        probes = cfg["probe_prompts"]
        self.assertGreaterEqual(len(probes), 8)
        self.assertTrue(any("punctuation" in p.lower() or "punctuate" in p.lower() for p in probes))

    def test_15_stage_5_dialogue_multi_turn_probes(self):
        """Test Stage 5 probe prompts contain multi-turn turn indicators."""
        from trainer.curriculum.dataset_handler import load_stage_config
        cfg = load_stage_config(5)
        probes = cfg["probe_prompts"]
        self.assertGreaterEqual(len(probes), 8)
        has_multi_turn = any("User:" in p and "Assistant:" in p for p in probes)
        self.assertTrue(has_multi_turn)

    def test_16_rule_score_detail_schema(self):
        """Test RuleScoreDetail initialization and field validation."""
        rule = RuleScoreDetail(rule_name="test_rule", score=0.95, passed=True, reason="All good")
        self.assertEqual(rule.rule_name, "test_rule")
        self.assertEqual(rule.score, 0.95)
        self.assertTrue(rule.passed)

    def test_17_failure_tags_captured(self):
        """Test that failing rule names are appended to failure_tags."""
        bad_completion = "bad text with no end"
        res = self.harness.evaluate_probe(
            probe_id=1,
            stage_id=4,
            prompt="Punctuate me",
            completion=bad_completion,
            target_score=0.95
        )
        self.assertIsInstance(res.failure_tags, list)

    def test_18_cli_entrypoint_execution(self):
        """Test main CLI entrypoint execution via subprocess dry-run."""
        import subprocess
        cmd = [sys.executable, "-m", "eval.probe_harness", "--stage", "4", "--output-dir", self.temp_dir]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Stage 4 (Punctuation) Probe Report", res.stdout)

    def test_19_target_score_override(self):
        """Test target_score_override parameter in run_stage_probes."""
        report = self.harness.run_stage_probes(
            stage_id=4,
            mock_mode=True,
            target_score_override=0.50
        )
        self.assertEqual(report.target_competence_score, 0.50)

    def test_20_all_5_stages_harness_execution(self):
        """Test running probe task harness across all 5 curriculum stages (1 to 5)."""
        for s in range(1, 6):
            rep = self.harness.run_stage_probes(stage_id=s, mock_mode=True)
            self.assertEqual(rep.stage_id, s)
            self.assertGreater(rep.total_probes, 0)


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  CIRCLE Part 15 — Stage 4-5 Probes & Harness Test Suite")
    print("="*65 + "\n")
    unittest.main()
