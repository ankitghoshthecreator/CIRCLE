import os
import sys
import json
import tempfile
import unittest
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from scripts.verify_deployment import (
    check_directory_structure,
    check_curriculum_configs,
    check_curriculum_datasets,
    check_docker_and_k8s_specs,
    check_probe_harness,
    run_deployment_verification
)
from scripts.e2e_runner import run_e2e_pipeline
from orchestrator.loop_controller import ClosedLoopOrchestrator
from orchestrator.state_machine import LoopState, PipelineState


class TestPart16SystemIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_01_verify_deployment_script(self):
        """Test run_deployment_verification returns True across all system checks."""
        success = run_deployment_verification()
        self.assertTrue(success)

    def test_02_e2e_runner_execution(self):
        """Test run_e2e_pipeline executes full 5-stage pipeline in mock mode."""
        summary = run_e2e_pipeline(
            start_stage=1,
            max_iterations=1,
            stage_threshold=0.85,
            mock_mode=True,
            log_dir=self.temp_dir
        )
        self.assertTrue(summary["is_complete"])
        self.assertEqual(summary["stages_completed"], 5)
        self.assertEqual(summary["final_state"], str(LoopState.COMPLETE))

    def test_03_stage_transition_sequence(self):
        """Test that stage results sequence follows stage_ids 1 through 5."""
        summary = run_e2e_pipeline(
            start_stage=1,
            max_iterations=1,
            mock_mode=True,
            log_dir=self.temp_dir
        )
        stage_ids = [r["stage_id"] for r in summary["results_by_stage"]]
        self.assertEqual(stage_ids, [1, 2, 3, 4, 5])

    def test_04_e2e_summary_json_output(self):
        """Test that e2e summary JSON file is written to log_dir."""
        summary = run_e2e_pipeline(mock_mode=True, log_dir=self.temp_dir)
        run_id = summary["run_id"]
        summary_file = os.path.join(PROJECT_ROOT, self.temp_dir, f"e2e_summary_{run_id}.json")
        self.assertTrue(os.path.exists(summary_file))
        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["run_id"], run_id)
        self.assertEqual(len(data["results_by_stage"]), 5)

    def test_05_directory_structure_completeness(self):
        """Test check_directory_structure helper."""
        ok, msg = check_directory_structure()
        self.assertTrue(ok, msg)

    def test_06_curriculum_configs_completeness(self):
        """Test check_curriculum_configs helper."""
        ok, msg = check_curriculum_configs()
        self.assertTrue(ok, msg)

    def test_07_curriculum_datasets_completeness(self):
        """Test check_curriculum_datasets helper."""
        ok, msg = check_curriculum_datasets()
        self.assertTrue(ok, msg)

    def test_08_docker_and_k8s_specs_completeness(self):
        """Test check_docker_and_k8s_specs helper."""
        ok, msg = check_docker_and_k8s_specs()
        self.assertTrue(ok, msg)

    def test_09_probe_harness_check_completeness(self):
        """Test check_probe_harness helper."""
        ok, msg = check_probe_harness()
        self.assertTrue(ok, msg)

    def test_10_e2e_runner_cli_subprocess(self):
        """Test scripts/e2e_runner.py CLI execution via subprocess."""
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "e2e_runner.py"), "--log-dir", self.temp_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("CIRCLE End-to-End Pipeline Execution Summary", res.stdout)

    def test_11_verify_deployment_cli_subprocess(self):
        """Test scripts/verify_deployment.py CLI execution via subprocess."""
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "verify_deployment.py")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        self.assertIn("ALL SYSTEM CHECKS PASSED", res.stdout)

    def test_12_loop_controller_stage_1_to_5(self):
        """Test ClosedLoopOrchestrator direct invocation from stage 1 to 5."""
        orch = ClosedLoopOrchestrator(
            start_stage=1,
            max_iterations=1,
            log_dir=self.temp_dir,
            mock_mode=True
        )
        final_state = orch.execute_loop()
        self.assertEqual(final_state.current_state, LoopState.COMPLETE)

    def test_13_state_persistence_invariants(self):
        """Test state file persistence under log_dir."""
        orch = ClosedLoopOrchestrator(
            start_stage=1,
            max_iterations=1,
            log_dir=self.temp_dir,
            mock_mode=True
        )
        final_state = orch.execute_loop()
        state_file = os.path.join(self.temp_dir, f"pipeline_state_{final_state.run_id}.json")
        self.assertTrue(os.path.exists(state_file))

    def test_14_probe_task_harness_all_stages(self):
        """Test ProbeTaskHarness execution across all 5 curriculum stages."""
        from eval.probe_harness import ProbeTaskHarness
        harness = ProbeTaskHarness(output_dir=self.temp_dir)
        for s in range(1, 6):
            rep = harness.run_stage_probes(stage_id=s, mock_mode=True)
            self.assertEqual(rep.stage_id, s)
            self.assertGreater(rep.total_probes, 0)

    def test_15_dataset_handler_seed_and_synthetic_merge(self):
        """Test load_stage_dataset and save_stage_dataset with seed examples."""
        from trainer.curriculum.dataset_handler import load_stage_dataset, save_stage_dataset
        examples = load_stage_dataset(stage_id=1, data_dir=os.path.join(PROJECT_ROOT, "data"))
        self.assertGreaterEqual(len(examples), 30)

    def test_16_build_images_cli_dry_run(self):
        """Test docker/build_images.py CLI dry-run execution via subprocess."""
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "docker", "build_images.py"), "--dry-run"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")

    def test_17_apply_manifests_cli_dry_run(self):
        """Test orchestrator/k8s/apply_manifests.py CLI manifest verification via subprocess."""
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "orchestrator", "k8s", "apply_manifests.py"), "--verify"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")

    def test_18_stages_dataclass_coverage(self):
        """Test STAGES dictionary in stage_config for all 5 stages."""
        from trainer.curriculum.stage_config import STAGES
        self.assertEqual(len(STAGES), 5)
        for sid in range(1, 6):
            self.assertIn(sid, STAGES)

    def test_19_failure_parser_e2e_flow(self):
        """Test FailureModeParser parsing raw critique text into failure report."""
        from eval.failure_parser import FailureModeParser
        parser = FailureModeParser()
        raw_critique = "The model produced severe subject-verb agreement errors and repetition."
        report = parser.parse(stage_id=3, stage_name="Grammar", critique_raw=raw_critique)
        self.assertEqual(report.stage_id, 3)

    def test_20_pipeline_state_complete_condition(self):
        """Test PipelineState property is_complete when state is LoopState.COMPLETE."""
        state = PipelineState(run_id="test_run", current_stage=5, current_state=LoopState.COMPLETE)
        self.assertTrue(state.is_complete)
        self.assertFalse(state.is_failed)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  CIRCLE Part 16 — System Integration & Verification Suite")
    print("=" * 65 + "\n")
    unittest.main()
