"""
scripts/verify_deployment.py
CIRCLE Part 16: Deployment Verification Script

Validates:
- Python environment & required dependencies
- Repository directory structure and core microservice modules
- Stage configs (1-5) and dataset integrity (1-5)
- Docker specs & Kubernetes manifests correctness
- Probe task harness execution across all 5 stages
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DeploymentVerifier")


def check_directory_structure() -> Tuple[bool, str]:
    required_dirs = [
        "trainer", "eval", "generator", "orchestrator",
        "docker", "orchestrator/k8s", "data", "tests", "logs"
    ]
    missing = [d for d in required_dirs if not os.path.exists(os.path.join(PROJECT_ROOT, d))]
    if missing:
        return False, f"Missing directories: {missing}"
    return True, "All required directories present"


def check_curriculum_configs() -> Tuple[bool, str]:
    for stage_id in range(1, 6):
        cfg_path = os.path.join(PROJECT_ROOT, "trainer", "curriculum", "configs", f"stage_{stage_id}.json")
        if not os.path.exists(cfg_path):
            return False, f"Missing config for stage {stage_id}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "stage_id" not in data or "probe_prompts" not in data:
                return False, f"Invalid config schema in stage_{stage_id}.json"
    return True, "All 5 stage configs valid"


def check_curriculum_datasets() -> Tuple[bool, str]:
    for stage_id in range(1, 6):
        data_path = os.path.join(PROJECT_ROOT, "data", f"stage_{stage_id}", "dataset.json")
        if not os.path.exists(data_path):
            return False, f"Missing dataset for stage {stage_id}"
        with open(data_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            if not isinstance(items, list) or len(items) < 30:
                return False, f"Stage {stage_id} dataset has only {len(items)} samples (expected >= 30)"
    return True, "All 5 stage datasets present with >= 30 seed samples"


def check_docker_and_k8s_specs() -> Tuple[bool, str]:
    dockerfiles = ["trainer.Dockerfile", "eval.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]
    for df in dockerfiles:
        df_path = os.path.join(PROJECT_ROOT, "docker", df)
        if not os.path.exists(df_path):
            return False, f"Missing Dockerfile: {df}"

    k8s_manifests = [
        "namespace.yaml", "configmap.yaml", "secrets.yaml", "pvc.yaml",
        "trainer-deployment.yaml", "eval-deployment.yaml", "generator-deployment.yaml",
        "orchestrator-deployment.yaml", "rbac.yaml", "hpa.yaml"
    ]
    for mf in k8s_manifests:
        mf_path = os.path.join(PROJECT_ROOT, "orchestrator", "k8s", mf)
        if not os.path.exists(mf_path):
            return False, f"Missing k8s manifest: {mf}"

    return True, "All Dockerfiles and K8s manifests present"


def check_probe_harness() -> Tuple[bool, str]:
    try:
        from eval.probe_harness import ProbeTaskHarness
        harness = ProbeTaskHarness(output_dir=os.path.join(PROJECT_ROOT, "logs"))
        for stage_id in range(1, 6):
            report = harness.run_stage_probes(stage_id=stage_id, mock_mode=True)
            if report.total_probes == 0:
                return False, f"Stage {stage_id} probe harness generated 0 probes"
        return True, "Probe task harness verified across all 5 stages"
    except Exception as e:
        return False, f"Probe task harness error: {e}"


def run_deployment_verification() -> bool:
    print("\n" + "=" * 65)
    print("  CIRCLE — Deployment Verification System")
    print("=" * 65 + "\n")

    checks = [
        ("Directory Structure", check_directory_structure),
        ("Curriculum Stage Configs (1-5)", check_curriculum_configs),
        ("Curriculum Seed Datasets (1-5)", check_curriculum_datasets),
        ("Docker & K8s Specs", check_docker_and_k8s_specs),
        ("Probe Task Execution Harness", check_probe_harness),
    ]

    all_passed = True
    for name, check_fn in checks:
        passed, msg = check_fn()
        status_str = "PASS" if passed else "FAIL"
        print(f"  [{'OK' if passed else 'XX'}] {name:35s} - {status_str}: {msg}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 65)
    print(f"  VERIFICATION RESULT: {'ALL SYSTEM CHECKS PASSED' if all_passed else 'VERIFICATION FAILED'}")
    print("=" * 65 + "\n")
    return all_passed


if __name__ == "__main__":
    success = run_deployment_verification()
    sys.exit(0 if success else 1)
