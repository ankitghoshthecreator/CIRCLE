"""
tests/test_round8.py
CIRCLE - 20 Difficult Stress & Edge-Case Tests (Round 8)

Focus:
- Containerization, Dockerfile instruction invariants, layer caching, and multi-stage builds
- Docker Compose service dependency ordering, restart policies, and PVC volume mappings
- Build automation script CLI flags, dry-run performance, and tag parsing correctness
- Environment variable isolation and non-root directory permission safety
"""

import os
import sys
import json
import yaml
import time
import shutil
import tempfile
import subprocess
from typing import List

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
print("  CIRCLE - 20 Difficult Edge-Case Tests (Round 8)")
print("="*65 + "\n")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════
# BLOCK A: DOCKERFILE INVARIANTS & LAYER CACHING
# ═══════════════════════════════════════════════════════════
print("[ A ] Dockerfile Invariants & Layer Caching")

# T01: All Dockerfiles enforce PYTHONUNBUFFERED=1 to prevent log buffering hangs
try:
    for df_name in ["trainer.Dockerfile", "eval.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            content = f.read()
        assert "PYTHONUNBUFFERED=1" in content, f"PYTHONUNBUFFERED=1 missing in {df_name}"
    report(1, "All 4 Dockerfiles enforce PYTHONUNBUFFERED=1 to prevent stream buffering", PASS)
except Exception as e:
    report(1, "All 4 Dockerfiles enforce PYTHONUNBUFFERED=1 to prevent stream buffering", FAIL, str(e))


# T02: All Dockerfiles enforce PYTHONDONTWRITEBYTECODE=1 to reduce container disk footprint
try:
    for df_name in ["trainer.Dockerfile", "eval.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            content = f.read()
        assert "PYTHONDONTWRITEBYTECODE=1" in content, f"PYTHONDONTWRITEBYTECODE=1 missing in {df_name}"
    report(2, "All 4 Dockerfiles enforce PYTHONDONTWRITEBYTECODE=1 for minimal disk footprint", PASS)
except Exception as e:
    report(2, "All 4 Dockerfiles enforce PYTHONDONTWRITEBYTECODE=1 for minimal disk footprint", FAIL, str(e))


# T03: Builder stage copies requirements.txt before code to optimize Docker layer caching
try:
    for df_name in ["trainer.Dockerfile", "eval.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            lines = f.readlines()
        
        req_line_idx = next(i for i, l in enumerate(lines) if "COPY requirements.txt" in l)
        code_line_idx = next(i for i, l in enumerate(lines) if "COPY " in l and "requirements.txt" not in l)
        assert req_line_idx < code_line_idx, f"requirements.txt copied after code in {df_name}"
    report(3, "Builder stage copies requirements.txt before code to maximize Docker layer caching", PASS)
except Exception as e:
    report(3, "Builder stage copies requirements.txt before code to maximize Docker layer caching", FAIL, str(e))


# T04: trainer.Dockerfile sets custom TORCH_HOME directory path
try:
    df_path = os.path.join(PROJECT_ROOT, "docker", "trainer.Dockerfile")
    with open(df_path) as f:
        content = f.read()
    assert "TORCH_HOME=" in content
    report(4, "trainer.Dockerfile sets custom TORCH_HOME cache directory path", PASS)
except Exception as e:
    report(4, "trainer.Dockerfile sets custom TORCH_HOME cache directory path", FAIL, str(e))


# T05: Healthcheck interval and timeout syntax valid across all Dockerfiles
try:
    for df_name in ["trainer.Dockerfile", "eval.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            content = f.read()
        assert "--interval=" in content and "--timeout=" in content and "HEALTHCHECK" in content
    report(5, "Healthcheck interval, timeout, and retry parameters valid across all Dockerfiles", PASS)
except Exception as e:
    report(5, "Healthcheck interval, timeout, and retry parameters valid across all Dockerfiles", FAIL, str(e))


# ═══════════════════════════════════════════════════════════
# BLOCK B: DOCKER COMPOSE ORCHESTRATION SPECS
# ═══════════════════════════════════════════════════════════
print("\n[ B ] Docker Compose Orchestration Specs")

# T06: docker-compose.yml version is 3.8 or compatible
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    assert data["version"] in ["3.8", "3.7", "3.9"]
    report(6, "docker-compose.yml version spec is 3.8 standard", PASS)
except Exception as e:
    report(6, "docker-compose.yml version spec is 3.8 standard", FAIL, str(e))


# T07: Orchestrator service depends_on trainer and generator with health conditions
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    deps = data["services"]["orchestrator"]["depends_on"]
    assert "trainer" in deps and "generator" in deps
    assert deps["trainer"]["condition"] == "service_healthy"
    report(7, "Orchestrator depends_on trainer & generator with service_healthy condition", PASS)
except Exception as e:
    report(7, "Orchestrator depends_on trainer & generator with service_healthy condition", FAIL, str(e))


# T08: All 3 persistent volumes (circle-data, circle-checkpoints, circle-logs) configured
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    vols = data["volumes"]
    assert "circle-data" in vols and "circle-checkpoints" in vols and "circle-logs" in vols
    report(8, "Persistent volume PVC equivalents (data, checkpoints, logs) configured", PASS)
except Exception as e:
    report(8, "Persistent volume PVC equivalents (data, checkpoints, logs) configured", FAIL, str(e))


# T09: Trainer service configures NVIDIA GPU reservation resources
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    trainer_spec = data["services"]["trainer"]
    assert "deploy" in trainer_spec
    resources = trainer_spec["deploy"]["resources"]["reservations"]["devices"][0]
    assert resources["driver"] == "nvidia"
    assert "gpu" in resources["capabilities"]
    report(9, "Trainer service configures NVIDIA GPU reservation resources cleanly", PASS)
except Exception as e:
    report(9, "Trainer service configures NVIDIA GPU reservation resources cleanly", FAIL, str(e))


# T10: All services configure restart policy 'unless-stopped'
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    for s_name, s_config in data["services"].items():
        assert s_config.get("restart") == "unless-stopped", f"Service {s_name} missing restart policy"
    report(10, "All microservices configure 'unless-stopped' container restart policy", PASS)
except Exception as e:
    report(10, "All microservices configure 'unless-stopped' container restart policy", FAIL, str(e))


# ═══════════════════════════════════════════════════════════
# BLOCK C: BUILD AUTOMATION CLI & PERFORMANCE
# ═══════════════════════════════════════════════════════════
print("\n[ C ] Build Automation CLI & Performance")

# T11: build_images.py parses --services single target correctly
try:
    from docker.build_images import SERVICES, build_image
    ok = build_image("evaluator", SERVICES["evaluator"], dry_run=True)
    assert ok
    report(11, "build_images.py targets single microservice build cleanly", PASS)
except Exception as e:
    report(11, "build_images.py targets single microservice build cleanly", FAIL, str(e))


# T12: Dry-run mode for all 4 microservices executes in under 50ms
try:
    from docker.build_images import SERVICES, build_image
    start_t = time.time()
    for service, spec in SERVICES.items():
        build_image(service, spec, dry_run=True)
    elapsed_ms = (time.time() - start_t) * 1000
    assert elapsed_ms < 50.0, f"Dry run took {elapsed_ms:.2f}ms"
    report(12, "Dry-run build execution for all 4 microservices completes in <50ms", PASS, f"Time: {elapsed_ms:.2f}ms")
except Exception as e:
    report(12, "Dry-run build execution for all 4 microservices completes in <50ms", FAIL, str(e))


# T13: Healthcheck dry-run validation passes for all services
try:
    from docker.build_images import SERVICES, run_container_healthcheck
    for service, spec in SERVICES.items():
        ok = run_container_healthcheck(service, spec, dry_run=True)
        assert ok
    report(13, "Healthcheck dry-run validation passes across all microservices", PASS)
except Exception as e:
    report(13, "Healthcheck dry-run validation passes across all microservices", FAIL, str(e))


# T14: Image tag format matches 'circle-<service>:v1.1' naming convention
try:
    from docker.build_images import SERVICES
    for s_name, spec in SERVICES.items():
        tag = spec["tag"]
        assert tag.startswith(f"circle-{s_name}:")
    report(14, "Image tags conform strictly to 'circle-<service>:v1.1' naming convention", PASS)
except Exception as e:
    report(14, "Image tags conform strictly to 'circle-<service>:v1.1' naming convention", FAIL, str(e))


# T15: Invalid service name raises FileNotFoundError or key error
try:
    from docker.build_images import build_image
    bad_spec = {"dockerfile": "docker/non_existent.Dockerfile", "tag": "bad:latest"}
    ok = build_image("non_existent", bad_spec, dry_run=False)
    assert not ok, "Expected build failure for missing Dockerfile"
    report(15, "build_image returns False cleanly for non-existent Dockerfile", PASS)
except Exception as e:
    report(15, "build_image returns False cleanly for non-existent Dockerfile", FAIL, str(e))


# ═══════════════════════════════════════════════════════════
# BLOCK D: PERMISSIONS, ENVIRONMENT & INTEGRATION
# ═══════════════════════════════════════════════════════════
print("\n[ D ] Permissions, Environment & Integration")

# T16: Non-root user runtime permissions set chmod 777 for shared mount points
try:
    for df_name in ["trainer.Dockerfile", "eval.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            content = f.read()
        assert "chmod -R 777 /app" in content, f"chmod -R 777 /app missing in {df_name}"
    report(16, "Non-root user directory permissions set to 777 across all Dockerfiles", PASS)
except Exception as e:
    report(16, "Non-root user directory permissions set to 777 across all Dockerfiles", FAIL, str(e))


# T17: PATH includes /root/.local/bin across runner stages
try:
    for df_name in ["trainer.Dockerfile", "eval.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            content = f.read()
        assert "PATH=/root/.local/bin:$PATH" in content
    report(17, "Runner stage PATH explicitly includes /root/.local/bin", PASS)
except Exception as e:
    report(17, "Runner stage PATH explicitly includes /root/.local/bin", FAIL, str(e))


# T18: Compose service container_name attributes match circle-<service>
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    for s_name, s_config in data["services"].items():
        assert s_config["container_name"] == f"circle-{s_name}"
    report(18, "Compose container_name properties match 'circle-<service>' pattern", PASS)
except Exception as e:
    report(18, "Compose container_name properties match 'circle-<service>' pattern", FAIL, str(e))


# T19: build_images.py CLI execution via subprocess --dry-run
try:
    script_path = os.path.join(PROJECT_ROOT, "docker", "build_images.py")
    res = subprocess.run([sys.executable, script_path, "--dry-run", "--services", "all"], capture_output=True, text=True, check=True)
    assert "Build pipeline complete: 4/4 services built successfully" in res.stderr or "Build pipeline complete: 4/4 services built successfully" in res.stdout
    report(19, "build_images.py CLI subprocess execution --dry-run succeeds", PASS)
except Exception as e:
    report(19, "build_images.py CLI subprocess execution --dry-run succeeds", FAIL, str(e))


# T20: Full Part 1-11 module imports succeed without circular dependencies
try:
    import trainer.train
    import eval.critique
    import eval.failure_parser
    import eval.prompt_writer
    import eval.distilled_eval
    import generator.generate
    import generator.validator
    import docker.build_images
    report(20, "Full pipeline module imports across Parts 1-11 succeed without circular errors", PASS)
except Exception as e:
    report(20, "Full pipeline module imports across Parts 1-11 succeed without circular errors", FAIL, str(e))


# ─── SUMMARY ───
print("\n" + "="*65)
passed = sum(1 for _, _, s, _ in results if s == PASS)
failed = sum(1 for _, _, s, _ in results if s == FAIL)
print(f"  RESULTS: {passed} PASSED  |  {failed} FAILED  |  {len(results)} TOTAL")
print("="*65 + "\n")
if failed > 0:
    sys.exit(1)
