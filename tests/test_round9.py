"""
tests/test_round9.py
CIRCLE - 20 Hard-Mode Stress & Edge-Case Tests (Round 9)

Focus:
- WORKDIR correctness across runner stages (no accidental /build leaks)
- Compose cross-service volume mount completeness (trainer/orchestrator need checkpoints)
- Compose env-var default injection (GROQ fallback, OLLAMA default, STAGE_ID, KUBERNETES_SERVICE_HOST)
- Dependency chain integrity (generator->evaluator ordering)
- Dockerfile base image pinning to python:3.10-slim across all stages
- build_images API surface (return types, callable functions, main() existence)
- Healthcheck start-period presence enforcement
- No duplicate container_name values across all compose services
- trainer/curriculum/ COPY presence in generator.Dockerfile (cross-module coupling)
- Orchestrator all-3-volumes completeness
- Compose build context set to '..' (project root, not './docker')
"""

import os
import sys
import json
import yaml
import time
import inspect
import importlib
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
print("  CIRCLE - 20 Hard-Mode Edge-Case Tests (Round 9)")
print("="*65 + "\n")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════
# BLOCK A: DOCKERFILE WORKDIR & BASE IMAGE PINNING
# ═══════════════════════════════════════════════════════════
print("[ A ] Dockerfile WORKDIR & Base Image Pinning")


# T01: Runner stage WORKDIR is /app (not /build) across all Dockerfiles
try:
    for df_name in ["trainer.Dockerfile", "eval.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            content = f.read()
        # runner stage begins at 'AS runner'; WORKDIR /app must appear after that
        runner_section = content.split("AS runner", 1)
        assert len(runner_section) == 2, f"{df_name}: 'AS runner' stage not found"
        assert "WORKDIR /app" in runner_section[1], \
            f"{df_name}: runner stage WORKDIR is not '/app'"
    report(1, "All runner stages set WORKDIR to /app (not /build or /)", PASS)
except Exception as e:
    report(1, "All runner stages set WORKDIR to /app (not /build or /)", FAIL, str(e))


# T02: Builder stage WORKDIR is /build across all Dockerfiles
try:
    for df_name in ["trainer.Dockerfile", "eval.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            content = f.read()
        builder_section = content.split("AS builder", 1)
        assert len(builder_section) == 2, f"{df_name}: 'AS builder' stage not found"
        runner_start = content.find("AS runner")
        builder_body = content[content.find("AS builder"):runner_start]
        assert "WORKDIR /build" in builder_body, \
            f"{df_name}: builder stage WORKDIR is not '/build'"
    report(2, "All builder stages set WORKDIR to /build (isolated from runtime)", PASS)
except Exception as e:
    report(2, "All builder stages set WORKDIR to /build (isolated from runtime)", FAIL, str(e))


# T03: All Dockerfiles use python:3.10-slim base image (no version drift to 3.9 or 3.11)
try:
    for df_name in ["trainer.Dockerfile", "eval.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            content = f.read()
        from_lines = [l.strip() for l in content.splitlines() if l.strip().upper().startswith("FROM ")]
        for fl in from_lines:
            assert "python:3.10-slim" in fl, \
                f"{df_name}: FROM line '{fl}' does not use python:3.10-slim"
    report(3, "All Dockerfiles pin base image to python:3.10-slim (no version drift)", PASS)
except Exception as e:
    report(3, "All Dockerfiles pin base image to python:3.10-slim (no version drift)", FAIL, str(e))


# T04: generator.Dockerfile copies trainer/curriculum/ for dataset schema access
try:
    df_path = os.path.join(PROJECT_ROOT, "docker", "generator.Dockerfile")
    with open(df_path) as f:
        content = f.read()
    assert "COPY trainer/curriculum/" in content, \
        "generator.Dockerfile missing 'COPY trainer/curriculum/' (DatasetSpec schema dependency)"
    report(4, "generator.Dockerfile bundles trainer/curriculum/ for DatasetSpec schema access", PASS)
except Exception as e:
    report(4, "generator.Dockerfile bundles trainer/curriculum/ for DatasetSpec schema access", FAIL, str(e))


# T05: All Dockerfiles' HEALTHCHECK includes --start-period (cold-start grace window)
try:
    for df_name in ["trainer.Dockerfile", "eval.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            content = f.read()
        assert "--start-period=" in content, \
            f"{df_name}: HEALTHCHECK missing --start-period (cold-start grace window)"
    report(5, "All Dockerfiles HEALTHCHECK includes --start-period cold-start grace period", PASS)
except Exception as e:
    report(5, "All Dockerfiles HEALTHCHECK includes --start-period cold-start grace period", FAIL, str(e))


# ═══════════════════════════════════════════════════════════
# BLOCK B: DOCKER COMPOSE DEEP ENV & VOLUME INVARIANTS
# ═══════════════════════════════════════════════════════════
print("\n[ B ] Docker Compose Deep Env & Volume Invariants")


# T06: trainer compose service injects STAGE_ID env var for curriculum loop stage tracking
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    trainer_env = data["services"]["trainer"]["environment"]
    stage_vars = [e for e in trainer_env if "STAGE_ID" in str(e)]
    assert len(stage_vars) >= 1, "STAGE_ID not found in trainer environment"
    report(6, "Compose trainer service injects STAGE_ID for curriculum loop stage tracking", PASS)
except Exception as e:
    report(6, "Compose trainer service injects STAGE_ID for curriculum loop stage tracking", FAIL, str(e))


# T07: evaluator compose service GROQ_API_KEY has shell-default fallback (:-mock_key pattern)
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    eval_env = data["services"]["evaluator"]["environment"]
    groq_entries = [e for e in eval_env if "GROQ_API_KEY" in str(e)]
    assert len(groq_entries) >= 1, "GROQ_API_KEY missing from evaluator environment"
    assert ":-" in str(groq_entries[0]), \
        f"GROQ_API_KEY has no shell fallback default (:-mock_key): {groq_entries[0]}"
    report(7, "Evaluator GROQ_API_KEY has shell-default fallback (prevents hard crash on missing key)", PASS)
except Exception as e:
    report(7, "Evaluator GROQ_API_KEY has shell-default fallback (prevents hard crash on missing key)", FAIL, str(e))


# T08: generator compose service OLLAMA_ENDPOINT_URL has default fallback value
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    gen_env = data["services"]["generator"]["environment"]
    ollama_entries = [e for e in gen_env if "OLLAMA_ENDPOINT_URL" in str(e)]
    assert len(ollama_entries) >= 1, "OLLAMA_ENDPOINT_URL missing from generator environment"
    assert ":-" in str(ollama_entries[0]), \
        f"OLLAMA_ENDPOINT_URL has no shell fallback: {ollama_entries[0]}"
    report(8, "Generator OLLAMA_ENDPOINT_URL has shell-default fallback (offline mock safety)", PASS)
except Exception as e:
    report(8, "Generator OLLAMA_ENDPOINT_URL has shell-default fallback (offline mock safety)", FAIL, str(e))


# T09: orchestrator compose service injects KUBERNETES_SERVICE_HOST env var for cluster detection
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    orch_env = data["services"]["orchestrator"]["environment"]
    k8s_entries = [e for e in orch_env if "KUBERNETES_SERVICE_HOST" in str(e)]
    assert len(k8s_entries) >= 1, "KUBERNETES_SERVICE_HOST missing from orchestrator environment"
    report(9, "Orchestrator injects KUBERNETES_SERVICE_HOST for in-cluster vs local detection", PASS)
except Exception as e:
    report(9, "Orchestrator injects KUBERNETES_SERVICE_HOST for in-cluster vs local detection", FAIL, str(e))


# T10: trainer compose service mounts BOTH circle-data AND circle-checkpoints volumes
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    trainer_vols = str(data["services"]["trainer"].get("volumes", []))
    assert "circle-data" in trainer_vols, "trainer missing circle-data volume"
    assert "circle-checkpoints" in trainer_vols, "trainer missing circle-checkpoints volume"
    report(10, "Trainer service mounts both circle-data and circle-checkpoints volumes", PASS)
except Exception as e:
    report(10, "Trainer service mounts both circle-data and circle-checkpoints volumes", FAIL, str(e))


# T11: orchestrator compose service mounts all 3 volumes (data, checkpoints, logs)
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    orch_vols = str(data["services"]["orchestrator"].get("volumes", []))
    assert "circle-data" in orch_vols, "orchestrator missing circle-data volume"
    assert "circle-checkpoints" in orch_vols, "orchestrator missing circle-checkpoints volume"
    assert "circle-logs" in orch_vols, "orchestrator missing circle-logs volume"
    report(11, "Orchestrator service mounts all 3 PVC volumes: data, checkpoints, logs", PASS)
except Exception as e:
    report(11, "Orchestrator service mounts all 3 PVC volumes: data, checkpoints, logs", FAIL, str(e))


# T12: No duplicate container_name values exist across all compose services
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    names = [s.get("container_name") for s in data["services"].values()]
    assert len(names) == len(set(names)), \
        f"Duplicate container_name detected: {[n for n in names if names.count(n) > 1]}"
    report(12, "No duplicate container_name values across all compose services", PASS)
except Exception as e:
    report(12, "No duplicate container_name values across all compose services", FAIL, str(e))


# T13: generator compose service depends_on evaluator (proper data validation ordering)
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    gen_deps = data["services"]["generator"].get("depends_on", [])
    # depends_on can be a list or a dict
    dep_names = list(gen_deps.keys()) if isinstance(gen_deps, dict) else gen_deps
    assert "evaluator" in dep_names, \
        f"generator.depends_on does not include 'evaluator': {dep_names}"
    report(13, "Generator service depends_on evaluator (ensures validator has critique schema)", PASS)
except Exception as e:
    report(13, "Generator service depends_on evaluator (ensures validator has critique schema)", FAIL, str(e))


# T14: All compose service build.context values are '..' (project root, not docker/ dir)
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    for s_name, s_config in data["services"].items():
        context = s_config.get("build", {}).get("context", "")
        assert context == "..", \
            f"Service '{s_name}' build.context is '{context}' (expected '..')"
    report(14, "All compose services build.context correctly set to '..' (project root)", PASS)
except Exception as e:
    report(14, "All compose services build.context correctly set to '..' (project root)", FAIL, str(e))


# ═══════════════════════════════════════════════════════════
# BLOCK C: BUILD AUTOMATION API SURFACE & RETURN TYPES
# ═══════════════════════════════════════════════════════════
print("\n[ C ] Build Automation API Surface & Return Types")


# T15: build_image() returns a bool (True/False), not None or int
try:
    from docker.build_images import SERVICES, build_image
    result = build_image("trainer", SERVICES["trainer"], dry_run=True)
    assert isinstance(result, bool), f"build_image returned {type(result).__name__}, expected bool"
    assert result is True
    report(15, "build_image() returns bool True on successful dry-run (not None or truthy int)", PASS)
except Exception as e:
    report(15, "build_image() returns bool True on successful dry-run (not None or truthy int)", FAIL, str(e))


# T16: build_image() for missing Dockerfile returns bool False (not raises, not None)
try:
    from docker.build_images import build_image
    bad = {"dockerfile": "docker/does_not_exist.Dockerfile", "tag": "test:fail"}
    result = build_image("ghost", bad, dry_run=False)
    assert isinstance(result, bool), f"build_image returned {type(result).__name__} on error, expected bool"
    assert result is False, f"Expected False for missing Dockerfile, got {result}"
    report(16, "build_image() returns bool False for missing Dockerfile (not raises, not None)", PASS)
except Exception as e:
    report(16, "build_image() returns bool False for missing Dockerfile (not raises, not None)", FAIL, str(e))


# T17: run_container_healthcheck() returns a bool on dry_run=True
try:
    from docker.build_images import SERVICES, run_container_healthcheck
    result = run_container_healthcheck("generator", SERVICES["generator"], dry_run=True)
    assert isinstance(result, bool), f"run_container_healthcheck returned {type(result).__name__}"
    assert result is True
    report(17, "run_container_healthcheck() returns bool True on dry-run (type-safe API)", PASS)
except Exception as e:
    report(17, "run_container_healthcheck() returns bool True on dry-run (type-safe API)", FAIL, str(e))


# T18: build_images.py exposes a callable main() function
try:
    from docker import build_images as bm
    assert hasattr(bm, "main"), "build_images module missing 'main' function"
    assert callable(bm.main), "'main' in build_images is not callable"
    report(18, "build_images.py exposes a callable main() entry point function", PASS)
except Exception as e:
    report(18, "build_images.py exposes a callable main() entry point function", FAIL, str(e))


# T19: build_images.py exposes inspect_image_size() function with correct signature (tag: str)
try:
    from docker.build_images import inspect_image_size
    sig = inspect.signature(inspect_image_size)
    params = list(sig.parameters.keys())
    assert "tag" in params, f"inspect_image_size missing 'tag' param: {params}"
    assert len(params) == 1, f"inspect_image_size should have 1 param (tag), got {params}"
    report(19, "inspect_image_size() has correct single-param signature (tag: str)", PASS)
except Exception as e:
    report(19, "inspect_image_size() has correct single-param signature (tag: str)", FAIL, str(e))


# T20: All SERVICES spec 'dockerfile' paths use forward slashes (cross-platform safety)
try:
    from docker.build_images import SERVICES
    for s_name, spec in SERVICES.items():
        df_path = spec["dockerfile"]
        assert "\\" not in df_path, \
            f"Service '{s_name}' dockerfile path uses backslash: '{df_path}' (must use forward slashes)"
    report(20, "All SERVICES dockerfile paths use forward slashes (cross-platform CI/CD safety)", PASS)
except Exception as e:
    report(20, "All SERVICES dockerfile paths use forward slashes (cross-platform CI/CD safety)", FAIL, str(e))


# ─── SUMMARY ───
print("\n" + "="*65)
passed = sum(1 for _, _, s, _ in results if s == PASS)
failed = sum(1 for _, _, s, _ in results if s == FAIL)
print(f"  RESULTS: {passed} PASSED  |  {failed} FAILED  |  {len(results)} TOTAL")
print("="*65 + "\n")
if failed > 0:
    sys.exit(1)
