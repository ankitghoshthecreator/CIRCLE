"""
tests/test_part11.py
CIRCLE - Part 11 Unit Tests: Containerization & Build Pipeline Optimization

Tests Dockerfile instruction schemas, build_images.py CLI automation,
docker-compose configuration validity, and healthcheck semantics.
"""

import os
import sys
import json
import yaml
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING)

PASS, FAIL = "PASS", "FAIL"
results = []

def report(tid, name, status, detail=""):
    icon = "OK" if status == PASS else "XX"
    print(f"  [{icon}] Test {tid:02d}: {name} - {status}" + (f"\n         Detail: {detail}" if detail else ""))
    results.append((tid, name, status, detail))


print("\n" + "="*65)
print("  CIRCLE - Part 11 Unit Tests (Containerization & Build Pipeline)")
print("="*65 + "\n")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# T01: Verify all 4 Dockerfiles exist in docker/ directory
try:
    docker_dir = os.path.join(PROJECT_ROOT, "docker")
    expected_dockerfiles = [
        "trainer.Dockerfile",
        "eval.Dockerfile",
        "generator.Dockerfile",
        "orchestrator.Dockerfile"
    ]
    for df in expected_dockerfiles:
        path = os.path.join(docker_dir, df)
        assert os.path.exists(path), f"Missing {df}"
    report(1, "All 4 microservice Dockerfiles exist in docker/ directory", PASS)
except Exception as e:
    report(1, "All 4 microservice Dockerfiles exist in docker/ directory", FAIL, str(e))


# T02: Verify multi-stage build instructions in trainer.Dockerfile
try:
    df_path = os.path.join(PROJECT_ROOT, "docker", "trainer.Dockerfile")
    with open(df_path) as f:
        content = f.read()

    assert "FROM python:3.10-slim AS builder" in content
    assert "FROM python:3.10-slim AS runner" in content
    assert "HEALTHCHECK" in content
    assert "ENTRYPOINT" in content
    report(2, "trainer.Dockerfile implements multi-stage build with HEALTHCHECK & ENTRYPOINT", PASS)
except Exception as e:
    report(2, "trainer.Dockerfile implements multi-stage build with HEALTHCHECK & ENTRYPOINT", FAIL, str(e))


# T03: Verify lightweight eval.Dockerfile & generator.Dockerfile healthchecks
try:
    for df_name in ["eval.Dockerfile", "generator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            content = f.read()
        assert "HEALTHCHECK" in content, f"{df_name} missing HEALTHCHECK"
        assert "AS runner" in content, f"{df_name} missing multi-stage build"
    report(3, "eval.Dockerfile & generator.Dockerfile specify healthchecks and multi-stage builds", PASS)
except Exception as e:
    report(3, "eval.Dockerfile & generator.Dockerfile specify healthchecks and multi-stage builds", FAIL, str(e))


# T04: Verify docker-compose.yml schema validity and 4 microservices
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    assert os.path.exists(compose_path)
    with open(compose_path) as f:
        compose_data = yaml.safe_load(f)

    assert "services" in compose_data
    services = compose_data["services"]
    assert "trainer" in services
    assert "evaluator" in services
    assert "generator" in services
    assert "orchestrator" in services
    assert "volumes" in compose_data
    report(4, "docker-compose.yml schema valid and configures all 4 microservices", PASS)
except Exception as e:
    report(4, "docker-compose.yml schema valid and configures all 4 microservices", FAIL, str(e))


# T05: Verify build_images.py DRY-RUN mode builds all 4 services without errors
try:
    from docker.build_images import SERVICES, build_image
    for service, spec in SERVICES.items():
        ok = build_image(service, spec, dry_run=True)
        assert ok, f"Dry run failed for {service}"
    report(5, "build_images.py dry-run mode validates all 4 microservice build specs", PASS)
except Exception as e:
    report(5, "build_images.py dry-run mode validates all 4 microservice build specs", FAIL, str(e))


# T06: Verify environment variable definitions in docker-compose.yml
try:
    compose_path = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
    with open(compose_path) as f:
        compose_data = yaml.safe_load(f)

    trainer_envs = compose_data["services"]["trainer"]["environment"]
    assert any("PYTHONUNBUFFERED" in str(e) for e in trainer_envs)
    assert any("STAGE_ID" in str(e) for e in trainer_envs)
    report(6, "docker-compose.yml defines required environment variables across services", PASS)
except Exception as e:
    report(6, "docker-compose.yml defines required environment variables across services", FAIL, str(e))


# T07: Verify non-root volume path permissions and directory creation in Dockerfiles
try:
    for df_name in ["trainer.Dockerfile", "generator.Dockerfile", "orchestrator.Dockerfile"]:
        df_path = os.path.join(PROJECT_ROOT, "docker", df_name)
        with open(df_path) as f:
            content = f.read()
        assert "/app/data" in content or "/app/checkpoints" in content
        assert "chmod" in content or "mkdir" in content
    report(7, "Dockerfiles configure app data/checkpoint directory permissions", PASS)
except Exception as e:
    report(7, "Dockerfiles configure app data/checkpoint directory permissions", FAIL, str(e))


# T08: Verify orchestrator.Dockerfile syntax and entrypoint
try:
    df_path = os.path.join(PROJECT_ROOT, "docker", "orchestrator.Dockerfile")
    with open(df_path) as f:
        content = f.read()
    assert "kubernetes" in content
    assert "ENTRYPOINT" in content
    report(8, "orchestrator.Dockerfile includes Kubernetes SDK and entrypoint", PASS)
except Exception as e:
    report(8, "orchestrator.Dockerfile includes Kubernetes SDK and entrypoint", FAIL, str(e))


# ─── SUMMARY ───
print("\n" + "="*65)
passed = sum(1 for _, _, s, _ in results if s == PASS)
failed = sum(1 for _, _, s, _ in results if s == FAIL)
print(f"  RESULTS: {passed} PASSED  |  {failed} FAILED  |  {len(results)} TOTAL")
print("="*65 + "\n")
if failed > 0:
    sys.exit(1)
