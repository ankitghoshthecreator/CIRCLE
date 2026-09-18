"""
docker/build_images.py
Part 11: Containerization & Build Pipeline Optimization

Automated CLI tool for building, tagging, inspecting, and validating
CIRCLE Docker microservice images locally.
"""

import os
import sys
import argparse
import subprocess
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("docker_builder")

# Microservice specs
SERVICES = {
    "trainer": {
        "dockerfile": "docker/trainer.Dockerfile",
        "tag": "circle-trainer:v1.1",
        "healthcheck_cmd": "python -c 'import torch, peft; print(\"Healthy\")'",
    },
    "evaluator": {
        "dockerfile": "docker/eval.Dockerfile",
        "tag": "circle-evaluator:v1.1",
        "healthcheck_cmd": "python -c 'import eval.failure_parser; print(\"Healthy\")'",
    },
    "generator": {
        "dockerfile": "docker/generator.Dockerfile",
        "tag": "circle-generator:v1.1",
        "healthcheck_cmd": "python -c 'import generator.validator; print(\"Healthy\")'",
    },
    "orchestrator": {
        "dockerfile": "docker/orchestrator.Dockerfile",
        "tag": "circle-orchestrator:v1.1",
        "healthcheck_cmd": "python -c 'import kubernetes; print(\"Healthy\")'",
    },
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_image(service_name: str, spec: Dict[str, str], dry_run: bool = False) -> bool:
    """Builds a single microservice Docker image using docker build."""
    dockerfile_path = os.path.join(PROJECT_ROOT, spec["dockerfile"])
    if not os.path.exists(dockerfile_path):
        logger.error(f"Dockerfile for '{service_name}' not found at {dockerfile_path}")
        return False

    cmd = [
        "docker", "build",
        "-f", spec["dockerfile"],
        "-t", spec["tag"],
        "-t", f"circle-{service_name}:latest",
        "."
    ]

    logger.info(f"Building [{service_name}] image: {' '.join(cmd)}")
    if dry_run:
        logger.info(f"[DRY-RUN] Would build '{service_name}' via {spec['dockerfile']}")
        return True

    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        logger.info(f"Successfully built [{service_name}] -> {spec['tag']}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build [{service_name}]: {e.stderr}")
        return False


def inspect_image_size(tag: str) -> Optional[str]:
    """Inspects the size of a built image tag."""
    cmd = ["docker", "inspect", "-f", "{{ .Size }}", tag]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        size_bytes = int(res.stdout.strip())
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    except Exception:
        return None


def run_container_healthcheck(service_name: str, spec: Dict[str, str], dry_run: bool = False) -> bool:
    """Runs a one-shot container execution to verify entrypoint health."""
    cmd = [
        "docker", "run", "--rm",
        spec["tag"],
        "sh", "-c", spec["healthcheck_cmd"]
    ]
    logger.info(f"Healthchecking [{service_name}] image: {spec['tag']}")
    if dry_run:
        logger.info(f"[DRY-RUN] Would run healthcheck for '{service_name}'")
        return True

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        logger.info(f"Healthcheck PASSED for [{service_name}]")
        return True
    except Exception as e:
        logger.error(f"Healthcheck FAILED for [{service_name}]: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="CIRCLE Docker Microservice Build Automation Tool")
    parser.add_argument("--services", nargs="+", choices=list(SERVICES.keys()) + ["all"], default=["all"])
    parser.add_argument("--dry-run", action="store_true", help="Simulate build without invoking docker binary")
    parser.add_argument("--healthcheck", action="store_true", help="Run healthcheck after build")
    args = parser.parse_args()

    targets = list(SERVICES.keys()) if "all" in args.services else args.services
    logger.info(f"Starting build pipeline for services: {targets}")

    success_count = 0
    for service in targets:
        spec = SERVICES[service]
        ok = build_image(service, spec, dry_run=args.dry_run)
        if ok:
            success_count += 1
            if not args.dry_run:
                size_str = inspect_image_size(spec["tag"])
                if size_str:
                    logger.info(f"[{service}] Image Size: {size_str}")
            if args.healthcheck:
                run_container_healthcheck(service, spec, dry_run=args.dry_run)

    logger.info(f"Build pipeline complete: {success_count}/{len(targets)} services built successfully.")
    if success_count < len(targets):
        sys.exit(1)


if __name__ == "__main__":
    main()
