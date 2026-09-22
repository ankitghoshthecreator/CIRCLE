"""
orchestrator/k8s/apply_manifests.py
CIRCLE Part 12: Kubernetes Manifest Apply Automation

Applies all CIRCLE k8s manifests in correct dependency order:
  1. namespace  →  2. secrets  →  3. configmap  →  4. pvcs
  5. rbac       →  6. deployments  →  7. services  →  8. hpa

Usage:
  python orchestrator/k8s/apply_manifests.py --dry-run
  python orchestrator/k8s/apply_manifests.py --apply
  python orchestrator/k8s/apply_manifests.py --apply --namespace circle
  python orchestrator/k8s/apply_manifests.py --delete   # teardown

Requires:
  kubectl configured with a valid kubeconfig pointing at your cluster.
"""

import os
import sys
import argparse
import subprocess
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("circle_k8s")

# ── Manifest apply order (dependency-first) ────────────────────────────────────
K8S_DIR = os.path.dirname(os.path.abspath(__file__))

MANIFEST_ORDER: List[Tuple[str, str]] = [
    ("namespace",              "namespace.yaml"),
    ("secrets",                "secrets.yaml"),
    ("configmap",              "configmap.yaml"),
    ("persistent-volumes",     "pvc.yaml"),
    ("rbac",                   "rbac.yaml"),
    ("trainer-deployment",     "trainer-deployment.yaml"),
    ("eval-deployment",        "eval-deployment.yaml"),
    ("generator-deployment",   "generator-deployment.yaml"),
    ("orchestrator-deployment","orchestrator-deployment.yaml"),
    ("trainer-service",        "trainer-service.yaml"),
    ("eval-service",           "eval-service.yaml"),
    ("generator-service",      "generator-service.yaml"),
    ("orchestrator-service",   "orchestrator-service.yaml"),
    ("generator-hpa",          "hpa.yaml"),
]


def run_kubectl(args: List[str], dry_run: bool = False) -> Tuple[bool, str]:
    """Execute a kubectl command. Returns (success, output)."""
    cmd = ["kubectl"] + args
    if dry_run:
        logger.info(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return True, "[dry-run]"
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=60
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()
    except FileNotFoundError:
        return False, "kubectl not found — ensure kubectl is installed and on PATH"


def apply_manifest(name: str, filename: str, dry_run: bool = False) -> bool:
    """Apply a single manifest file."""
    filepath = os.path.join(K8S_DIR, filename)
    if not os.path.exists(filepath):
        logger.error(f"Manifest not found: {filepath}")
        return False

    extra = ["--dry-run=client"] if dry_run else []
    ok, out = run_kubectl(["apply", "-f", filepath] + extra, dry_run=False)
    if ok:
        logger.info(f"  [OK] {name:35s}  ->  {filename}")
        if out and out != "[dry-run]":
            logger.debug(out)
    else:
        logger.error(f"  [XX] {name:35s}  FAILED: {out}")
    return ok


def delete_manifest(name: str, filename: str, dry_run: bool = False) -> bool:
    """Delete a single manifest file (ignore not-found errors)."""
    filepath = os.path.join(K8S_DIR, filename)
    if not os.path.exists(filepath):
        logger.warning(f"Manifest not found (skip): {filepath}")
        return True

    extra = ["--dry-run=client"] if dry_run else ["--ignore-not-found"]
    ok, out = run_kubectl(["delete", "-f", filepath] + extra, dry_run=False)
    if ok:
        logger.info(f"  [DEL] {name:35s}  ->  {filename}")
    else:
        logger.error(f"  [XX]  {name:35s}  FAILED: {out}")
    return ok


def verify_manifests() -> bool:
    """Check all manifest files exist before attempting apply."""
    missing = []
    for name, filename in MANIFEST_ORDER:
        filepath = os.path.join(K8S_DIR, filename)
        if not os.path.exists(filepath):
            missing.append(filename)
    if missing:
        logger.error(f"Missing manifest files: {missing}")
        return False
    logger.info(f"All {len(MANIFEST_ORDER)} manifest files verified present.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="CIRCLE Kubernetes Manifest Apply/Delete Automation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run",  action="store_true",
                       help="Validate manifests without applying (kubectl apply --dry-run=client)")
    group.add_argument("--apply",    action="store_true",
                       help="Apply all manifests to the cluster in dependency order")
    group.add_argument("--delete",   action="store_true",
                       help="Delete all CIRCLE resources from the cluster (reverse order)")
    group.add_argument("--verify",   action="store_true",
                       help="Verify all manifest files exist without calling kubectl")

    parser.add_argument("--namespace", default="circle",
                        help="Target Kubernetes namespace (default: circle)")
    args = parser.parse_args()

    if args.verify:
        ok = verify_manifests()
        sys.exit(0 if ok else 1)

    if not verify_manifests():
        sys.exit(1)

    # ── DRY RUN ───────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("  CIRCLE K8s Manifest Apply — DRY RUN")
        logger.info("=" * 60)
        success = 0
        for name, filename in MANIFEST_ORDER:
            filepath = os.path.join(K8S_DIR, filename)
            ok, out = run_kubectl(
                ["apply", "-f", filepath, "--dry-run=client"], dry_run=False
            )
            if ok:
                logger.info(f"  [OK] {name:35s}  ->  {filename}")
                success += 1
            else:
                logger.error(f"  [XX] {name:35s}  ->  {out}")
        logger.info(f"\nDry-run complete: {success}/{len(MANIFEST_ORDER)} manifests valid.")
        sys.exit(0 if success == len(MANIFEST_ORDER) else 1)

    # ── APPLY ─────────────────────────────────────────────────────────────────
    if args.apply:
        logger.info("=" * 60)
        logger.info("  CIRCLE K8s Manifest Apply — LIVE")
        logger.info("=" * 60)
        success = 0
        for name, filename in MANIFEST_ORDER:
            ok = apply_manifest(name, filename, dry_run=False)
            if ok:
                success += 1
        logger.info(f"\nApply complete: {success}/{len(MANIFEST_ORDER)} manifests applied.")
        if success < len(MANIFEST_ORDER):
            logger.error("Some manifests failed to apply. Check output above.")
            sys.exit(1)
        logger.info("All CIRCLE manifests applied successfully.")
        sys.exit(0)

    # ── DELETE ────────────────────────────────────────────────────────────────
    if args.delete:
        logger.info("=" * 60)
        logger.info("  CIRCLE K8s Manifest Delete — TEARDOWN")
        logger.info("=" * 60)
        success = 0
        # Delete in reverse order (teardown dependencies last)
        for name, filename in reversed(MANIFEST_ORDER):
            ok = delete_manifest(name, filename, dry_run=False)
            if ok:
                success += 1
        logger.info(f"\nTeardown complete: {success}/{len(MANIFEST_ORDER)} manifests deleted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
