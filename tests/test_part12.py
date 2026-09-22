"""
tests/test_part12.py
CIRCLE Part 12: Kubernetes Manifests Unit Test Suite

Validates all k8s manifest YAML schemas, required fields, resource specs,
label selectors, PVC storage sizes, Service port mappings, RBAC rules,
HPA bounds, and apply_manifests.py API surface.
"""

import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

PASS, FAIL = "PASS", "FAIL"
results = []

def report(tid, name, status, detail=""):
    icon = "OK" if status == PASS else "XX"
    print(f"  [{icon}] Test {tid:02d}: {name} - {status}" + (f"\n         Detail: {detail}" if detail else ""))
    results.append((tid, name, status, detail))

def load_yaml_multi(filepath):
    """Load a multi-document YAML file as a list of dicts."""
    with open(filepath) as f:
        return [doc for doc in yaml.safe_load_all(f) if doc is not None]

def load_yaml(filepath):
    with open(filepath) as f:
        return yaml.safe_load(f)


print("\n" + "="*65)
print("  CIRCLE Part 12 — Kubernetes Manifests Test Suite")
print("="*65 + "\n")

K8S_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "orchestrator", "k8s")

print("[ A ] Manifest File Existence & Structure")

# T01: All 14 required manifest files exist
try:
    required = [
        "namespace.yaml", "secrets.yaml", "configmap.yaml", "pvc.yaml",
        "rbac.yaml",
        "trainer-deployment.yaml", "eval-deployment.yaml",
        "generator-deployment.yaml", "orchestrator-deployment.yaml",
        "trainer-service.yaml", "eval-service.yaml",
        "generator-service.yaml", "orchestrator-service.yaml",
        "hpa.yaml",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(K8S_DIR, f))]
    assert not missing, f"Missing manifest files: {missing}"
    report(1, f"All {len(required)} required manifest files exist in orchestrator/k8s/", PASS)
except Exception as e:
    report(1, "All 14 required manifest files exist in orchestrator/k8s/", FAIL, str(e))


# T02: apply_manifests.py exists and MANIFEST_ORDER contains exactly 14 entries
try:
    apply_path = os.path.join(K8S_DIR, "apply_manifests.py")
    assert os.path.exists(apply_path), "apply_manifests.py not found"
    sys.path.insert(0, os.path.dirname(K8S_DIR))
    from orchestrator.k8s.apply_manifests import MANIFEST_ORDER
    assert len(MANIFEST_ORDER) == 14, f"Expected 14 manifests in MANIFEST_ORDER, got {len(MANIFEST_ORDER)}"
    report(2, "apply_manifests.py exists and MANIFEST_ORDER has exactly 14 entries", PASS)
except Exception as e:
    report(2, "apply_manifests.py exists and MANIFEST_ORDER has exactly 14 entries", FAIL, str(e))


# T03: namespace.yaml defines kind: Namespace with name 'circle'
try:
    doc = load_yaml(os.path.join(K8S_DIR, "namespace.yaml"))
    assert doc["kind"] == "Namespace"
    assert doc["metadata"]["name"] == "circle"
    report(3, "namespace.yaml defines kind: Namespace with name 'circle'", PASS)
except Exception as e:
    report(3, "namespace.yaml defines kind: Namespace with name 'circle'", FAIL, str(e))


# T04: configmap.yaml contains all required config keys
try:
    doc = load_yaml(os.path.join(K8S_DIR, "configmap.yaml"))
    assert doc["kind"] == "ConfigMap"
    assert doc["metadata"]["name"] == "circle-config"
    data = doc["data"]
    for key in ["DATA_DIR", "CHECKPOINT_DIR", "STAGE_ID", "GENERATOR_BACKEND",
                "OLLAMA_ENDPOINT_URL", "LORA_RANK", "LORA_ALPHA", "GROQ_MODEL"]:
        assert key in data, f"ConfigMap missing key: {key}"
    report(4, "configmap.yaml contains all required runtime configuration keys", PASS)
except Exception as e:
    report(4, "configmap.yaml contains all required runtime configuration keys", FAIL, str(e))


print("\n[ B ] PVC & Secrets Schema Validation")


# T05: pvc.yaml defines exactly 3 PVCs with correct names
try:
    docs = load_yaml_multi(os.path.join(K8S_DIR, "pvc.yaml"))
    pvc_names = [d["metadata"]["name"] for d in docs if d.get("kind") == "PersistentVolumeClaim"]
    assert set(pvc_names) == {"circle-data-pvc", "circle-checkpoints-pvc", "circle-logs-pvc"}, \
        f"PVC names mismatch: {pvc_names}"
    report(5, "pvc.yaml defines exactly 3 PVCs: data, checkpoints, logs", PASS)
except Exception as e:
    report(5, "pvc.yaml defines exactly 3 PVCs: data, checkpoints, logs", FAIL, str(e))


# T06: circle-checkpoints-pvc requests >= 20Gi (largest, holds adapter weights)
try:
    docs = load_yaml_multi(os.path.join(K8S_DIR, "pvc.yaml"))
    ckpt = next(d for d in docs if d.get("metadata", {}).get("name") == "circle-checkpoints-pvc")
    storage = ckpt["spec"]["resources"]["requests"]["storage"]
    size_gi = int(storage.replace("Gi", ""))
    assert size_gi >= 20, f"circle-checkpoints-pvc too small: {storage}"
    report(6, "circle-checkpoints-pvc requests >= 20Gi for QLoRA adapter weight storage", PASS)
except Exception as e:
    report(6, "circle-checkpoints-pvc requests >= 20Gi for QLoRA adapter weight storage", FAIL, str(e))


# T07: All 3 PVCs use ReadWriteMany access mode (multi-pod concurrent access)
try:
    docs = load_yaml_multi(os.path.join(K8S_DIR, "pvc.yaml"))
    for doc in docs:
        if doc.get("kind") != "PersistentVolumeClaim":
            continue
        modes = doc["spec"]["accessModes"]
        assert "ReadWriteMany" in modes, \
            f"{doc['metadata']['name']} missing ReadWriteMany: {modes}"
    report(7, "All 3 PVCs use ReadWriteMany access mode for concurrent pod access", PASS)
except Exception as e:
    report(7, "All 3 PVCs use ReadWriteMany access mode for concurrent pod access", FAIL, str(e))


# T08: secrets.yaml defines kind: Secret with name 'circle-secrets' and GROQ_API_KEY key
try:
    doc = load_yaml(os.path.join(K8S_DIR, "secrets.yaml"))
    assert doc["kind"] == "Secret"
    assert doc["metadata"]["name"] == "circle-secrets"
    assert "GROQ_API_KEY" in doc["data"], "GROQ_API_KEY missing from circle-secrets"
    assert doc["type"] == "Opaque"
    report(8, "secrets.yaml defines Opaque Secret 'circle-secrets' with GROQ_API_KEY", PASS)
except Exception as e:
    report(8, "secrets.yaml defines Opaque Secret 'circle-secrets' with GROQ_API_KEY", FAIL, str(e))


print("\n[ C ] Deployment Resource Specs & Volume Mounts")


# T09: trainer-deployment.yaml requests nvidia.com/gpu resource
try:
    doc = load_yaml(os.path.join(K8S_DIR, "trainer-deployment.yaml"))
    containers = doc["spec"]["template"]["spec"]["containers"]
    trainer = next(c for c in containers if c["name"] == "trainer")
    gpu_limit = trainer["resources"]["limits"].get("nvidia.com/gpu", "0")
    assert str(gpu_limit) != "0", "trainer-deployment missing nvidia.com/gpu limit"
    report(9, "trainer-deployment.yaml requests nvidia.com/gpu resource limit", PASS)
except Exception as e:
    report(9, "trainer-deployment.yaml requests nvidia.com/gpu resource limit", FAIL, str(e))


# T10: trainer-deployment.yaml mounts all 3 PVCs
try:
    doc = load_yaml(os.path.join(K8S_DIR, "trainer-deployment.yaml"))
    containers = doc["spec"]["template"]["spec"]["containers"]
    trainer = next(c for c in containers if c["name"] == "trainer")
    mounts = {m["mountPath"] for m in trainer.get("volumeMounts", [])}
    assert "/app/data" in mounts, "trainer missing /app/data volumeMount"
    assert "/app/checkpoints" in mounts, "trainer missing /app/checkpoints volumeMount"
    assert "/app/logs" in mounts, "trainer missing /app/logs volumeMount"
    report(10, "trainer-deployment.yaml mounts all 3 PVCs: data, checkpoints, logs", PASS)
except Exception as e:
    report(10, "trainer-deployment.yaml mounts all 3 PVCs: data, checkpoints, logs", FAIL, str(e))


# T11: trainer-deployment.yaml has both readinessProbe and livenessProbe configured
try:
    doc = load_yaml(os.path.join(K8S_DIR, "trainer-deployment.yaml"))
    containers = doc["spec"]["template"]["spec"]["containers"]
    trainer = next(c for c in containers if c["name"] == "trainer")
    assert "readinessProbe" in trainer, "trainer missing readinessProbe"
    assert "livenessProbe" in trainer, "trainer missing livenessProbe"
    report(11, "trainer-deployment.yaml has both readinessProbe and livenessProbe", PASS)
except Exception as e:
    report(11, "trainer-deployment.yaml has both readinessProbe and livenessProbe", FAIL, str(e))


# T12: orchestrator-deployment.yaml uses serviceAccountName circle-orchestrator-sa
try:
    doc = load_yaml(os.path.join(K8S_DIR, "orchestrator-deployment.yaml"))
    pod_spec = doc["spec"]["template"]["spec"]
    sa = pod_spec.get("serviceAccountName", "")
    assert sa == "circle-orchestrator-sa", \
        f"orchestrator serviceAccountName is '{sa}' (expected 'circle-orchestrator-sa')"
    report(12, "orchestrator-deployment.yaml uses serviceAccountName 'circle-orchestrator-sa'", PASS)
except Exception as e:
    report(12, "orchestrator-deployment.yaml uses serviceAccountName 'circle-orchestrator-sa'", FAIL, str(e))


print("\n[ D ] RBAC, Services & HPA Validation")


# T13: rbac.yaml defines ServiceAccount, Role, and RoleBinding
try:
    docs = load_yaml_multi(os.path.join(K8S_DIR, "rbac.yaml"))
    kinds = {d["kind"] for d in docs}
    assert "ServiceAccount" in kinds, "rbac.yaml missing ServiceAccount"
    assert "Role" in kinds, "rbac.yaml missing Role"
    assert "RoleBinding" in kinds, "rbac.yaml missing RoleBinding"
    report(13, "rbac.yaml defines ServiceAccount, Role, and RoleBinding", PASS)
except Exception as e:
    report(13, "rbac.yaml defines ServiceAccount, Role, and RoleBinding", FAIL, str(e))


# T14: rbac.yaml Role includes batch/jobs verbs (required for orchestrator to manage training Jobs)
try:
    docs = load_yaml_multi(os.path.join(K8S_DIR, "rbac.yaml"))
    role = next(d for d in docs if d.get("kind") == "Role")
    rules = role["rules"]
    batch_rule = next((r for r in rules if "batch" in r.get("apiGroups", [])), None)
    assert batch_rule is not None, "Role missing batch apiGroup rule"
    assert "jobs" in batch_rule["resources"], "Role missing 'jobs' in batch resources"
    assert "create" in batch_rule["verbs"], "Role missing 'create' verb for batch/jobs"
    report(14, "RBAC Role includes batch/jobs rules with create/delete verbs", PASS)
except Exception as e:
    report(14, "RBAC Role includes batch/jobs rules with create/delete verbs", FAIL, str(e))


# T15: All 4 Services use type: ClusterIP (not NodePort or LoadBalancer)
try:
    service_files = ["trainer-service.yaml", "eval-service.yaml",
                     "generator-service.yaml", "orchestrator-service.yaml"]
    for sf in service_files:
        doc = load_yaml(os.path.join(K8S_DIR, sf))
        assert doc["kind"] == "Service", f"{sf} is not a Service"
        svc_type = doc["spec"].get("type", "ClusterIP")
        assert svc_type == "ClusterIP", f"{sf} has type '{svc_type}' (expected ClusterIP)"
    report(15, "All 4 Services use type: ClusterIP (internal access only)", PASS)
except Exception as e:
    report(15, "All 4 Services use type: ClusterIP (internal access only)", FAIL, str(e))


# T16: Service ports match expected: trainer=8080, eval=8081, generator=8082, orchestrator=8083
try:
    expected_ports = {
        "trainer-service.yaml": 8080,
        "eval-service.yaml": 8081,
        "generator-service.yaml": 8082,
        "orchestrator-service.yaml": 8083,
    }
    for sf, expected_port in expected_ports.items():
        doc = load_yaml(os.path.join(K8S_DIR, sf))
        ports = doc["spec"]["ports"]
        actual = ports[0]["port"]
        assert actual == expected_port, f"{sf}: port is {actual}, expected {expected_port}"
    report(16, "Service port mapping correct: trainer=8080, eval=8081, gen=8082, orch=8083", PASS)
except Exception as e:
    report(16, "Service port mapping correct: trainer=8080, eval=8081, gen=8082, orch=8083", FAIL, str(e))


# T17: hpa.yaml targets circle-generator Deployment with minReplicas=1 maxReplicas=4
try:
    doc = load_yaml(os.path.join(K8S_DIR, "hpa.yaml"))
    assert doc["kind"] == "HorizontalPodAutoscaler"
    ref = doc["spec"]["scaleTargetRef"]
    assert ref["name"] == "circle-generator", f"HPA targets '{ref['name']}' (expected 'circle-generator')"
    assert doc["spec"]["minReplicas"] == 1
    assert doc["spec"]["maxReplicas"] == 4
    report(17, "hpa.yaml targets circle-generator with minReplicas=1 maxReplicas=4", PASS)
except Exception as e:
    report(17, "hpa.yaml targets circle-generator with minReplicas=1 maxReplicas=4", FAIL, str(e))


# T18: HPA CPU target utilization is 60% (burst threshold)
try:
    doc = load_yaml(os.path.join(K8S_DIR, "hpa.yaml"))
    metrics = doc["spec"]["metrics"]
    cpu_metric = next(m for m in metrics if m.get("resource", {}).get("name") == "cpu")
    target_util = cpu_metric["resource"]["target"]["averageUtilization"]
    assert target_util == 60, f"HPA CPU target is {target_util}% (expected 60%)"
    report(18, "HPA CPU scale-up target is 60% average utilization", PASS)
except Exception as e:
    report(18, "HPA CPU scale-up target is 60% average utilization", FAIL, str(e))


# T19: apply_manifests.py verify_manifests() returns True when all files present
try:
    from orchestrator.k8s.apply_manifests import verify_manifests
    ok = verify_manifests()
    assert ok is True, "verify_manifests() returned False despite all files being present"
    report(19, "apply_manifests.verify_manifests() returns True when all 14 files present", PASS)
except Exception as e:
    report(19, "apply_manifests.verify_manifests() returns True when all 14 files present", FAIL, str(e))


# T20: All Deployments have namespace: circle set in metadata
try:
    deployment_files = [
        "trainer-deployment.yaml", "eval-deployment.yaml",
        "generator-deployment.yaml", "orchestrator-deployment.yaml",
    ]
    for df in deployment_files:
        doc = load_yaml(os.path.join(K8S_DIR, df))
        ns = doc["metadata"].get("namespace", "")
        assert ns == "circle", f"{df}: namespace is '{ns}' (expected 'circle')"
    report(20, "All 4 Deployments have namespace: circle set in metadata", PASS)
except Exception as e:
    report(20, "All 4 Deployments have namespace: circle set in metadata", FAIL, str(e))


# ─── SUMMARY ───
print("\n" + "="*65)
passed = sum(1 for _, _, s, _ in results if s == PASS)
failed = sum(1 for _, _, s, _ in results if s == FAIL)
print(f"  RESULTS: {passed} PASSED  |  {failed} FAILED  |  {len(results)} TOTAL")
print("="*65 + "\n")
if failed > 0:
    sys.exit(1)
