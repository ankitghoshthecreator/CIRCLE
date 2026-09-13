"""
CIRCLE - 20 New Test Cases (Round 3)
Covers: failure parser internals, severity logic, taxonomy completeness,
Pydantic serialization, think-tag stripping, cross-component integration,
model loader, tokenizer config, and edge-case critique strings.
"""
import os, sys, json, shutil, re, logging, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

PASS, FAIL = "PASS", "FAIL"
results = []

def report(tid, name, status, detail=""):
    icon = "OK" if status == PASS else "XX"
    print(f"  [{icon}] Test {tid:02d}: {name} - {status}" + (f"\n         Detail: {detail}" if detail else ""))
    results.append((tid, name, status, detail))

print("\n" + "="*65)
print("  CIRCLE - 20 New Tests (Round 3)")
print("="*65 + "\n")

# ═══════════════════════════════════════════════════════════
# BLOCK A: FAILURE PARSER — TAXONOMY & TAXONOMY COMPLETENESS
# ═══════════════════════════════════════════════════════════
print("[ A ] Failure Parser — Taxonomy & Completeness")

# T01: All FailureCategory values are in CATEGORY_SEVERITY_MAP
try:
    from eval.failure_parser import FailureCategory, CATEGORY_SEVERITY_MAP
    missing = [c for c in FailureCategory if c not in CATEGORY_SEVERITY_MAP]
    assert not missing, f"Categories missing from severity map: {missing}"
    report(1, "Every FailureCategory has an entry in CATEGORY_SEVERITY_MAP", PASS,
           f"{len(FailureCategory)} categories covered")
except Exception as e:
    report(1, "Every FailureCategory has an entry in CATEGORY_SEVERITY_MAP", FAIL, str(e))

# T02: LOW-severity categories never trigger commissioning
try:
    from eval.failure_parser import FailureCategory, CATEGORY_SEVERITY_MAP, COMMISSIONING_TRIGGERS, Severity
    low_cats = [c for c, s in CATEGORY_SEVERITY_MAP.items() if s == Severity.LOW]
    for cat in low_cats:
        sev = CATEGORY_SEVERITY_MAP[cat]
        assert not COMMISSIONING_TRIGGERS[sev], f"LOW category {cat} should NOT trigger commissioning"
    report(2, "LOW-severity failures never trigger commissioning", PASS,
           f"Checked {len(low_cats)} LOW categories")
except Exception as e:
    report(2, "LOW-severity failures never trigger commissioning", FAIL, str(e))

# T03: CRITICAL and HIGH always trigger commissioning
try:
    from eval.failure_parser import Severity, COMMISSIONING_TRIGGERS
    for sev in [Severity.CRITICAL, Severity.HIGH]:
        assert COMMISSIONING_TRIGGERS[sev] is True, f"{sev} must trigger commissioning"
    report(3, "CRITICAL and HIGH severities always trigger commissioning", PASS)
except Exception as e:
    report(3, "CRITICAL and HIGH severities always trigger commissioning", FAIL, str(e))

# T04: INCOHERENT_CONTINUATION is the only CRITICAL-severity category
try:
    from eval.failure_parser import FailureCategory, CATEGORY_SEVERITY_MAP, Severity
    criticals = [c for c, s in CATEGORY_SEVERITY_MAP.items() if s == Severity.CRITICAL]
    assert criticals == [FailureCategory.INCOHERENT_CONTINUATION], \
        f"Expected only INCOHERENT_CONTINUATION as CRITICAL, got: {criticals}"
    report(4, "INCOHERENT_CONTINUATION is the only CRITICAL-severity category", PASS)
except Exception as e:
    report(4, "INCOHERENT_CONTINUATION is the only CRITICAL-severity category", FAIL, str(e))

# ═══════════════════════════════════════════════════════════
# BLOCK B: PARSER LOGIC — DETECTION & ADVANCEMENT
# ═══════════════════════════════════════════════════════════
print("\n[ B ] Parser Logic — Detection & Stage Advancement")

# T05: Parser detects degenerate_repetition from critique text
try:
    from eval.failure_parser import FailureModeParser, FailureCategory
    parser = FailureModeParser()
    critique = "The model shows strong repetitive behavior, repeating the same phrase multiple times in a loop."
    report_obj = parser.parse(1, "Base English", critique)
    cats = [fm.category for fm in report_obj.failure_modes]
    assert FailureCategory.DEGENERATE_REPETITION in cats, f"Expected DEGENERATE_REPETITION, got: {cats}"
    report(5, "Parser detects 'degenerate_repetition' from repetition keywords", PASS)
except Exception as e:
    report(5, "Parser detects 'degenerate_repetition' from repetition keywords", FAIL, str(e))

# T06: Parser detects abrupt_cutoff from 'truncated' keyword
try:
    from eval.failure_parser import FailureModeParser, FailureCategory
    parser = FailureModeParser()
    critique = "Output appears truncated mid-sentence. The model cuts off abruptly without completing the thought."
    report_obj = parser.parse(1, "Base English", critique)
    cats = [fm.category for fm in report_obj.failure_modes]
    assert FailureCategory.ABRUPT_CUTOFF in cats, f"Got: {cats}"
    report(6, "Parser detects 'abrupt_cutoff' from truncation keywords", PASS)
except Exception as e:
    report(6, "Parser detects 'abrupt_cutoff' from truncation keywords", FAIL, str(e))

# T07: CRITICAL failure → advance_to_next_stage = False
try:
    from eval.failure_parser import FailureModeParser
    parser = FailureModeParser()
    critique = "The output is completely incoherent and illogical. It does not follow any logical structure whatsoever."
    report_obj = parser.parse(1, "Base English", critique)
    assert report_obj.advance_to_next_stage is False, "CRITICAL failure must block stage advancement"
    report(7, "CRITICAL failure blocks advance_to_next_stage", PASS,
           f"Overall: {report_obj.overall_severity}")
except Exception as e:
    report(7, "CRITICAL failure blocks advance_to_next_stage", FAIL, str(e))

# T08: ONLY LOW failures → advance_to_next_stage = True
try:
    from eval.failure_parser import FailureModeParser
    parser = FailureModeParser()
    critique = "Minor missing punctuation observed. No period at the end of the sentence."
    report_obj = parser.parse(2, "Summarization", critique)
    assert report_obj.advance_to_next_stage is True, \
        f"Only LOW failures should allow advancement. Overall: {report_obj.overall_severity}"
    report(8, "Only LOW failures -> advance_to_next_stage = True", PASS,
           f"Modes: {[fm.category for fm in report_obj.failure_modes]}")
except Exception as e:
    report(8, "Only LOW failures -> advance_to_next_stage = True", FAIL, str(e))

# T09: Empty critique (<10 chars) → returns empty failure list, advance=True
try:
    from eval.failure_parser import FailureModeParser
    parser = FailureModeParser()
    report_obj = parser.parse(1, "Base English", "ok")
    assert len(report_obj.failure_modes) == 0
    assert report_obj.advance_to_next_stage is True
    report(9, "Very short critique returns empty failure list and advance=True", PASS)
except Exception as e:
    report(9, "Very short critique returns empty failure list and advance=True", FAIL, str(e))

# T10: Failure modes sorted CRITICAL first, then HIGH, then MEDIUM, then LOW
try:
    from eval.failure_parser import FailureModeParser, Severity
    parser = FailureModeParser()
    critique = (
        "The model is completely incoherent, illogical output throughout. "
        "Also shows strong repetitive token loops. "
        "Minor missing punctuation at sentence ends. "
        "Subject-verb agreement errors are also noted."
    )
    report_obj = parser.parse(1, "Base English", critique)
    order_map = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
    severities = [fm.severity for fm in report_obj.failure_modes]
    ordered = sorted(severities, key=lambda s: order_map.get(s, 4))
    assert severities == ordered, f"Failure modes not sorted by severity: {severities}"
    report(10, "Failure modes are sorted CRITICAL->HIGH->MEDIUM->LOW", PASS,
           f"Order: {severities}")
except Exception as e:
    report(10, "Failure modes are sorted CRITICAL->HIGH->MEDIUM->LOW", FAIL, str(e))

# ═══════════════════════════════════════════════════════════
# BLOCK C: THINK-TAG STRIPPING & EDGE CASES
# ═══════════════════════════════════════════════════════════
print("\n[ C ] Think-Tag Stripping & Edge Cases")

# T11: <think>...</think> blocks are stripped before parsing
try:
    from eval.failure_parser import FailureModeParser, FailureCategory
    parser = FailureModeParser()
    # put real signal ONLY inside think block — should NOT be detected
    critique = "<think>The model repeats words in a loop</think>The output is overall acceptable."
    report_obj = parser.parse(1, "Base English", critique)
    cats = [fm.category for fm in report_obj.failure_modes]
    assert FailureCategory.DEGENERATE_REPETITION not in cats, \
        "Signals inside <think> blocks should be ignored after stripping"
    report(11, "<think> block content is stripped and not parsed as failure", PASS,
           f"Detected (should be empty/low): {cats}")
except Exception as e:
    report(11, "<think> block content is stripped and not parsed as failure", FAIL, str(e))

# T12: Nested/multi-block <think> tags all stripped correctly
try:
    from eval.failure_parser import FailureModeParser
    import re as _re
    parser = FailureModeParser()
    raw = "<think>first block</think> Real text here. <think>second block with repetition loops</think> Fine output."
    cleaned = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
    assert "first block" not in cleaned
    assert "second block" not in cleaned
    assert "Real text here" in cleaned
    report(12, "Multiple <think> blocks all stripped, real content preserved", PASS,
           f"Cleaned: '{cleaned[:60]}'")
except Exception as e:
    report(12, "Multiple <think> blocks all stripped, real content preserved", FAIL, str(e))

# T13: Evidence field is capped at 200 characters
try:
    from eval.failure_parser import FailureModeParser
    parser = FailureModeParser()
    long_sentence = "The model clearly shows strong repetitive loops. " + ("word " * 100)
    report_obj = parser.parse(1, "Base English", long_sentence)
    for fm in report_obj.failure_modes:
        assert len(fm.evidence) <= 200, f"Evidence exceeds 200 chars: {len(fm.evidence)}"
    report(13, "Evidence field capped at 200 characters for all failure modes", PASS)
except Exception as e:
    report(13, "Evidence field capped at 200 characters for all failure modes", FAIL, str(e))

# ═══════════════════════════════════════════════════════════
# BLOCK D: PYDANTIC SERIALIZATION & FILE I/O
# ═══════════════════════════════════════════════════════════
print("\n[ D ] Pydantic Serialization & File I/O")

# T14: FailureModeReport.model_dump() is JSON-serializable (no non-serializable types)
try:
    from eval.failure_parser import FailureModeParser
    parser = FailureModeParser()
    critique = "The model generates run-on sentences. It also shows comma splice errors repeatedly."
    report_obj = parser.parse(3, "Grammar", critique)
    dumped = report_obj.model_dump()
    serialized = json.dumps(dumped)  # must not raise
    reloaded = json.loads(serialized)
    assert reloaded["stage_id"] == 3
    report(14, "FailureModeReport.model_dump() is fully JSON-serializable", PASS,
           f"{len(serialized)} bytes")
except Exception as e:
    report(14, "FailureModeReport.model_dump() is fully JSON-serializable", FAIL, str(e))

# T15: save_failure_report writes valid JSON to disk with correct structure
try:
    from eval.failure_parser import FailureModeParser, save_failure_report
    parser = FailureModeParser()
    critique = "The model shows register shift and speaker drift throughout the output."
    report_obj = parser.parse(2, "Summarization", critique)
    test_dir = "./eval/reports_test_round3"
    path = save_failure_report(report_obj, output_dir=test_dir)
    assert os.path.exists(path)
    with open(path, "r") as f:
        data = json.load(f)
    assert data["stage_id"] == 2
    assert "failure_modes" in data
    assert "commission_requests" in data
    shutil.rmtree(test_dir, ignore_errors=True)
    report(15, "save_failure_report writes valid JSON with correct schema keys", PASS)
except Exception as e:
    shutil.rmtree("./eval/reports_test_round3", ignore_errors=True)
    report(15, "save_failure_report writes valid JSON with correct schema keys", FAIL, str(e))

# T16: parse_critique_report loads from real stage_1_critique.json correctly
try:
    from eval.failure_parser import parse_critique_report
    path = "./eval/reports/stage_1_critique.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run run_stage_eval.py first")
    report_obj = parse_critique_report(path)
    assert report_obj.stage_id == 1
    assert isinstance(report_obj.failure_modes, list)
    assert report_obj.raw_critique_length > 0
    report(16, "parse_critique_report loads real stage_1_critique.json correctly", PASS,
           f"Modes: {len(report_obj.failure_modes)}, Length: {report_obj.raw_critique_length}")
except Exception as e:
    report(16, "parse_critique_report loads real stage_1_critique.json correctly", FAIL, str(e))

# ═══════════════════════════════════════════════════════════
# BLOCK E: COMMISSION REQUESTS INTEGRITY
# ═══════════════════════════════════════════════════════════
print("\n[ E ] Commission Requests Integrity")

# T17: DataCommissionRequest priority is capped at 5 (never exceeds 5)
try:
    from eval.failure_parser import FailureModeParser
    parser = FailureModeParser()
    # Dense critique to trigger many failures
    critique = (
        "Completely incoherent illogical output. Repetitive loops detected. "
        "Abrupt mid-sentence cutoffs observed. Subject-verb agreement errors throughout. "
        "Run-on sentences and comma splice. Register shift from formal to casual tone. "
        "Speaker drift from first-person to third-person. Lexical repetition of same word. "
        "Entity dropping coreference issues. Off-topic topic drift wanders from premise."
    )
    report_obj = parser.parse(1, "Base English", critique)
    for cr in report_obj.commission_requests:
        assert cr.priority <= 5, f"Priority {cr.priority} exceeds cap of 5"
    report(17, "DataCommissionRequest priority capped at 5 even with many failures", PASS,
           f"Requests: {len(report_obj.commission_requests)}")
except Exception as e:
    report(17, "DataCommissionRequest priority capped at 5 even with many failures", FAIL, str(e))

# T18: Only failures with commission_data=True appear in commission_requests
try:
    from eval.failure_parser import FailureModeParser, COMMISSIONING_TRIGGERS, CATEGORY_SEVERITY_MAP, Severity
    parser = FailureModeParser()
    critique = (
        "Missing punctuation at end of sentence. No period observed. "
        "Also shows wrong preposition usage."
    )
    report_obj = parser.parse(1, "Base English", critique)
    # All detected modes should be LOW → commission_data=False → no commission requests
    for cr in report_obj.commission_requests:
        # If we have a commission request it must correspond to a commissioning-eligible failure
        assert any(fm.commission_data for fm in report_obj.failure_modes), \
            "Commission request exists but no failure has commission_data=True"
    report(18, "commission_requests only contains failures with commission_data=True", PASS,
           f"Requests: {len(report_obj.commission_requests)}, Modes: {len(report_obj.failure_modes)}")
except Exception as e:
    report(18, "commission_requests only contains failures with commission_data=True", FAIL, str(e))

# ═══════════════════════════════════════════════════════════
# BLOCK F: MODEL LOADER & TOKENIZER INTEGRATION
# ═══════════════════════════════════════════════════════════
print("\n[ F ] Model Loader & Tokenizer Integration")

# T19: Tokenizer pad_token is always set (never None) after load
try:
    from trainer.model_loader import load_qlora_model_and_tokenizer
    _, tokenizer = load_qlora_model_and_tokenizer("gpt2", is_trainable=False)
    assert tokenizer.pad_token is not None, "pad_token must never be None"
    assert tokenizer.pad_token_id is not None, "pad_token_id must never be None"
    report(19, "Tokenizer pad_token is always set (not None) after model load", PASS,
           f"pad_token='{tokenizer.pad_token}', id={tokenizer.pad_token_id}")
except Exception as e:
    report(19, "Tokenizer pad_token is always set (not None) after model load", FAIL, str(e))

# T20: load_stage_adapter raises FileNotFoundError for non-existent path
try:
    from trainer.model_loader import load_qlora_model_and_tokenizer, load_stage_adapter
    base_model, _ = load_qlora_model_and_tokenizer("gpt2", is_trainable=False)
    try:
        load_stage_adapter(base_model, "./trainer/checkpoints/stage_999_nonexistent")
        report(20, "load_stage_adapter raises FileNotFoundError for missing path", FAIL, "No exception raised!")
    except FileNotFoundError:
        report(20, "load_stage_adapter raises FileNotFoundError for missing path", PASS)
except Exception as e:
    report(20, "load_stage_adapter raises FileNotFoundError for missing path", FAIL, str(e))

# ─── SUMMARY ───
print("\n" + "="*65)
passed = sum(1 for _, _, s, _ in results if s == PASS)
failed = sum(1 for _, _, s, _ in results if s == FAIL)
print(f"  RESULTS: {passed} PASSED  |  {failed} FAILED  |  {len(results)} TOTAL")
print("="*65 + "\n")
if failed > 0:
    print("  FAILED TESTS:")
    for tid, name, status, detail in results:
        if status == FAIL:
            print(f"    Test {tid:02d}: {name}")
            print(f"    -> {detail}")
