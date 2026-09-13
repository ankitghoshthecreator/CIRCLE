# Walkthrough - Part 7: Targeted Prompt Writer

Part 7 adds the **Targeted Prompt Writer** engine ([eval/prompt_writer.py](file:///d:/CIRCLE/eval/prompt_writer.py)), completing the bridge between model failure diagnosis (Part 6) and synthetic data generation (Part 9).

---

## Changes Implemented

### 1. Evaluator Engine ([eval/prompt_writer.py](file:///d:/CIRCLE/eval/prompt_writer.py))
- **`META_PROMPT_TEMPLATES`**: Curated expert templates covering all 17 failure mode categories (`DEGENERATE_REPETITION`, `SPEAKER_DRIFT`, `CLAUSE_BOUNDARY_ERROR`, `INCOHERENT_CONTINUATION`, etc.).
- **`TargetedPromptSpec` & `TargetedPromptBatch`**: Pydantic schemas for JSON-serializable generation specifications with priority multipliers, sample counts, and engine parameters (`temperature`, `top_p`).
- **`TargetedPromptWriter`**:
  - `generate_prompt_spec()`: Maps a `DataCommissionRequest` into a `TargetedPromptSpec`.
  - `build_batch_from_report()`: Groups, scales sample counts by priority, and sorts specifications (1 = highest priority).
  - `save_prompt_batch()` & `load_prompt_batch()`: High-performance JSON file persistence.

### 2. Unit Testing ([tests/test_part7.py](file:///d:/CIRCLE/tests/test_part7.py))
- 8 comprehensive unit tests covering template coverage, prompt formulation, priority sorting, sample multiplier scaling, JSON I/O, and real critique integration.

---

## Verification Results

### Unit & Regression Tests (48 / 48 Passed)
| Test Module | Tests | Result | Focus |
| :--- | :---: | :---: | :--- |
| `tests/test_part7.py` | 8 | **PASSED** | Part 7 prompt specs, meta-prompt templates, priority sorting, JSON I/O |
| `tests/test_round3.py` | 20 | **PASSED** | Parser taxonomy, severity rules, think-tag stripping, Pydantic schemas |
| `tests/test_integration.py` | 10 | **PASSED** | Training loss decay, adapter checkpoints, dataset roundtrips |
| `tests/test_all_parts.py` | 10 | **PASSED** | Replay buffer, tokenizer pad token, stage configs, Groq critic |

**Total**: **48 / 48 PASSED (100%)**

---

## Remote Git Status
- **Commit**: `6f2766c` — `feat(eval): implement Part 7 Targeted Prompt Writer with meta-prompt templates & priority-sorted batching`
- **Branch**: `main -> main` on `https://github.com/ankitghoshthecreator/CIRCLE.git`
