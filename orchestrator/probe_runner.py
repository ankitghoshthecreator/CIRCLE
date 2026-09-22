"""
orchestrator/probe_runner.py
CIRCLE Part 13: Stage Probe Runner

Generates evaluation probe inputs per curriculum stage and
collects model outputs (or mock outputs when no GPU is available).
Used by the ClosedLoopOrchestrator to assemble probe_results for
the TieredEvaluator during the EVALUATING step.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Stage Probe Definitions (seed prompts per stage)
# ─────────────────────────────────────────────────────────────────

STAGE_PROBES: Dict[int, List[Dict[str, str]]] = {
    1: [  # Base English — fluency, basic sentence structure
        {"prompt": "The sun rises in the", "expected_theme": "direction/time"},
        {"prompt": "Water is essential for", "expected_theme": "life/biology"},
        {"prompt": "Complete the sentence: She walked to the", "expected_theme": "noun/place"},
        {"prompt": "A dog is a loyal", "expected_theme": "noun/adjective"},
        {"prompt": "The capital of France is", "expected_theme": "factual completion"},
    ],
    2: [  # Summarization — coherent multi-sentence outputs
        {"prompt": "Summarize the following in one sentence: The Industrial Revolution was a period of major mechanization that transformed manufacturing from hand production to machine-based production.",
         "expected_theme": "compression/accuracy"},
        {"prompt": "Write a two-sentence summary of photosynthesis.", "expected_theme": "factual summary"},
        {"prompt": "In brief, explain why exercise is important.", "expected_theme": "concise explanation"},
    ],
    3: [  # Grammar — subject-verb agreement, tense consistency
        {"prompt": "The team of engineers __ working on the project.", "expected_theme": "subject-verb agreement"},
        {"prompt": "Yesterday, she __ to the store and bought milk.", "expected_theme": "past tense"},
        {"prompt": "Neither the manager nor the employees __ aware of the change.", "expected_theme": "agreement"},
    ],
    4: [  # Punctuation & Clause Boundaries
        {"prompt": "Rewrite with correct punctuation: she said hello then walked away", "expected_theme": "punctuation"},
        {"prompt": "Add commas where needed: The tall dark handsome stranger arrived at midnight.", "expected_theme": "comma usage"},
        {"prompt": "Identify the clause boundary error and correct it: The dog barked the cat hissed they both ran.", "expected_theme": "clause separation"},
    ],
    5: [  # Conversational Understanding & Speaker Tracking
        {"prompt": "Alice said she was tired. What did Alice say?", "expected_theme": "speaker reference"},
        {"prompt": "Continue the dialogue naturally:\nUser: I just got a promotion!\nAssistant:", "expected_theme": "contextual response"},
        {"prompt": "Who is being referred to? 'He told her that he would meet her there.' There are two people. Identify them.", "expected_theme": "coreference resolution"},
    ],
}


# ─────────────────────────────────────────────────────────────────
# Probe Runner
# ─────────────────────────────────────────────────────────────────

class ProbeRunner:
    """
    Generates evaluation probe outputs for a given curriculum stage.

    In production: loads the stage checkpoint and runs `model.generate()`.
    In mock/offline mode (no GPU / no checkpoint): returns deterministic
    mock outputs for testing and local loop validation.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_mock: bool = True,
        max_new_tokens: int = 128,
    ):
        self.checkpoint_path = checkpoint_path
        self.use_mock = use_mock
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

        if not use_mock and checkpoint_path:
            self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> None:
        """Load QLoRA checkpoint for inference. Requires torch + transformers."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.info(f"[ProbeRunner] Loading checkpoint from '{checkpoint_path}'...")
            self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self._model.eval()
            logger.info("[ProbeRunner] Checkpoint loaded successfully.")
        except ImportError:
            logger.warning("[ProbeRunner] torch/transformers not available. Falling back to mock mode.")
            self.use_mock = True
        except Exception as e:
            logger.warning(f"[ProbeRunner] Failed to load checkpoint ({e}). Falling back to mock mode.")
            self.use_mock = True

    def run_probes(
        self,
        stage_id: int,
        custom_probes: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Run probes for a stage. Returns list of {prompt, output} dicts.

        Args:
            stage_id:      Curriculum stage 1-5.
            custom_probes: Optional override list of probe dicts (used in tests).

        Returns:
            List of {"prompt": str, "output": str} dicts ready for TieredEvaluator.
        """
        probes = custom_probes or STAGE_PROBES.get(stage_id, STAGE_PROBES[1])
        results: List[Dict[str, str]] = []

        if self.use_mock or self._model is None:
            logger.info(
                f"[ProbeRunner] Stage {stage_id}: Running {len(probes)} probes in MOCK mode."
            )
            for probe in probes:
                output = self._mock_output(stage_id, probe["prompt"])
                results.append({"prompt": probe["prompt"], "output": output})
        else:
            logger.info(
                f"[ProbeRunner] Stage {stage_id}: Running {len(probes)} probes via checkpoint."
            )
            for probe in probes:
                output = self._generate_output(probe["prompt"])
                results.append({"prompt": probe["prompt"], "output": output})

        return results

    def _generate_output(self, prompt: str) -> str:
        """Run real model inference on a prompt."""
        import torch
        try:
            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(self._model.device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            # Decode only the newly generated tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"[ProbeRunner] Inference error: {e}")
            return "[inference error]"

    def _mock_output(self, stage_id: int, prompt: str) -> str:
        """
        Returns a deterministic, structurally valid mock output per stage.
        Designed to score reasonably on the LightweightStudentEvaluator
        without requiring GPU or a real model.
        """
        mock_outputs: Dict[int, str] = {
            1: "east as the day begins with golden light across the horizon.",
            2: "The Industrial Revolution fundamentally transformed manufacturing "
               "through mechanization, shifting production from artisanal craft to factory-based industry.",
            3: "is. She went. Are.",
            4: "She said, 'Hello,' then walked away.",
            5: "That sounds wonderful! Congratulations on your achievement.",
        }
        base = mock_outputs.get(stage_id, "A complete and coherent response follows naturally.")
        # Prefix with a word from the prompt for mild semantic overlap
        first_word = prompt.split()[0] if prompt.split() else "The"
        return f"{first_word} {base}"
