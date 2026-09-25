"""
generator/generate.py
Part 9: Local DeepSeek Data Generator Engine

Ingests TargetedPromptBatch specifications produced by Part 7 (TargetedPromptWriter)
and executes synthetic data generation tasks against local LLM endpoints
(Ollama API, vLLM OpenAI API, or local DeepSeek models). Produces structured
SyntheticDataBatch training pairs ready for validation and curriculum replay buffer integration.
"""

import os
import json
import time
import logging
import urllib.request
import urllib.error
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from eval.prompt_writer import TargetedPromptSpec, TargetedPromptBatch
from trainer.curriculum.dataset_handler import CurriculumExample

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Pydantic Schemas for Synthetic Data
# ──────────────────────────────────────────────────────────────

class SyntheticDataSample(BaseModel):
    sample_id: str = Field(..., description="Unique identifier for this synthetic training example")
    stage_id: int = Field(..., description="Curriculum stage ID")
    category: str = Field(..., description="Targeted failure category")
    input_text: str = Field(..., description="Generated input prompt/question")
    target_text: str = Field(..., description="Generated high-quality ground-truth completion")
    prompt_spec_id: str = Field(..., description="Originating TargetedPromptSpec request_id")
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    reward_score: float = Field(default=0.0, description="RL reward score from Groq critic (0–1); 0.0 = not yet scored")

    model_config = {"use_enum_values": True}


class SyntheticDataBatch(BaseModel):
    stage_id: int
    total_samples: int
    samples: List[SyntheticDataSample] = Field(default_factory=list)
    generation_time_sec: float = Field(0.0, description="Total generation time in seconds")

    model_config = {"use_enum_values": True}


# ──────────────────────────────────────────────────────────────
# Local Data Generator Engine
# ──────────────────────────────────────────────────────────────

class LocalDataGeneratorEngine:
    """
    Local synthetic data generator supporting Ollama API, vLLM HTTP API,
    and automatic Mock fallback execution.
    """

    def __init__(
        self,
        backend: str = "ollama",
        endpoint_url: str = "http://localhost:11434",
        model_name: str = None,
        request_timeout_sec: float = 10.0
    ):
        import os
        self.backend = backend.lower()
        self.endpoint_url = endpoint_url.strip().rstrip("/")
        # Prefer env var so any locally pulled model works without code changes
        self.model_name = (
            model_name
            or os.getenv("GENERATOR_MODEL", "qwen2.5-coder:7b")
        )
        self.request_timeout_sec = request_timeout_sec
        self._endpoint_checked = False
        self._endpoint_available = False

    def is_endpoint_available(self) -> bool:
        """Checks if the configured local server endpoint is responding (cached)."""
        if self._endpoint_checked:
            return self._endpoint_available
        try:
            url = f"{self.endpoint_url}/api/tags" if self.backend == "ollama" else f"{self.endpoint_url}/models"
            req = urllib.request.Request(url, headers={"User-Agent": "CIRCLE-Generator"})
            # Use 3 s timeout (was 0.2 s) so Ollama has time to respond on slower machines
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                self._endpoint_available = (resp.status == 200)
        except Exception:
            self._endpoint_available = False
        self._endpoint_checked = True
        return self._endpoint_available

    def generate_samples_for_spec(
        self,
        spec: TargetedPromptSpec,
        num_samples: Optional[int] = None
    ) -> List[SyntheticDataSample]:
        """Generates synthetic training samples for a single TargetedPromptSpec."""
        count = num_samples if num_samples is not None else spec.target_sample_count
        samples: List[SyntheticDataSample] = []

        endpoint_active = self.is_endpoint_available()
        if not endpoint_active:
            logger.debug(f"Local {self.backend} endpoint not detected at '{self.endpoint_url}'. Using Mock Fallback Generator.")

        for idx in range(1, count + 1):
            if endpoint_active and self.backend == "ollama":
                sample = self._generate_via_ollama(spec, idx)
            elif endpoint_active and self.backend == "vllm":
                sample = self._generate_via_vllm(spec, idx)
            else:
                sample = self._generate_mock_sample(spec, idx)
            samples.append(sample)

        return samples

    def _generate_mock_sample(self, spec: TargetedPromptSpec, idx: int) -> SyntheticDataSample:
        """Generates a high-quality mock synthetic training pair for local testing."""
        sample_id = f"synth_s{spec.stage_id}_{spec.category}_{idx:03d}"
        focus = spec.suggested_prompt_focus

        if spec.category == "degenerate_repetition":
            inp = f"Explain the concept of progressive iteration in machine learning ({focus})."
            tgt = (
                "Progressive iteration refers to step-by-step refinement of model weights. "
                "Each stage introduces new objectives without repeating previous mistakes."
            )
        elif spec.category == "speaker_drift":
            inp = f"Write a first-person perspective narrative about space exploration ({focus})."
            tgt = (
                "I gazed through the viewport as Earth shrank into a pale blue dot. "
                "My pulse raced with excitement for the uncharted void ahead."
            )
        elif spec.category == "abrupt_cutoff":
            inp = f"Write a complete summary of data pipeline validation ({focus})."
            tgt = (
                "Data pipeline validation ensures schema integrity and deduplication before ingestion. "
                "This guarantees reliable model training outcomes."
            )
        else:
            inp = f"Synthetic prompt for curriculum stage {spec.stage_id} addressing {spec.category} ({focus})."
            tgt = f"Synthetic target response resolving {spec.category} with clear structural quality."

        return SyntheticDataSample(
            sample_id=sample_id,
            stage_id=spec.stage_id,
            category=spec.category,
            input_text=inp,
            target_text=tgt,
            prompt_spec_id=spec.request_id,
            generation_metadata={
                "backend": "mock_fallback",
                "model_name": self.model_name,
                "priority": spec.priority
            }
        )

    def _generate_via_ollama(self, spec: TargetedPromptSpec, idx: int) -> SyntheticDataSample:
        """Executes HTTP request to local Ollama API server."""
        sample_id = f"synth_s{spec.stage_id}_{spec.category}_{idx:03d}"
        url = f"{self.endpoint_url}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": f"{spec.meta_prompt}\nGenerate a JSON object with 'input_text' and 'target_text'.",
            "stream": False,
            "format": "json"
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=min(2.0, self.request_timeout_sec)) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                response_text = result.get("response", "{}")
                parsed = json.loads(response_text)
                inp = parsed.get("input_text", f"Generated prompt for {spec.category}")
                tgt = parsed.get("target_text", f"Generated target response for {spec.category}")

                return SyntheticDataSample(
                    sample_id=sample_id,
                    stage_id=spec.stage_id,
                    category=spec.category,
                    input_text=inp,
                    target_text=tgt,
                    prompt_spec_id=spec.request_id,
                    generation_metadata={"backend": "ollama", "model_name": self.model_name}
                )
        except Exception as e:
            logger.debug(f"Ollama API request failed ({e}). Disabling endpoint for batch and falling back to mock generator.")
            self._endpoint_available = False
            return self._generate_mock_sample(spec, idx)

    def _generate_via_vllm(self, spec: TargetedPromptSpec, idx: int) -> SyntheticDataSample:
        """Executes HTTP request to local vLLM OpenAI-compatible server."""
        sample_id = f"synth_s{spec.stage_id}_{spec.category}_{idx:03d}"
        url = f"{self.endpoint_url}/v1/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a synthetic dataset generator. Return JSON with keys 'input_text' and 'target_text'."},
                {"role": "user", "content": spec.meta_prompt}
            ],
            "temperature": 0.7
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=min(2.0, self.request_timeout_sec)) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return SyntheticDataSample(
                    sample_id=sample_id,
                    stage_id=spec.stage_id,
                    category=spec.category,
                    input_text=parsed.get("input_text", f"Generated input for {spec.category}"),
                    target_text=parsed.get("target_text", f"Generated target for {spec.category}"),
                    prompt_spec_id=spec.request_id,
                    generation_metadata={"backend": "vllm", "model_name": self.model_name}
                )
        except Exception as e:
            logger.debug(f"vLLM API request failed ({e}). Disabling endpoint for batch and falling back to mock generator.")
            self._endpoint_available = False
            return self._generate_mock_sample(spec, idx)

    def generate_batch_from_prompts(
        self,
        prompt_batch: TargetedPromptBatch,
        max_samples_per_spec: Optional[int] = None
    ) -> SyntheticDataBatch:
        """Executes data generation across all prompt specifications in a TargetedPromptBatch."""
        start_t = time.time()
        all_samples: List[SyntheticDataSample] = []

        logger.info(f"Starting synthetic data generation for Stage {prompt_batch.stage_id} ({prompt_batch.total_requests} specs)...")

        for spec in prompt_batch.prompt_specs:
            count = max_samples_per_spec if max_samples_per_spec is not None else spec.target_sample_count
            spec_samples = self.generate_samples_for_spec(spec, num_samples=count)
            all_samples.extend(spec_samples)

        elapsed = round(time.time() - start_t, 3)

        return SyntheticDataBatch(
            stage_id=prompt_batch.stage_id,
            total_samples=len(all_samples),
            samples=all_samples,
            generation_time_sec=elapsed
        )

    def save_synthetic_batch(self, batch: SyntheticDataBatch, output_path: str) -> str:
        """Saves SyntheticDataBatch as JSON file to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(batch.model_dump(), indent=2))
        logger.info(f"Saved SyntheticDataBatch ({batch.total_samples} samples) to '{output_path}'")
        return output_path

    # ──────────────────────────────────────────────────────────────
    # RL-specific generation (OllamaPromptRequest format)
    # ──────────────────────────────────────────────────────────────

    def rl_generate_from_request(
        self,
        request,            # OllamaPromptRequest from generator.rl_prompt_builder
        reward_score: float = 0.0,
    ) -> "SyntheticDataBatch":
        """
        Execute one OllamaPromptRequest from the RL loop.

        Sends the system + user prompt directly to Ollama (or mock) and
        attaches the Groq reward_score to every generated sample so it
        can be used later for weighted training.

        Args:
            request:       OllamaPromptRequest built by RLPromptBuilder.
            reward_score:  Scalar reward from RLRewardCalculator (0–1).

        Returns:
            SyntheticDataBatch ready for validation and merging.
        """
        import time
        start_t = time.time()
        samples: List[SyntheticDataSample] = []

        endpoint_active = self.is_endpoint_available()

        for idx in range(1, request.num_samples + 1):
            sample_id = f"rl_s{request.stage_id}_{request.failure_category}_{idx:03d}"

            if endpoint_active and self.backend == "ollama":
                raw = self._rl_call_ollama(
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    sample_id=sample_id,
                    request=request,
                    idx=idx,
                )
            else:
                raw = self._rl_mock_sample(
                    sample_id=sample_id,
                    request=request,
                    idx=idx,
                )

            # Attach RL reward score
            raw.reward_score = reward_score
            samples.append(raw)

        elapsed = round(time.time() - start_t, 3)
        batch = SyntheticDataBatch(
            stage_id=request.stage_id,
            total_samples=len(samples),
            samples=samples,
            generation_time_sec=elapsed,
        )
        logger.info(
            f"[RL-Generate] {len(samples)} samples for [{request.severity}] "
            f"{request.failure_category} via {'ollama' if endpoint_active else 'mock'} "
            f"in {elapsed:.2f}s (reward={reward_score:.3f})"
        )
        return batch

    def _rl_call_ollama(
        self, system_prompt: str, user_prompt: str,
        sample_id: str, request, idx: int
    ) -> "SyntheticDataSample":
        """Send a chat-style request to Ollama /api/chat and parse JSON array."""
        url = f"{self.endpoint_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.request_timeout_sec) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result.get("message", {}).get("content", "{}")
                # Ollama may return a JSON array or a single object
                parsed = json.loads(content)
                if isinstance(parsed, list) and parsed:
                    item = parsed[min(idx - 1, len(parsed) - 1)]
                elif isinstance(parsed, dict):
                    item = parsed
                else:
                    item = {}
                inp = item.get("input_text", f"Corrective prompt for {request.failure_category} ({idx})")
                tgt = item.get("target_text", f"Corrective response for {request.failure_category} ({idx})")
                return SyntheticDataSample(
                    sample_id=sample_id,
                    stage_id=request.stage_id,
                    category=request.failure_category,
                    input_text=inp,
                    target_text=tgt,
                    prompt_spec_id=request.request_id,
                    generation_metadata={
                        "backend": "ollama_rl",
                        "model_name": self.model_name,
                        "severity": request.severity,
                    },
                )
        except Exception as e:
            logger.warning(f"[RL-Generate] Ollama call failed ({e}). Falling back to mock.")
            self._endpoint_available = False
            return self._rl_mock_sample(sample_id, request, idx)

    def _rl_mock_sample(
        self, sample_id: str, request, idx: int
    ) -> "SyntheticDataSample":
        """Deterministic high-quality mock sample for RL loop testing."""
        cat = request.failure_category
        mocks = {
            "degenerate_repetition": (
                f"Describe the process of photosynthesis in detail (sample {idx}).",
                "Photosynthesis converts light energy into glucose. Chlorophyll absorbs sunlight, "
                "water molecules are split releasing oxygen, and carbon dioxide is fixed into sugars "
                "through the Calvin cycle.",
            ),
            "abrupt_cutoff": (
                f"Explain why regular exercise is important for health (sample {idx}).",
                "Regular exercise strengthens the cardiovascular system, improves mental health by "
                "releasing endorphins, and helps maintain a healthy body weight. Consistent physical "
                "activity also reduces the risk of chronic diseases such as diabetes and hypertension.",
            ),
            "speaker_drift": (
                f"Write a first-person account of visiting a new city (sample {idx}).",
                "I arrived at the station as the sun was setting, my bag heavy on my shoulder. "
                "I found a small café near the square and ordered coffee, watching the locals "
                "pass by as I planned my first evening in the city.",
            ),
            "vocabulary_poverty": (
                f"Describe the atmosphere of a thunderstorm (sample {idx}).",
                "The tempestuous storm unleashed torrential rain across the parched landscape. "
                "Jagged lightning illuminated the roiling cumulus clouds while resonant thunder "
                "reverberated through the valley below.",
            ),
        }
        inp, tgt = mocks.get(
            cat,
            (
                f"Generate a high-quality response addressing {cat} (sample {idx}).",
                f"This is a well-formed, complete, and diverse response that correctly "
                f"addresses the linguistic requirement for stage {request.stage_id}.",
            )
        )
        return SyntheticDataSample(
            sample_id=sample_id,
            stage_id=request.stage_id,
            category=cat,
            input_text=inp,
            target_text=tgt,
            prompt_spec_id=request.request_id,
            generation_metadata={
                "backend": "mock_rl",
                "model_name": "mock",
                "severity": request.severity,
            },
        )

    def load_synthetic_batch(self, input_path: str) -> SyntheticDataBatch:
        """Loads SyntheticDataBatch from JSON file."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"SyntheticDataBatch file not found: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SyntheticDataBatch(**data)

    def convert_to_curriculum_examples(self, batch: SyntheticDataBatch) -> List[CurriculumExample]:
        """Converts SyntheticDataBatch into CurriculumExample objects for training replay buffers."""
        examples: List[CurriculumExample] = []
        for s in batch.samples:
            ex = CurriculumExample(
                input_text=s.input_text,
                target_text=s.target_text,
                stage_id=s.stage_id,
                source="synthetic",
                metadata={
                    "sample_id": s.sample_id,
                    "category": s.category,
                    "prompt_spec_id": s.prompt_spec_id,
                    "generated_at": s.generated_at
                }
            )
            examples.append(ex)
        return examples
