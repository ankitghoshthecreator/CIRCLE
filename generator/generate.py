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
        model_name: str = "deepseek-r1:32b",
        request_timeout_sec: float = 10.0
    ):
        self.backend = backend.lower()
        self.endpoint_url = endpoint_url.strip().rstrip("/")
        self.model_name = model_name
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
            with urllib.request.urlopen(req, timeout=0.2) as resp:
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
