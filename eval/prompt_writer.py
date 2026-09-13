"""
eval/prompt_writer.py
Part 7: Targeted Prompt Writer

Synthesizes high-quality, targeted data-generation meta-prompts based on
DataCommissionRequest objects emitted by Part 6 (FailureModeParser). These
meta-prompts instruct downstream generator engines (DeepSeek 32B / Ollama / vLLM)
to produce training samples specifically addressing identified model failure modes.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from eval.failure_parser import DataCommissionRequest, FailureModeReport, FailureCategory

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Expert Meta-Prompt Templates for Synthetic Data Generators
# ──────────────────────────────────────────────────────────────

META_PROMPT_TEMPLATES: Dict[str, str] = {
    # Fluency / Generation
    FailureCategory.DEGENERATE_REPETITION.value: (
        "Generate training examples containing varied sentence structure and high lexical diversity. "
        "Each target completion MUST avoid repeating identical n-grams, tokens, or phrases. "
        "Focus area: {focus_hint}. Provide clean, progressive explanations without looping."
    ),
    FailureCategory.ABRUPT_CUTOFF.value: (
        "Generate complete, fully resolved text outputs with proper concluding punctuation and closing logic. "
        "Do not terminate mid-thought or mid-sentence. "
        "Focus area: {focus_hint}. Ensure natural closing transitions."
    ),
    FailureCategory.INCOHERENT_CONTINUATION.value: (
        "Generate paired premises and highly logical, coherent follow-on completions. "
        "Each completion must strictly maintain cause-and-effect reasoning and context consistency. "
        "Focus area: {focus_hint}."
    ),

    # Syntax / Grammar
    FailureCategory.SUBJECT_VERB_MISMATCH.value: (
        "Generate sentences with complex noun phrases, compound subjects, and pre-modifying clauses. "
        "Ensure strict subject-verb agreement across singular, plural, and collective nouns. "
        "Focus area: {focus_hint}."
    ),
    FailureCategory.RUN_ON_SENTENCE.value: (
        "Generate well-structured paragraph examples demonstrating clear sentence boundary segmentation. "
        "Use appropriate conjunctions, semicolons, and periods to separate independent clauses cleanly. "
        "Focus area: {focus_hint}."
    ),
    FailureCategory.CLAUSE_BOUNDARY_ERROR.value: (
        "Generate complex multi-clause sentences (subordinate, relative, and coordinate clauses) "
        "with correct comma placement and structural subordinating conjunctions. "
        "Focus area: {focus_hint}."
    ),
    FailureCategory.MISSING_PUNCTUATION.value: (
        "Generate text samples demonstrating meticulous punctuation usage, including quotation marks, "
        "commas in series, apostrophes, and terminal punctuation (periods, question marks). "
        "Focus area: {focus_hint}."
    ),
    FailureCategory.INCORRECT_PREPOSITION.value: (
        "Generate idiomatic text examples focusing on accurate prepositional phrases, phrasal verbs, "
        "and spatial/temporal prepositions (e.g., 'depend on', 'interested in', 'prior to'). "
        "Focus area: {focus_hint}."
    ),

    # Semantics / Discourse
    FailureCategory.ENTITY_DROPPING.value: (
        "Generate multi-paragraph passages containing multiple named entities (people, places, organizations). "
        "Ensure coreference continuity throughout the passage without dropping or losing track of entities. "
        "Focus area: {focus_hint}."
    ),
    FailureCategory.SPEAKER_DRIFT.value: (
        "Generate conversational or narrative passages with consistent perspective and grammatical person "
        "(1st person 'I/we', 2nd person 'you', or 3rd person 'he/she/they'). Avoid mid-text perspective shifts. "
        "Focus area: {focus_hint}."
    ),
    FailureCategory.REGISTER_SHIFT.value: (
        "Generate stylized responses that maintain a uniform tone and register (e.g., academic, business formal, "
        "or technical documentation) without slipping into informal slang or colloquialisms. "
        "Focus area: {focus_hint}."
    ),
    FailureCategory.SEMANTIC_CIRCULARITY.value: (
        "Generate informational responses that provide novel facts, details, and actionable content rather than "
        "merely restating or rephrasing the input prompt in circles. "
        "Focus area: {focus_hint}."
    ),
    FailureCategory.TOPIC_DRIFT.value: (
        "Generate highly focused responses that directly answer the core user prompt without tangential detours "
        "or irrelevant background narrative. "
        "Focus area: {focus_hint}."
    ),

    # Vocabulary
    FailureCategory.LEXICAL_REPETITION.value: (
        "Generate passage examples using rich vocabulary and precise synonyms, avoiding overusing the same "
        "key verbs or adjectives across consecutive sentences. "
        "Focus area: {focus_hint}."
    ),
    FailureCategory.VOCABULARY_POVERTY.value: (
        "Generate sophisticated text samples incorporating advanced, precise vocabulary and descriptive verbs. "
        "Focus area: {focus_hint}."
    ),
    FailureCategory.DOMAIN_VOCABULARY_MISS.value: (
        "Generate specialized domain text (technical, legal, scientific, or medical) with accurate, "
        "industry-standard terminology and concepts. "
        "Focus area: {focus_hint}."
    ),

    # Default / Other
    FailureCategory.OTHER.value: (
        "Generate clean, high-quality, diverse training examples addressing general generation quality. "
        "Focus area: {focus_hint}."
    )
}


# ──────────────────────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────────────────────

class TargetedPromptSpec(BaseModel):
    request_id: str = Field(..., description="Unique identifier for this prompt specification")
    stage_id: int = Field(..., description="Curriculum stage ID requiring data generation")
    category: str = Field(..., description="Failure category string")
    severity: str = Field(..., description="Severity level")
    priority: int = Field(..., description="Priority ranking (1=highest, 5=lowest)")
    meta_prompt: str = Field(..., description="Formatted prompt instruction for synthetic data generator")
    suggested_prompt_focus: str = Field(..., description="Specific weakness focus hint")
    target_sample_count: int = Field(10, description="Number of synthetic examples to generate")
    generation_guidance: Dict[str, str] = Field(default_factory=dict, description="Metadata guidance for generator")

    model_config = {"use_enum_values": True}


class TargetedPromptBatch(BaseModel):
    stage_id: int
    stage_name: str
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    total_requests: int
    total_target_samples: int
    prompt_specs: List[TargetedPromptSpec] = Field(default_factory=list)

    model_config = {"use_enum_values": True}


# ──────────────────────────────────────────────────────────────
# Targeted Prompt Writer Engine
# ──────────────────────────────────────────────────────────────

class TargetedPromptWriter:
    """
    Ingests failure reports & commission requests, producing actionable prompt
    batches for the synthetic data generator.
    """

    def __init__(self, custom_templates: Optional[Dict[str, str]] = None):
        self.templates = META_PROMPT_TEMPLATES.copy()
        if custom_templates:
            self.templates.update(custom_templates)

    def get_template(self, category: str) -> str:
        """Returns the meta-prompt template for a category, falling back to OTHER."""
        return self.templates.get(category, self.templates[FailureCategory.OTHER.value])

    def generate_prompt_spec(
        self,
        request: DataCommissionRequest,
        idx: int = 1,
        num_samples: int = 10
    ) -> TargetedPromptSpec:
        """Translates a single DataCommissionRequest into a TargetedPromptSpec."""
        cat_str = request.failure_category
        focus_hint = request.suggested_prompt_focus or f"Correcting {cat_str} issues"

        template = self.get_template(cat_str)
        meta_prompt = template.format(focus_hint=focus_hint)

        req_id = f"stage_{request.stage_id}_req_{idx:02d}_{cat_str}"

        return TargetedPromptSpec(
            request_id=req_id,
            stage_id=request.stage_id,
            category=cat_str,
            severity=request.severity,
            priority=request.priority,
            meta_prompt=meta_prompt,
            suggested_prompt_focus=focus_hint,
            target_sample_count=num_samples,
            generation_guidance={
                "target_model_engine": "DeepSeek-32B",
                "temperature": "0.7",
                "top_p": "0.95"
            }
        )

    def build_batch_from_report(
        self,
        report: FailureModeReport,
        default_samples_per_request: int = 10
    ) -> TargetedPromptBatch:
        """
        Builds a full TargetedPromptBatch from a FailureModeReport,
        sorting specs by priority (1 = highest priority).
        """
        specs: List[TargetedPromptSpec] = []
        for idx, req in enumerate(report.commission_requests, 1):
            # Scale sample count slightly by priority (high priority gets more samples)
            priority_multiplier = max(1, (6 - req.priority))
            samples = default_samples_per_request * priority_multiplier // 2
            spec = self.generate_prompt_spec(req, idx=idx, num_samples=samples)
            specs.append(spec)

        # Sort by priority (1 -> 5)
        specs.sort(key=lambda s: (s.priority, s.category))

        total_samples = sum(s.target_sample_count for s in specs)

        return TargetedPromptBatch(
            stage_id=report.stage_id,
            stage_name=report.stage_name,
            total_requests=len(specs),
            total_target_samples=total_samples,
            prompt_specs=specs
        )

    def save_prompt_batch(self, batch: TargetedPromptBatch, output_path: str) -> str:
        """Saves TargetedPromptBatch as JSON to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(batch.model_dump(), indent=2))
        logger.info(f"Saved TargetedPromptBatch ({batch.total_requests} specs) to '{output_path}'")
        return output_path

    def load_prompt_batch(self, input_path: str) -> TargetedPromptBatch:
        """Loads TargetedPromptBatch from JSON file."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"TargetedPromptBatch file not found: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TargetedPromptBatch(**data)
