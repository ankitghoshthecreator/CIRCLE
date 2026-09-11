from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class CurriculumStage:
    stage_id: int
    name: str
    description: str
    objectives: List[str]
    replay_ratio: float = 0.2

STAGES: Dict[int, CurriculumStage] = {
    1: CurriculumStage(
        stage_id=1,
        name="Base English",
        description="Next-token prediction on general English corpora for fluency and basic syntax.",
        objectives=["token_fluency", "basic_syntax", "vocab_distribution"],
        replay_ratio=0.0
    ),
    2: CurriculumStage(
        stage_id=2,
        name="Summarization",
        description="Compress long passages into faithful summaries for meaning extraction.",
        objectives=["meaning_extraction", "passage_compression", "content_retention"],
        replay_ratio=0.2
    ),
    3: CurriculumStage(
        stage_id=3,
        name="Grammar",
        description="Explicit syntactic well-formedness, subject-verb agreement, and tense consistency.",
        objectives=["syntactic_correctness", "agreement", "tense_consistency"],
        replay_ratio=0.2
    ),
    4: CurriculumStage(
        stage_id=4,
        name="Punctuation",
        description="Fine-grained clause boundary marking and punctuation dynamics.",
        objectives=["clause_boundaries", "punctuation_precision", "sentence_segmentation"],
        replay_ratio=0.25
    ),
    5: CurriculumStage(
        stage_id=5,
        name="Conversational Understanding",
        description="Multi-turn dialogue, speaker tracking, and context preservation.",
        objectives=["turn_taking", "speaker_tracking", "context_carryover"],
        replay_ratio=0.3
    ),
}
