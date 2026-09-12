import os
import json
from dataclasses import dataclass
from typing import List, Dict
from trainer.curriculum.dataset_handler import load_stage_config

@dataclass
class CurriculumStage:
    stage_id: int
    name: str
    description: str
    objectives: List[str]
    target_competence_score: float
    replay_ratio: float
    probe_prompts: List[str]

def get_stage(stage_id: int) -> CurriculumStage:
    """Loads CurriculumStage dataclass instance from JSON config file."""
    config = load_stage_config(stage_id)
    return CurriculumStage(
        stage_id=config["stage_id"],
        name=config["name"],
        description=config["description"],
        objectives=config["objectives"],
        target_competence_score=config.get("target_competence_score", 0.85),
        replay_ratio=config.get("replay_ratio", 0.2),
        probe_prompts=config.get("probe_prompts", [])
    )

STAGES: Dict[int, CurriculumStage] = {stage_id: get_stage(stage_id) for stage_id in range(1, 6)}
