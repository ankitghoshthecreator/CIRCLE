import os
import glob
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
        objectives=config.get("objectives", []),
        target_competence_score=config.get("target_competence_score", 0.85),
        replay_ratio=config.get("replay_ratio", 0.2),
        probe_prompts=config.get("probe_prompts", [])
    )

def _load_all_stages() -> Dict[int, CurriculumStage]:
    """Dynamically loads all available stage configurations from trainer/curriculum/configs/."""
    configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    stages = {}
    if os.path.exists(configs_dir):
        for fname in os.listdir(configs_dir):
            if fname.startswith("stage_") and fname.endswith(".json"):
                try:
                    sid = int(fname.replace("stage_", "").replace(".json", ""))
                    stages[sid] = get_stage(sid)
                except ValueError:
                    pass
    if not stages:
        # Default fallback stages 1..5
        stages = {stage_id: get_stage(stage_id) for stage_id in range(1, 6)}
    return dict(sorted(stages.items()))

STAGES: Dict[int, CurriculumStage] = _load_all_stages()
