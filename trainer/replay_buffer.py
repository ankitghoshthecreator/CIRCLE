import os
import json
import random
import logging
from typing import List, Dict
from trainer.curriculum.dataset_handler import load_stage_dataset, CurriculumExample

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ReplayBufferManager:
    """Manages historical stage datasets and provides stratified sampling to prevent catastrophic forgetting."""
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir

    def load_prior_stage_examples(self, current_stage_id: int) -> Dict[int, List[CurriculumExample]]:
        """Loads dataset examples for all completed prior stages (1 to current_stage_id - 1)."""
        prior_data = {}
        for stage_id in range(1, current_stage_id):
            examples = load_stage_dataset(stage_id, data_dir=self.data_dir, use_seed_fallback=False)
            if examples:
                prior_data[stage_id] = examples
        return prior_data

    def sample_replay_examples(self, current_stage_id: int, target_count: int) -> List[CurriculumExample]:
        """Performs stratified sampling across prior stages to collect target_count replay examples."""
        if current_stage_id <= 1 or target_count <= 0:
            return []

        prior_data = self.load_prior_stage_examples(current_stage_id)
        if not prior_data:
            logging.info("No prior stage data available for replay sampling.")
            return []

        total_available = sum(len(exs) for exs in prior_data.values())
        if total_available <= target_count:
            # Return all available prior stage examples
            all_samples = []
            for exs in prior_data.values():
                all_samples.extend(exs)
            return all_samples

        num_prior_stages = len(prior_data)
        per_stage_target = max(1, target_count // num_prior_stages)

        replay_samples = []
        for stage_id, exs in prior_data.items():
            sample_size = min(len(exs), per_stage_target)
            sampled = random.sample(exs, sample_size)
            replay_samples.extend(sampled)

        # Fill any remaining slots if total gathered is less than target_count
        remaining_slots = target_count - len(replay_samples)
        if remaining_slots > 0:
            already_sampled_ids = {id(x) for x in replay_samples}
            all_pool = [x for exs in prior_data.values() for x in exs if id(x) not in already_sampled_ids]
            if all_pool:
                extra = random.sample(all_pool, min(len(all_pool), remaining_slots))
                replay_samples.extend(extra)

        return replay_samples

    def get_blended_dataset(
        self,
        current_stage_id: int,
        current_examples: List[CurriculumExample],
        replay_ratio: float
    ) -> List[CurriculumExample]:
        """Blends current stage dataset with sampled prior stage replay examples based on replay_ratio."""
        if current_stage_id <= 1 or replay_ratio <= 0.0 or not current_examples:
            logging.info(f"Stage {current_stage_id}: No replay blending required.")
            return current_examples

        target_replay_count = max(1, int(len(current_examples) * replay_ratio))
        replay_samples = self.sample_replay_examples(current_stage_id, target_replay_count)

        if not replay_samples:
            return current_examples

        blended = list(current_examples) + replay_samples
        logging.info(
            f"=== Replay Buffer Blended for Stage {current_stage_id} ===\n"
            f"Current Stage Examples: {len(current_examples)}\n"
            f"Replay Samples Added: {len(replay_samples)} (Replay Ratio: {replay_ratio * 100:.1f}%)\n"
            f"Total Blended Dataset Size: {len(blended)}"
        )
        return blended
