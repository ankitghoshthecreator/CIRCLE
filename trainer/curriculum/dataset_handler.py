import os
import json
import logging
import torch
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CurriculumExample(BaseModel):
    """Pydantic schema representing a single curriculum training sample."""
    input_text: str = Field(..., description="Prompt or passage text")
    target_text: str = Field(..., description="Target completion or expected output text")
    stage_id: int = Field(..., description="Curriculum stage ID (1-5)")
    source: str = Field(default="seed", description="Source of data: 'seed' or 'synthetic'")
    metadata: Dict = Field(default_factory=dict, description="Optional metadata or critic tags")

class CurriculumDataset(Dataset):
    """PyTorch Dataset for formatting and tokenizing curriculum instruction/response pairs."""
    def __init__(self, examples: List[CurriculumExample], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if ex.input_text and ex.target_text:
            formatted_text = f"{ex.input_text}\n{ex.target_text}"
        else:
            formatted_text = ex.input_text or ex.target_text

        encodings = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

def load_stage_config(stage_id: int) -> dict:
    """Loads the JSON configuration file for a given stage_id."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs", f"stage_{stage_id}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Curriculum config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_dataset_dir(base_data_dir: str = "./data") -> str:
    os.makedirs(base_data_dir, exist_ok=True)
    return base_data_dir

def save_stage_dataset(stage_id: int, examples: List[CurriculumExample], data_dir: str = "./data"):
    """Saves a list of CurriculumExample items to JSON file under data_dir/stage_<ID>/dataset.json."""
    stage_dir = os.path.join(data_dir, f"stage_{stage_id}")
    os.makedirs(stage_dir, exist_ok=True)
    file_path = os.path.join(stage_dir, "dataset.json")

    serialized = [ex.model_dump() for ex in examples]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)

    logging.info(f"Saved {len(examples)} examples for Stage {stage_id} to '{file_path}'.")

def load_stage_dataset(stage_id: int, data_dir: str = "./data") -> List[CurriculumExample]:
    """Loads dataset for stage_id. If missing, returns default seed examples for that stage."""
    file_path = os.path.join(data_dir, f"stage_{stage_id}", "dataset.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            return [CurriculumExample(**item) for item in raw_data]

    logging.info(f"No existing dataset file found at '{file_path}'. Loading default seed dataset...")
    return get_default_seed_examples(stage_id)

def get_default_seed_examples(stage_id: int) -> List[CurriculumExample]:
    """Returns fallback initial seed examples for stage_id."""
    seeds_map = {
        1: [
            CurriculumExample(input_text="The model learns fluency through next token prediction.", target_text="", stage_id=1, source="seed"),
            CurriculumExample(input_text="PyTorch enables efficient neural network development.", target_text="", stage_id=1, source="seed")
        ],
        2: [
            CurriculumExample(input_text="Passage: QLoRA quantizes model weights to 4-bit precision.", target_text="Summary: QLoRA uses 4-bit quantization.", stage_id=2, source="seed")
        ],
        3: [
            CurriculumExample(input_text="Incorrect: She write clean code.", target_text="Correct: She writes clean code.", stage_id=3, source="seed")
        ],
        4: [
            CurriculumExample(input_text="Unpunctated: although it was late we finished the project", target_text="Punctated: Although it was late, we finished the project.", stage_id=4, source="seed")
        ],
        5: [
            CurriculumExample(input_text="User: What is QLoRA?\nAssistant:", target_text="QLoRA is a parameter-efficient fine-tuning method.", stage_id=5, source="seed")
        ]
    }
    return seeds_map.get(stage_id, seeds_map[1])
