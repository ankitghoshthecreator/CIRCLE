import os
import sys
import argparse
import logging
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from trainer.model_loader import load_qlora_model_and_tokenizer, load_stage_adapter
from trainer.curriculum.stage_config import STAGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SimpleTextDataset(Dataset):
    """Dataset wrapper for tokenized text passages."""
    def __init__(self, texts: list, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["input_ids"][idx].clone()
        }

def get_default_stage_samples(stage_id: int) -> list:
    """Returns seed synthetic training samples for dry-run testing across curriculum stages."""
    samples_map = {
        1: [
            "The quick brown fox jumps over the lazy dog.",
            "Language models learn patterns from text sequentially.",
            "PyTorch provides flexible tools for deep learning."
        ],
        2: [
            "Passage: Artificial intelligence is transforming technology. Summary: AI transforms tech.",
            "Passage: QLoRA enables fine-tuning models on low VRAM GPUs. Summary: QLoRA saves GPU VRAM."
        ],
        3: [
            "Correct: She writes code every day. Incorrect: She write code every day.",
            "Correct: They are building a neural network. Incorrect: They is building a neural network."
        ],
        4: [
            "Clause test: Although it was raining, we decided to continue the experiment.",
            "Punctuation test: Hello, user! How can I help you today?"
        ],
        5: [
            "User: What is CIRCLE? Assistant: CIRCLE is a closed-loop curriculum training engine.",
            "User: How does critique work? Assistant: The 70B critic analyzes outputs to commission data."
        ]
    }
    return samples_map.get(stage_id, samples_map[1])

def train_stage(
    stage_id: int = 1,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    lr: float = 2e-4,
    checkpoint_dir: str = "./trainer/checkpoints",
    data_texts: list = None
):
    if stage_id not in STAGES:
        raise ValueError(f"Invalid stage_id: {stage_id}. Choose between 1 and 5.")

    stage = STAGES[stage_id]
    logging.info(f"=== Starting QLoRA Training for Stage {stage.stage_id}: {stage.name} ===")
    logging.info(f"Objectives: {stage.objectives}")

    # Step 1: Load 4-bit quantized base model and tokenizer
    model, tokenizer = load_qlora_model_and_tokenizer(model_name_or_path="gpt2", is_trainable=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 2: Prepare dataset & dataloader
    texts = data_texts if data_texts else get_default_stage_samples(stage_id)
    dataset = SimpleTextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 3: Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = (len(dataloader) // grad_accum_steps + 1) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=total_steps)

    # Step 4: Training Loop
    model.train()
    for epoch in range(epochs):
        logging.info(f"--- Epoch {epoch + 1}/{epochs} ---")
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            total_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            logging.info(f"Step {step + 1}/{len(dataloader)} - Loss: {loss.item() * grad_accum_steps:.4f}")

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1} Complete - Average Loss: {avg_loss:.4f}")

    # Step 5: Save QLoRA Adapter Checkpoint
    stage_checkpoint_path = os.path.join(checkpoint_dir, f"stage_{stage_id}")
    os.makedirs(stage_checkpoint_path, exist_ok=True)
    logging.info(f"Saving stage {stage_id} QLoRA adapter checkpoint to '{stage_checkpoint_path}'...")
    model.save_pretrained(stage_checkpoint_path)
    tokenizer.save_pretrained(stage_checkpoint_path)
    logging.info(f"Stage {stage_id} training pass completed successfully.")

    return stage_checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIRCLE Base PyTorch QLoRA Trainer")
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage ID (1-5)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per step")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="./trainer/checkpoints", help="Path to checkpoint directory")
    args = parser.parse_args()

    train_stage(
        stage_id=args.stage,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )
