"""
scripts/export_merged_model.py
CIRCLE Project: Export Standalone Merged Model

Merges trained stage QLoRA adapter weights back into the base model to create
a standalone, fully unified HuggingFace PyTorch model (.safetensors).

Usage:
  python scripts/export_merged_model.py --stage 5 --output-dir ./export_models/circle_v1
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_standalone_model(stage_id: int = 5, output_dir: str = "./export_models/circle_v1"):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: PyTorch, Transformers, or PEFT is not installed.")
        return False

    checkpoint_path = os.path.join(PROJECT_ROOT, "trainer", "checkpoints", f"stage_{stage_id}")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path '{checkpoint_path}' does not exist.")
        return False

    print(f"1. Loading base model 'gpt2'...")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"2. Loading Stage {stage_id} QLoRA adapter weights from '{checkpoint_path}'...")
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)

    print(f"3. Merging QLoRA adapter into base model weights...")
    merged_model = peft_model.merge_and_unload()

    out_abs_path = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(out_abs_path, exist_ok=True)

    print(f"4. Saving standalone merged model weights to '{out_abs_path}'...")
    merged_model.save_pretrained(out_abs_path)
    tokenizer.save_pretrained(out_abs_path)

    print(f"\n=================================================")
    print(f"  SUCCESSFULLY EXPORTED STANDALONE MODEL")
    print(f"=================================================")
    print(f" Export Path : {out_abs_path}")
    print(f" Files Saved : config.json, model.safetensors, tokenizer files")
    print(f"=================================================\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="CIRCLE Export Merged Standalone Model")
    parser.add_argument("--stage", type=int, default=5, help="Stage checkpoint ID to merge (default: 5)")
    parser.add_argument("--output-dir", type=str, default="./export_models/circle_v1", help="Export target directory")
    args = parser.parse_args()

    export_standalone_model(stage_id=args.stage, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
