"""
scripts/sample_inference.py
CIRCLE Part 16: Sample Output Inference Script

Generate sample text completions from trained stage QLoRA adapter checkpoints or base models.

Usage:
  python scripts/sample_inference.py --stage 1 --prompt "Machine learning algorithms process data in order to"
  python scripts/sample_inference.py --stage 4 --prompt "although it was raining we went for a run"
  python scripts/sample_inference.py --stage 5 --prompt "User: What is QLoRA?\nAssistant:"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_sample(stage_id: int, prompt: str, checkpoint_dir: str = "./trainer/checkpoints") -> str:
    try:
        import torch
        from trainer.model_loader import load_qlora_model_and_tokenizer, load_stage_adapter
    except ImportError:
        return "PyTorch or HuggingFace transformers is not installed on this environment."

    adapter_path = os.path.join(PROJECT_ROOT, checkpoint_dir, f"stage_{stage_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(adapter_path):
        print(f"Loading trained Stage {stage_id} QLoRA checkpoint from '{adapter_path}'...")
        base_model, tokenizer = load_qlora_model_and_tokenizer("gpt2", is_trainable=False)
        model = load_stage_adapter(base_model, adapter_path)
    else:
        print(f"No checkpoint found at '{adapter_path}'. Evaluating base model...")
        model, tokenizer = load_qlora_model_and_tokenizer("gpt2", is_trainable=False)

    model.eval()
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="CIRCLE Sample Output Inference")
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage ID (1-5)")
    parser.add_argument("--prompt", type=str, default="The primary purpose of a language model is to", help="Test prompt text")
    parser.add_argument("--checkpoint-dir", type=str, default="./trainer/checkpoints", help="Checkpoints directory")
    args = parser.parse_args()

    completion = generate_sample(args.stage, args.prompt, args.checkpoint_dir)
    print("\n=================================================")
    print(f"  Stage {args.stage} Model Output Sample")
    print("=================================================")
    print(f" PROMPT    : {args.prompt}")
    print(f" COMPLETION: {completion}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
