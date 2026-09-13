import os
import sys
import json
import argparse
import logging
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.model_loader import load_qlora_model_and_tokenizer, load_stage_adapter
from trainer.curriculum.dataset_handler import load_stage_config
from eval.critique import GroqCriticAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_stage_checkpoint(
    stage_id: int = 1,
    checkpoint_dir: str = "./trainer/checkpoints",
    output_dir: str = "./eval/reports"
) -> dict:
    stage_config = load_stage_config(stage_id)
    stage_name = stage_config["name"]
    objectives = stage_config["objectives"]
    probe_prompts = stage_config.get("probe_prompts", [])

    logging.info(f"=== Starting Probe Evaluation for Stage {stage_id}: {stage_name} ===")
    logging.info(f"Number of probe prompts: {len(probe_prompts)}")

    adapter_path = os.path.join(checkpoint_dir, f"stage_{stage_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load model & stage adapter
    if os.path.exists(adapter_path):
        logging.info(f"Loading trained adapter checkpoint from '{adapter_path}'...")
        base_model, tokenizer = load_qlora_model_and_tokenizer("gpt2", is_trainable=False)
        model = load_stage_adapter(base_model, adapter_path)
    else:
        logging.info(f"Adapter checkpoint not found at '{adapter_path}'. Evaluating base model...")
        model, tokenizer = load_qlora_model_and_tokenizer("gpt2", is_trainable=False)

    model.eval()

    # Step 2: Run inference on probe prompts
    probe_results = []
    for idx, prompt in enumerate(probe_prompts, 1):
        logging.info(f"Generating completion for Probe {idx}/{len(probe_prompts)}...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        probe_results.append({
            "prompt": prompt,
            "output": generated_text
        })

    # Step 3: Send probe results to Groq 70B Critic
    critic = GroqCriticAgent()
    critique_report = critic.analyze_stage_outputs(
        stage_id=stage_id,
        stage_name=stage_name,
        objectives=objectives,
        probe_results=probe_results
    )

    # Step 4: Save critique report
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f"stage_{stage_id}_critique.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(critique_report, f, indent=2)

    logging.info(f"Evaluation finished. Critique report saved to '{report_file}'.")
    return critique_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIRCLE Stage Probe Evaluator")
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage ID to evaluate (1-5)")
    parser.add_argument("--checkpoint-dir", type=str, default="./trainer/checkpoints", help="Path to checkpoints directory")
    parser.add_argument("--output-dir", type=str, default="./eval/reports", help="Path to report output directory")
    args = parser.parse_args()

    evaluate_stage_checkpoint(
        stage_id=args.stage,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir
    )
