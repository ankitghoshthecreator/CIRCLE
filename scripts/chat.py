"""
scripts/chat.py
CIRCLE Interactive Chat CLI

Allows interactive conversation and text generation with your trained CIRCLE GPT-2 QLoRA model.

Usage:
  python scripts/chat.py --checkpoint trainer/checkpoints/checkpoint_best
  python scripts/chat.py --checkpoint trainer/checkpoints/stage_1
  python scripts/chat.py --mock
"""

import os
import sys
import argparse
import logging
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CIRCLE-Chat")


def _find_adapter_dir(checkpoint_path: str) -> Optional[str]:
    """Find a valid adapter directory, checking target path and stage folders."""
    candidates = [
        checkpoint_path,
        os.path.join(PROJECT_ROOT, "trainer", "checkpoints", "stage_5"),
        os.path.join(PROJECT_ROOT, "trainer", "checkpoints", "stage_4"),
        os.path.join(PROJECT_ROOT, "trainer", "checkpoints", "stage_3"),
        os.path.join(PROJECT_ROOT, "trainer", "checkpoints", "stage_2"),
        os.path.join(PROJECT_ROOT, "trainer", "checkpoints", "stage_1"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            if (
                os.path.exists(os.path.join(c, "adapter_config.json"))
                or os.path.exists(os.path.join(c, "adapter_model.bin"))
                or os.path.exists(os.path.join(c, "adapter_model.safetensors"))
            ):
                return c
    return None


def load_inference_model(checkpoint_path: str, base_model_name: str = "gpt2"):
    """Load base model + trained PEFT/QLoRA adapter for inference."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    logger.info(f"Loading tokenizer from '{base_model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading base model '{base_model_name}' on device: {device}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    adapter_dir = _find_adapter_dir(checkpoint_path)
    if adapter_dir:
        logger.info(f"Loading QLoRA adapter from '{adapter_dir}'...")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        logger.warning(
            f"No QLoRA adapter found at '{checkpoint_path}' or stage directories. Using base model '{base_model_name}'."
        )
        model = base_model

    model.eval()
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate response text for user prompt using chat format."""
    import torch

    formatted_prompt = f"User: {prompt}\nAssistant:" if not prompt.startswith("User:") else prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Stop generation if model tries to generate a new turn
    if "User:" in response:
        response = response.split("User:")[0].strip()
    if "\n\n" in response:
        response = response.split("\n\n")[0].strip()

    return response


def run_interactive_chat(checkpoint_path: str, max_new_tokens: int, temp: float, mock: bool):
    """Run interactive terminal chat loop."""
    print("\n" + "=" * 65)
    print("           🤖 CIRCLE Interactive LLM Terminal Chat 🤖")
    print("=" * 65)
    print(f" Checkpoint : {checkpoint_path}")
    print(f" Max Tokens : {max_new_tokens}  |  Temperature: {temp}")
    print(" Commands   : Type 'exit', 'quit', or 'q' to end chat.")
    print("=" * 65 + "\n")

    if mock:
        print("[System] Running in MOCK Mode (simulated LLM generation).\n")
        while True:
            try:
                user_input = input("\nUser > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break
                # Simulated response
                mock_resp = f"I am your CIRCLE LLM. In response to '{user_input[:40]}...', I understand and can generate fluent English text based on Stage 5 curriculum training!"
                print(f"\nCIRCLE LLM > {mock_resp}")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break
        return

    # Real model inference
    try:
        model, tokenizer, device = load_inference_model(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print("\n[Error] Could not load model checkpoint. Fallback to mock mode by passing --mock.")
        return

    print("\n✅ Model loaded successfully! Start chatting below:\n")

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=user_input,
                max_new_tokens=max_new_tokens,
                temperature=temp,
            )
            print(f"\nCIRCLE LLM > {response if response else '(No output generated)'}")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="CIRCLE Interactive LLM Chat CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./trainer/checkpoints/checkpoint_best",
        help="Path to trained QLoRA checkpoint directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.1 = deterministic, 1.0 = creative)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Run simulated chat loop (no model load needed)",
    )
    args = parser.parse_args()

    ckpt_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.checkpoint))
    run_interactive_chat(
        checkpoint_path=ckpt_path,
        max_new_tokens=args.max_tokens,
        temp=args.temperature,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
