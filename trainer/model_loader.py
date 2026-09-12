import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_bnb_config():
    """Returns 4-bit quantization config optimized for low VRAM targets (4GB)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

def load_qlora_model_and_tokenizer(model_name_or_path: str = "gpt2", is_trainable: bool = True):
    """Loads tokenizer and base model configured with QLoRA 4-bit quantization and LoRA adapters."""
    logging.info(f"Loading tokenizer for '{model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    bnb_config = get_bnb_config() if torch.cuda.is_available() else None

    logging.info(f"Loading base model '{model_name_or_path}' (4-bit quant={torch.cuda.is_available()})...")
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    if is_trainable:
        if torch.cuda.is_available():
            model = prepare_model_for_kbit_training(model)

        # Apply QLoRA adapters to linear projection layers
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj", "c_fc"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer

def load_stage_adapter(base_model, adapter_path: str):
    """Loads a trained stage-specific QLoRA adapter onto the base model."""
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter checkpoint path not found: {adapter_path}")
    
    logging.info(f"Loading QLoRA adapter from '{adapter_path}'...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model
