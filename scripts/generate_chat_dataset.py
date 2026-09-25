"""
scripts/generate_chat_dataset.py
CIRCLE Diversity Chat Dataset Generator via Ollama (Qwen 2.5)

Generates 1,000 high-variety instruction & response pairs covering:
  - Greetings & Friendly Chat (150)
  - Technical & Software QA (Docker, Python, Linux, K8s) (300)
  - Science & General Knowledge Explanations (250)
  - Problem Solving & Logic (150)
  - Assistance & Helpful Dialogue (150)

Saves output directly to data/stage_5.json for QLoRA fine-tuning.
"""

import os
import sys
import json
import random
import logging
import requests
import argparse
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ChatDataGen")


CATEGORIES = {
    "greetings": [
        "Friendly greeting and casual inquiry",
        "Polite introduction and offering assistance",
        "Asking how someone is doing and responding warmly",
        "Good morning / good evening polite dialogue",
    ],
    "technical": [
        "Explaining Docker containers and virtualization",
        "Python programming concepts, functions, and list comprehensions",
        "Kubernetes pods, services, and horizontal pod autoscaling",
        "Linux CLI commands, permissions, and bash scripting",
        "REST API design, JSON payloads, and HTTP status codes",
        "Database indexing, SQL joins, and transactions",
        "Git branching, merging, and version control best practices",
    ],
    "science_general": [
        "Explaining photosynthesis and plant energy conversion",
        "How the solar system and gravity work",
        "Basics of artificial intelligence and machine learning",
        "Human anatomy, respiratory, and circulatory system basics",
        "Climate change, renewable energy, and environmental impact",
    ],
    "logic_reasoning": [
        "Step-by-step math word problems and solutions",
        "Deductive logic and conditional reasoning puzzles",
        "Comparing pros and cons of cloud computing vs on-premise",
        "Algorithmic time complexity (Big-O notation) breakdown",
    ],
    "assistance": [
        "Summarizing complex text into 3 bullet points",
        "Writing a professional email for a project status update",
        "Drafting a concise resume summary for a software engineer",
        "Explaining how to debug a broken program systematically",
    ],
}


PROMPT_TEMPLATE = """You are a dataset creation assistant. Generate a high-quality, diverse dataset of {count} distinct conversational question and answer pairs for training an AI chat assistant on the topic: '{subtopic}'.

Each item MUST be formatted in valid JSON with exactly two keys:
  "user_prompt": "<clear user question or greeting>",
  "assistant_response": "<accurate, direct, helpful answer in 2-4 sentences>"

Output ONLY a raw JSON list of objects like this:
[
  {{"user_prompt": "Hello! How are you doing today?", "assistant_response": "Hello! I am doing great and ready to help you. How can I assist you today?"}},
  {{"user_prompt": "What is Docker?", "assistant_response": "Docker is a containerization platform that packages applications and their dependencies together. It ensures software runs consistently across different computing environments."}}
]

Do not include markdown code block ticks (```json) or extra conversational commentary. Output valid JSON list only.
"""


def generate_batch_via_ollama(
    endpoint_url: str,
    model_name: str,
    subtopic: str,
    count: int = 10,
) -> List[Dict[str, str]]:
    """Query Ollama API to generate a batch of user_prompt / assistant_response pairs."""
    prompt = PROMPT_TEMPLATE.format(count=count, subtopic=subtopic)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.95,
        },
    }

    url = f"{endpoint_url.rstrip('/')}/api/generate"
    try:
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            raw_text = resp.json().get("response", "").strip()
            # Clean up potential markdown formatting
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                valid_pairs = []
                for item in parsed:
                    if isinstance(item, dict) and "user_prompt" in item and "assistant_response" in item:
                        u = str(item["user_prompt"]).strip()
                        a = str(item["assistant_response"]).strip()
                        if u and a:
                            valid_pairs.append({"user_prompt": u, "assistant_response": a})
                return valid_pairs
    except Exception as e:
        logger.warning(f"Ollama generation failed for subtopic '{subtopic}': {e}")

    return []


def generate_full_chat_dataset(
    target_count: int = 1000,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5-coder:7b",
    output_file: str = "./data/stage_5.json",
) -> int:
    """Generate target_count diverse QA pairs and save to Stage 5 curriculum file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    all_examples: List[Dict[str, Any]] = []

    # Flatten all subtopics
    subtopics = []
    for cat, topic_list in CATEGORIES.items():
        subtopics.extend(topic_list)

    batch_size = 10
    total_generated = 0
    attempts = 0
    max_attempts = (target_count // batch_size) * 3

    logger.info(
        f"🚀 Starting generation of {target_count} diverse chat pairs via Ollama ({ollama_model})..."
    )

    while total_generated < target_count and attempts < max_attempts:
        subtopic = random.choice(subtopics)
        attempts += 1
        logger.info(
            f"Generating batch {attempts} | Total pairs so far: {total_generated}/{target_count} | Topic: {subtopic[:35]}..."
        )

        pairs = generate_batch_via_ollama(
            endpoint_url=ollama_url,
            model_name=ollama_model,
            subtopic=subtopic,
            count=batch_size,
        )

        for p in pairs:
            # Standard CIRCLE CurriculumExample structure with chat formatting
            example = {
                "stage_id": 5,
                "input_text": f"User: {p['user_prompt']}\nAssistant:",
                "target_text": f" {p['assistant_response']}",
            }
            all_examples.append(example)
            total_generated += 1
            if total_generated >= target_count:
                break

    # Save dataset to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    logger.info(
        f"✅ Successfully generated and saved {len(all_examples)} high-diversity chat examples to '{output_file}'!"
    )
    return len(all_examples)


def main():
    parser = argparse.ArgumentParser(description="Generate 1000 Chat Dataset Examples via Ollama")
    parser.add_argument("--count", type=int, default=1000, help="Total dataset pairs to generate")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GENERATOR_MODEL", "qwen2.5-coder:7b"),
        help="Ollama model name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/stage_5.json",
        help="Output dataset file path",
    )
    args = parser.parse_args()

    count = generate_full_chat_dataset(
        target_count=args.count,
        ollama_url=args.ollama_url,
        ollama_model=args.model,
        output_file=os.path.join(PROJECT_ROOT, args.output),
    )
    print(f"\nCompleted! Total examples generated: {count}")


if __name__ == "__main__":
    main()
