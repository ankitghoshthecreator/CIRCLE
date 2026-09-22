import os
import json
import logging
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class GroqCriticAgent:
    """High-capacity evaluation agent connecting to Groq-hosted 70B models for failure mode critique."""
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            logging.warning("GROQ_API_KEY not set or using placeholder. API calls may fail.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            logging.info(f"Initialized GroqCriticAgent with model '{self.model_name}'.")

    def analyze_stage_outputs(
        self,
        stage_id: int,
        stage_name: str,
        objectives: List[str],
        probe_results: List[Dict[str, str]]
    ) -> Dict:
        """Sends model probe outputs to Groq 70B critic and returns diagnostic qualitative critique."""
        if not self.client:
            logging.error("Groq client not initialized due to missing API key.")
            return {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "objectives": objectives,
                "error": "Missing or invalid GROQ_API_KEY",
                "critique_raw": "Mock Critique: Unable to reach Groq API. Please check GROQ_API_KEY in .env."
            }

        logging.info(f"Submitting {len(probe_results)} probe responses for Stage {stage_id} ({stage_name}) to Groq 70B Critic...")

        formatted_probes = ""
        for idx, item in enumerate(probe_results, 1):
            formatted_probes += (
                f"\n--- Probe {idx} ---\n"
                f"Prompt: {item.get('prompt')}\n"
                f"Model Output: {item.get('output')}\n"
            )

        system_prompt = (
            "You are CIRCLE's expert developmental NLP critic agent. "
            "Your task is to analyze the performance of a language model checkpoint trained on a specific curriculum stage.\n"
            "Evaluate whether the model output demonstrates required stage objectives and list specific, categorized failure modes.\n"
            "Structure your response clearly with:\n"
            "1. LINGUISTIC COMPETENCE ASSESSMENT\n"
            "2. IDENTIFIED FAILURE MODES (e.g., run_on_sentences, entity_dropping, subject_verb_mismatch, speaker_drift)\n"
            "3. RECOMMENDATIONS FOR SYNTHETIC DATA GENERATION"
        )

        user_content = (
            f"Curriculum Stage {stage_id}: {stage_name}\n"
            f"Target Objectives: {', '.join(objectives)}\n\n"
            f"Probe Outputs from Checkpoint:\n{formatted_probes}\n\n"
            f"Provide your detailed diagnostic critique below:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
                max_tokens=800
            )

            critique_text = response.choices[0].message.content
            logging.info(f"Received critique from Groq 70B Critic for Stage {stage_id}.")

            return {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "objectives": objectives,
                "model_name": self.model_name,
                "critique_raw": critique_text,
                "probe_results": probe_results
            }
        except Exception as e:
            logging.error(f"Error calling Groq API: {e}")
            return {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "objectives": objectives,
                "error": str(e),
                "critique_raw": f"Error generating critique: {e}",
                "probe_results": probe_results
            }

if __name__ == "__main__":
    agent = GroqCriticAgent()
    sample_probes = [
        {"prompt": "The primary purpose of a language model is to", "output": "generate text next token prediction."}
    ]
    report = agent.analyze_stage_outputs(1, "Base English", ["fluency", "syntax"], sample_probes)
    print("--- Diagnostic Critique Report ---")
    print(report.get("critique_raw"))
