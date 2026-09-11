import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_targeted_prompts(failure_report: dict) -> list:
    failure_modes = failure_report.get("failure_modes", [])
    logging.info(f"Generating generation prompts for failures: {failure_modes}")
    prompts = []
    for mode in failure_modes:
        prompts.append(f"Generate training examples specifically targeting: {mode}")
    return prompts

if __name__ == "__main__":
    report = {"failure_modes": ["run_on_sentences", "speaker_drift"]}
    prompts = generate_targeted_prompts(report)
    logging.info(f"Generated {len(prompts)} targeted prompts.")
