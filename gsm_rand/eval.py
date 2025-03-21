import re
import json
import sys

sys.path.append("..")
from gsm_symbolic.gsm_symbolic import generate_response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion: str) -> str:
    completion = completion.replace("$", "")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def summarize_results(results):
    correct = sum(result["correct"] for result in results)
    accuracy = correct / len(results)

    correct_by_step = {}
    for result in results:
        correct_by_step[result["deduction_step"]] = (
            correct_by_step.get(result["deduction_step"], 0) + result["correct"]
        )

    return accuracy, correct_by_step


def main(model_id, question_name):
    if os.path.exists(f"data/results/{question_name}_{model_id}.json"):
        # print(f"Skipping {question_name}_{model_id} because it already exists")
        return

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model_name = EVAL_MODELS[model_id]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(f"data/questions/{question_name}_questions.json", "r") as f:
        questions = json.load(f)

    results = []
    for question in questions:
        print(question["sample_id"], " ---> ", question["deduction_step"])
        response = generate_response(model, tokenizer, question["prompt"])
        results.append(
            {
                "sample_id": question["sample_id"],
                "deduction_step": question["deduction_step"],
                "prompt": question["prompt"],
                "answer": question["answer"],
                "response": response,
                "response_answer": extract_answer(response),
                "correct": question["answer"] == extract_answer(response),
            }
        )
        print(question["answer"], " ---> ", extract_answer(response), end="\n\n")

    with open(f"data/results/{question_name}_{model_id}.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def summarize_main(model_id, question_name):
    with open(f"data/results/{question_name}_{model_id}.json", "r") as f:
        results = json.load(f)
    accuracy, correct_by_step = summarize_results(results)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Correct by step: {correct_by_step}")


if __name__ == "__main__":
    import os
    from config import EVAL_MODELS, EVAL_QUESTION_NAMES

    os.makedirs("data/results", exist_ok=True)

    for model_id in EVAL_MODELS:    
        for question_name in EVAL_QUESTION_NAMES:
            main(model_id, question_name)
            print(question_name, model_id)
            summarize_main(model_id, question_name)
