import json
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch
from tqdm import tqdm
from utils import create_folder
from config import GSM_SYMBOLIC_MODELS


class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, prompt_length):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        if (
            input_ids.shape[1] <= self.prompt_length
        ):  # Skip if we haven't generated beyond prompt
            return False

        # Only decode the newly generated text (everything after the prompt)
        generated_text = self.tokenizer.decode(input_ids[0][self.prompt_length :])
        return any(keyword in generated_text for keyword in self.keywords)


def load_examples(file_path="GSM_symbolic.jsonl"):
    """Load examples from the JSONL file."""
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def load_few_shot_examples(file_path="gsm_few_shot.json"):
    """Load few-shot examples from the JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    return examples


def group_examples_by_original_id(examples):
    """Group examples by their original ID."""
    grouped_examples = {}
    for example in examples:
        if example["original_id"] not in grouped_examples:
            grouped_examples[example["original_id"]] = []
        grouped_examples[example["original_id"]].append(example)
    return grouped_examples


def format_prompt(shot_examples, new_question):
    """Format the prompt with few-shot examples and the new question."""
    prompt = ""
    # Add shot examples
    for example in shot_examples:
        prompt += f"Question: {example['question']}\n"
        prompt += f"Answer: {example['answer']}\n\n"

    # Add the new question
    prompt += f"Question: {new_question}\n"
    prompt += "Answer: Let's think step by step."
    return prompt


def generate_response(model, tokenizer, prompt, apply_chat_template=False):
    """Generate a response using the model."""
    if apply_chat_template:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"].to(model.device),
            max_new_tokens=1000,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=[
                StoppingCriteriaList(
                    [
                        KeywordStoppingCriteria(
                            ["Question:"],
                            tokenizer,
                            prompt_length=inputs["input_ids"].shape[1],
                        )
                    ]
                )
            ],
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip()
    if answer.endswith("Question:"):
        answer = answer[: -len("Question:")].strip()
    return answer


def main(
    model_id,
    apply_chat_template,
    num_questions,
    num_variants,
):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model_name = GSM_SYMBOLIC_MODELS[model_id]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    few_shot_examples = load_few_shot_examples()
    examples = load_examples()
    grouped_examples = group_examples_by_original_id(examples)
    all_original_ids = random.sample(list(grouped_examples.keys()), num_questions)

    all_results = {}
    for question_id in tqdm(all_original_ids):
        questions_variants = random.sample(grouped_examples[question_id], num_variants)
        original_question = grouped_examples[question_id][0]

        # First generate response for the original question
        prompt = format_prompt(few_shot_examples, original_question["question"])
        response = generate_response(model, tokenizer, prompt, apply_chat_template)
        results = [
            {
                "prompt": prompt,
                "new_question": original_question["question"],
                "new_question_answer": original_question["answer"],
                "response": response,
            }
        ]

        # Then generate response for new questions
        for new_question in questions_variants:
            prompt = format_prompt(few_shot_examples, new_question["question"])
            response = generate_response(model, tokenizer, prompt)
            results.append(
                {
                    # "prompt": prompt,
                    "new_question": new_question["question"],
                    "new_question_answer": new_question["answer"],
                    "response": response,
                }
            )

        all_results[question_id] = results

    return all_results


if __name__ == "__main__":
    create_folder("results/gsm_symbolic_results")

    num_questions = 100
    num_variants = 50
    for model_id in GSM_SYMBOLIC_MODELS:
        apply_chat_template = model_id in ["qwen_1_5B_it", "gemma_9B_it"]
        res = main(model_id, apply_chat_template, num_questions, num_variants)
        with open(
            f"results/gsm_symbolic_results/{model_id}_nq{num_questions}_nv{num_variants}.json",
            "w",
        ) as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
