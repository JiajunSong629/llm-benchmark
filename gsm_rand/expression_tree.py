import json

from eval import extract_answer
from data import TaskTemplate, task_templates
from main import magician_cards_template, cleaner_work_template

TASK_TEMPLATES = {}
for t in task_templates:
    t_name = "_".join([c.lower() for c in t.type_name.split()])
    TASK_TEMPLATES[t_name] = t
TASK_TEMPLATES["magician_cards"] = magician_cards_template
TASK_TEMPLATES["cleaner_work"] = cleaner_work_template


class ExpressionTreeParser:
    def __init__(self, type_name):
        self.type_name = type_name
        self.template = TASK_TEMPLATES[type_name]

    def _extract_final_question(self, prompt):
        return "Question: " + prompt.split("Question:")[-1].strip()

    def parse(self, question, deduction, full_prompt=True):
        if full_prompt:
            question = self._extract_final_question(question)
        else:
            question = question

        # compare question with template to determine the variables
        variable_mapping = self._match(question, self.template.question_template)

        # parse deduction to get the expression tree
        formula, status = self._unwrap_deduction(deduction)

        reverse_mapping = {}
        for k, v in variable_mapping.items():
            if v in reverse_mapping:
                reverse_mapping[v].append(k)
            else:
                reverse_mapping[v] = [k]

        return formula, status

    def _match(self, question, template) -> dict:
        """
        Match the question with the template to determine the variable mapping
        """
        real_words = question.split()
        template_words = template.split()

        mapping = {}
        for real_word, template_word in zip(real_words, template_words):
            if template_word.startswith("{") and template_word.endswith("}"):
                var_name = template_word[1:-1]
                mapping[var_name] = real_word

            elif "{" in template_word and "}" in template_word:
                var_start = template_word.index("{")
                var_end = template_word.index("}") + 1
                var_name = template_word[var_start + 1 : var_end - 1]

                # Get the template pattern (e.g., "{rate}kg")
                pattern_before = template_word[:var_start]
                pattern_after = template_word[var_end:]

                # Remove the same patterns from real word to get the value
                real_value = real_word
                if pattern_before and real_word.startswith(pattern_before):
                    real_value = real_value[len(pattern_before) :]
                if pattern_after and real_value.endswith(pattern_after):
                    real_value = real_value[: -len(pattern_after)]

                mapping[var_name] = real_value
        return mapping

    def _unwrap_deduction(self, deduction):
        """
        Unwrap the deduction to get the expression tree
        """

        def is_number(string):
            try:
                float(string)
                return True
            except ValueError:
                return False

        import re

        # Extract all calculations and their results
        calculations = re.findall(r"<<(.*?)>>", deduction)
        final_answer = extract_answer(deduction)

        # Store expressions for each result
        result_to_expr = {}

        # Process calculations in order
        for calc in calculations:
            expr, result = calc.split("=")
            expr = expr.strip()
            result = result.replace("%", "").strip()

            # Split expression into parts (numbers and operators)
            parts = re.split(r"(\d+\.?\d*|\.\d+)", expr)
            new_expr_parts = []

            for part in parts:
                if (
                    part.strip()
                    and is_number(part.strip())
                    and part.strip() in result_to_expr
                ):
                    # If this number appears as a previous result, replace it with its expression
                    new_expr_parts.append(f"({result_to_expr[part.strip()]})")
                else:
                    new_expr_parts.append(part)

            # Store the complete expression for this result
            result_to_expr[result] = "".join(new_expr_parts)

        if final_answer in result_to_expr:
            final_expr = result_to_expr[final_answer]
            return f"{final_expr} = {final_answer}", True

        # TODO: hack for percentage
        if str(float(final_answer) / 100) in result_to_expr:
            final_expr = result_to_expr[str(float(final_answer) / 100)]
            return f"{final_expr} = {final_answer}", True

        return max(result_to_expr.values(), key=len), False


def main(model, question):
    import json
    import os

    os.makedirs("data/expression_tree", exist_ok=True)

    eval_results = json.load(open(f"data/results/{question}_{model}.json"))
    etparser = ExpressionTreeParser(question)

    results = []
    parsed_count = 0
    for eval_result in eval_results:
        try:
            expression_tree, status = etparser.parse(
                question=eval_result["prompt"],
                deduction=eval_result["response"],
                full_prompt=True,
            )
            results.append(
                {
                    "prompt": eval_result["prompt"],
                    "response": eval_result["response"],
                    "expression_tree": expression_tree,
                    "success": status,
                }
            )
            parsed_count += status
        except Exception as e:
            results.append(
                {
                    "prompt": eval_result["prompt"],
                    "response": eval_result["response"],
                    "expression_tree": str(e),
                    "success": False,
                }
            )

    print(f"{model} {question} Parsed {parsed_count} / {len(eval_results)}")
    json.dump(
        results,
        open(f"data/expression_tree/{question}_{model}.json", "w"),
        indent=4,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    from config import EVAL_MODELS, EVAL_QUESTION_NAMES

    for model in EVAL_MODELS:
        for question in EVAL_QUESTION_NAMES:
            main(model, question)
    # d = "Let's think step by step. First, find the number of floors the worker needs to clean per day: 240 / 10=<<240/10=24>>24.\nThen find the number of minutes the worker needs to clean per day: 24 floors * 20 minutes/floor = <<24*20=480>>480 minutes.\nThen find the number of hours the worker needs to clean per day: 480 minutes / 60 minutes/hour = <<480/60=8>>8 hours.\nThen find the percentage of the day the worker spends cleaning floors: 8 hours / 10 hours = <<8/10=0.8>>0.8 or 80%.\n#### 80"
    # parser = ExpressionTreeParser("cleaner_work")
    # print(parser._unwrap_deduction(d))
