from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import OpenAI
import json
import dotenv
import os
import hashlib

dotenv.load_dotenv(override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
API_BASE_MODEL = os.getenv("API_BASE_MODEL")


@dataclass
class ComputationNode:
    id: str
    name: str
    value: float
    operation: Optional[str] = None
    operands: Optional[List[str]] = None
    is_leaf: bool = False  # New field to mark leaf nodes

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "operation": self.operation,
            "operands": self.operands,
            "is_leaf": self.is_leaf,
        }


def call_llm(
    prompt: str,
    system_prompt: str = None,
    response_format: str = None,
) -> str:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=API_BASE_URL,
    )

    response = client.chat.completions.create(
        model=API_BASE_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": response_format},
    )

    print("RAW RESPONSE", response.choices[0].message.content)
    return json.loads(response.choices[0].message.content)


class IncrementalParser:
    def __init__(self, stop_word: str = "."):
        self.stop_word = stop_word
        self.computation_graph = []
        self.leaf_nodes = []  # Store leaf nodes separately

    def _one_step_parse(self, question: str, one_step_deduction: str) -> str:
        """
        Apply parsing on the one_step_deduction, where the deduction is assumed to be a single sentence to avoid wrong parsing. During the parsing, the variable mapping and computation graph will be updated to include the new intermediate variables.

        Args:
            one_step_deduction: a single sentence deduction

        Returns:
            a string of the parsed deduction
        """
        prompt = f"""
Parse the new deduction into computation nodes. Each node must follow this schema:
{{
    "id": "abc12345",  // Exactly 8 chars using letters and numbers
    "name": "descriptive_name",
    "value": float,  // Use the exact values from the deduction
    "operation": "operation_type_or_null",
    "operands": ["node_id1", "node_id2"],  // Must be node IDs, not values
    "is_leaf": bool  // Mark leaf nodes as true, otherwise false
}}

Question: {question}
Current graph: {self.computation_graph if self.computation_graph else "None"}
New deduction: {one_step_deduction}

IMPORTANT:
- Extract values EXACTLY as they appear in the deduction
- Node IDs must be exactly 8 characters using letters (a-z) and numbers (0-9)
- Example IDs: "calc1234", "rate5678", "step9abc"
- Do NOT recalculate or validate any values
- Do NOT correct mathematical errors
- Simply parse and structure the information given

Rules:
1. Only parse the new deduction, don't solve or verify the math
2. Only return new nodes not already in the graph
3. Operands must reference existing node IDs or new node IDs
4. Return valid JSON only - either a single node object or array of nodes

Example operands format:
❌ "operands": [4, 2]           // Wrong: using values
✅ "operands": ["calc1234"]    // Correct: using 8-char node IDs
"""

        system_prompt = "You are a mechanical parser that extracts computation nodes from text. Do not verify or correct calculations. Return only valid JSON without any formatting or markers."

        print(prompt)
        new_nodes = call_llm(
            prompt, system_prompt=system_prompt, response_format="json_object"
        )
        print(new_nodes)
        print("#" * 100)

        if isinstance(new_nodes, list):
            self.computation_graph += [ComputationNode(**node) for node in new_nodes]
        else:
            self.computation_graph.append(ComputationNode(**new_nodes))

    def extract_leaf_nodes(self, question: str) -> List[ComputationNode]:
        """Extract leaf nodes (given values) from the question."""
        prompt = f"""
Extract the given numerical values from the question into computation nodes. Each node must follow this schema:
{{
    "id": "abc12345",  // Exactly 8 chars using letters and numbers
    "name": "standardized_name",  // Use consistent names across similar problems
    "value": float,
    "operation": null,
    "operands": null,
    "is_leaf": true
}}

Question: {question}

IMPORTANT:
- Use standardized names that will be consistent across similar problems
- Node IDs must be exactly 8 characters using letters (a-z) and numbers (0-9)
- Example IDs: "dist1234", "time5678", "rate9abc"
- Return only nodes for values explicitly given in the question
- Return valid JSON array of nodes

Example for a similar problem:
Question: "A fog bank covers 2 miles in 240 minutes. If the town is 60 miles wide..."
Nodes: [
    {{"id": "dist1234", "name": "distance_per_step", "value": 2.0, "is_leaf": true}},
    {{"id": "time5678", "name": "time_per_step", "value": 240.0, "is_leaf": true}},
    {{"id": "totl9012", "name": "total_distance", "value": 60.0, "is_leaf": true}}
]
"""
        system_prompt = "You are a precise parser that extracts given values from word problems. Use consistent naming across similar problems. Return only valid JSON without any formatting or markers."

        leaf_nodes = call_llm(
            prompt, system_prompt=system_prompt, response_format="json_object"
        )
        return [ComputationNode(**node) for node in leaf_nodes]

    def parse(
        self,
        question: str,
        deduction: str,
    ) -> List[ComputationNode]:

        self.computation_graph = self.extract_leaf_nodes(question)
        self.leaf_nodes = self.computation_graph.copy()
        # Then continue with the regular parsing
        steps_of_deduction = deduction.split(self.stop_word)
        for step in steps_of_deduction:
            if not any(char.isdigit() for char in step):
                continue
            if "####" in step or "answer" in step:
                continue
            self._one_step_parse(question, step)

        return self.computation_graph


def test():
    parser = IncrementalParser()

    # Test with original problem
    question1 = "A fog bank rolls in from the ocean to cover a city. It takes 354 minutes to cover every 3 miles of the city. If the city is 81 miles across from the oceanfront to the opposite inland edge, how many minutes will it take for the fog bank to cover the whole city?"
    deduction1 = """Let's think step by step. The fog bank covers 3 miles in 354 minutes. So it covers 1 mile in 354 / 3 = 118 minutes. The city is 81 miles across. So it will take 81 * 118 = 9618 minutes. The answer is 9618. #### 9618"""

    # Test with similar problem (different numbers)
    question2 = "A fog bank rolls in from the ocean to cover a town. It takes 240 minutes to cover every 2 miles of the town. If the town is 60 miles across from the oceanfront to the opposite inland edge, how many minutes will it take for the fog bank to cover the whole town?"
    deduction2 = """Let's think step by step. The fog bank covers 2 miles in 240 minutes. So it covers 1 mile in 240 / 2 = 120 minutes. The town is 60 miles across. So it will take 60 * 120 = 7200 minutes. The answer is 7200. #### 7200"""

    # Parse first problem
    graph1 = parser.parse(question1, deduction1)
    graph_dict = [node.to_dict() for node in graph1]
    with open(f"computation_graph_1.json", "w") as f:
        json.dump(graph_dict, f, indent=4)

    graph2 = parser.parse(question2, deduction2)
    graph_dict = [node.to_dict() for node in graph2]
    with open(f"computation_graph_2.json", "w") as f:
        json.dump(graph_dict, f, indent=4)


def run():
    import os

    model = "gemma_9B_it"
    parser = IncrementalParser()
    with open(f"results/gsm_parser_gpt/{model}_to_parse.json", "r") as f:
        data = json.load(f)

    if not os.path.exists(f"results/gsm_incremental_parsed/{model}"):
        os.makedirs(f"results/gsm_incremental_parsed/{model}")

    parsed, failed = 0, 0
    for qid, q in data.items():
        question = q["new_question"]
        deduction = q["response"]
        try:
            graph = parser.parse(question, deduction)
            graph_dict = [node.to_dict() for node in graph]
        except Exception as e:
            failed += 1
            print(f"Error parsing question {qid}: {e}")
            continue

        parsed += 1
        with open(f"results/gsm_incremental_parsed/{model}/{qid}.json", "w") as f:
            json.dump(graph_dict, f, indent=4)

    print(f"Parsed {parsed} questions, failed {failed} questions")


if __name__ == "__main__":
    run()
