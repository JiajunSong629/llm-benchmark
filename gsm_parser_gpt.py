from openai import OpenAI
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import random
import re
import dotenv
import os

dotenv.load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


@dataclass(frozen=True, eq=True)
class ComputationNode:
    value: float
    operation: Optional[str] = None  # '+', '-', '*', '/', None for leaf nodes
    entity: Optional[str] = None  # corresponds to keys in variables_dict
    operands: Optional[tuple["ComputationNode"]] = (
        None  # Changed from List to Optional[tuple]
    )

    def __post_init__(self):
        if self.operands is None:
            object.__setattr__(self, "operands", ())
        elif isinstance(self.operands, list):
            object.__setattr__(self, "operands", tuple(self.operands))


def extract_first_part(text: str) -> str:
    pattern = r"####\s*([-+]?\d*\.?\d+)"
    match = re.search(pattern, text)
    if match:
        return text[: match.start()]
    return text


def parse_deduction_with_gpt(deduction: str, question: str) -> ComputationNode:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    prompt = f"""Given a math word problem and its solution deduction, create a computation graph showing the complete mathematical reasoning process.
    First extract key numerical values and their meanings from the question, then track all calculations in the deduction.

    Focus on:
    1. Identifying important numbers and what they represent from the question
    2. Tracking ALL intermediate calculations and their dependencies
    3. Ensuring the final answer is derived from all necessary previous steps

    Each node in the graph should contain:
    - value: numerical result
    - entity: what this value represents (extracted from the question context)
    - operation: mathematical operation ('+', '-', '*', '/' or null for leaf nodes)
    - operands: list of input nodes

    Json Schema:
    {{
        "nodes": [
            {{
                "id": "node1",
                "value": float,
                "entity": string,
                "operation": string or null,
                "operands": ["node_id1", "node_id2"]
            }}
        ],
        "edges": [
            {{
                "from": "node_id",
                "to": "node_id"
            }}
        ],
        "final_node_id": "node_id"  # Must point to the final calculation that gives the answer
    }}

    Question: {question}
    Parse the following deduction, capturing ALL mathematical steps:
    Deduction: {deduction}
    """

    print("Parsing deduction with GPT...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a precise parser that converts mathematical deductions into computation graphs.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    print("... Done")

    with open("response.json", "w") as f:
        f.write(response.choices[0].message.content)

    graph_data = json.loads(response.choices[0].message.content)
    return graph_data


def build_computation_graph(graph_data: Dict) -> ComputationNode:
    nodes = {}

    # First pass: create all nodes without operands
    for node_data in graph_data["nodes"]:
        nodes[node_data["id"]] = ComputationNode(
            value=float(node_data["value"]),
            operation=node_data["operation"],
            entity=node_data["entity"],
        )

    # Second pass: link operands
    for node_data in graph_data["nodes"]:
        if node_data["operands"]:
            nodes[node_data["id"]] = ComputationNode(
                value=nodes[node_data["id"]].value,
                operation=nodes[node_data["id"]].operation,
                entity=nodes[node_data["id"]].entity,
                operands=tuple(nodes[op_id] for op_id in node_data["operands"]),
            )

    return nodes[graph_data["final_node_id"]]


def restore_solution(root_node: ComputationNode) -> str:
    """Restore a one-line solution from the computation graph"""

    def _process_node(node: ComputationNode) -> str:
        if not node.operands:
            return str(node.value)

        operands = [_process_node(op) for op in node.operands]
        if len(operands) == 2:
            return f"({operands[0]} {node.operation} {operands[1]})"
        else:
            # For cases with more than 2 operands
            return (
                f"({node.operation.join([' ' + op + ' ' for op in operands]).strip()})"
            )

    expression = _process_node(root_node)
    return f"{expression} = {root_node.value}"


def plot_computation_tree(
    root_node: ComputationNode, save_path: str = "computation_tree.png"
):
    """Plot the computation tree using graphviz, similar to gsm_parser.py implementation"""
    try:
        from graphviz import Digraph
    except ImportError:
        print("Please install graphviz: pip install graphviz")
        return

    dot = Digraph(comment="Computation Graph")
    dot.attr(rankdir="BT")  # Bottom to top layout

    processed_nodes = set()

    def add_node_to_graph(node: ComputationNode) -> str:
        # Create a unique identifier for the node
        node_id = str(id(node))

        if node in processed_nodes:
            return node_id

        processed_nodes.add(node)

        # Format node label with integer values
        if node.operands:  # Intermediate node
            label = f"{node.value}"
            shape = "ellipse"
        else:  # Leaf node (input)
            label = f"{node.entity}\n{node.value}" if node.entity else f"{node.value}"
            shape = "box"

        # Add the node
        dot.node(node_id, label, shape=shape)

        if node.operands:
            # Create an operation node
            op_node_id = f"op_{node_id}"
            dot.node(
                op_node_id,
                node.operation,
                shape="circle",
                style="filled",
                fillcolor="lightgray",
            )

            # Connect operation node to result
            dot.edge(op_node_id, node_id)

            # Process operands and connect them to operation node
            for operand in node.operands:
                operand_id = add_node_to_graph(operand)
                dot.edge(operand_id, op_node_id)

        return node_id

    # Start the recursive process from root
    add_node_to_graph(root_node)

    # Save the graph and cleanup the DOT source file
    dot.render(save_path.rsplit(".", 1)[0], view=False, format="png", cleanup=True)


# Example usage:
if __name__ == "__main__":
    os.makedirs("parsed_gsm", exist_ok=True)

    qa = json.load(open("original_qa.json", "r"))
    for qid, qa_dict in qa.items():
        if os.path.exists(f"parsed_gsm/graph_data_{qid}.json") and os.path.exists(
            f"parsed_gsm/graph_data_{qid}.png"
        ):
            print(f"Skipping question {qid} as it has already been parsed")
            continue

        try:
            question = qa_dict["question"]
            generated_deduction = qa_dict["answer"]
            graph_data = parse_deduction_with_gpt(generated_deduction, question)
            with open(f"parsed_gsm/graph_data_{qid}.json", "w") as f:
                json.dump(graph_data, f)

            computation_graph = build_computation_graph(graph_data)
            plot_computation_tree(
                computation_graph, save_path=f"parsed_gsm/graph_data_{qid}.png"
            )

            solution_gpt = restore_solution(computation_graph)
            print(f"Restored solution: {solution_gpt}")
        except Exception as e:
            print(f"Error parsing question {qid}: {e}")
            continue
