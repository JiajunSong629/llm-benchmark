from typing import List, Dict, Tuple
import torch
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm
import json

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ThoughtNode:
    text: str  # The generated text segment
    token_probs: List[Tuple[str, float]]  # (token, probability) pairs
    attention_weights: torch.Tensor  # Attention weights for this generation step
    depth: int  # Current depth in the graph
    parent_id: str  # ID of the parent node
    node_id: str  # Unique identifier for this node


class ThoughtPathAnalyzer:
    def __init__(
        self,
        model,
        tokenizer,
        prompt: str,
        top_n: int,
        max_depth: int,
        max_new_tokens: int = 50,
        allow_steps: int = 5,
        apply_chat_template: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        # self.prompt = prompt
        self.top_n = top_n
        self.max_depth = max_depth
        self.max_new_tokens = max_new_tokens
        self.allow_steps = allow_steps
        self.graph = nx.DiGraph()

        self.model.to(device)

        if apply_chat_template:
            self.prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            self.prompt = prompt

    def _get_next_tokens(
        self, input_ids: torch.Tensor
    ) -> Tuple[List[Tuple[str, float]], torch.Tensor]:
        """Get top-n next tokens and their probabilities, along with attention weights."""
        with torch.no_grad():
            outputs = self.model(input_ids.to(device), output_attentions=True)

        logits = outputs.logits[:, -1, :]
        attention = outputs.attentions[-1]  # Get last layer attention

        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], self.top_n)

        tokens = [
            (self.tokenizer.decode(idx), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]

        return tokens, attention

    def _generate_path(
        self, current_text: str, depth: int, parent_id: str, token_choice_idx: int = 0
    ) -> ThoughtNode:
        """Generate next segment until reaching specified number of full stops."""
        input_ids = self.tokenizer.encode(current_text, return_tensors="pt")
        generated_text = ""
        num_tokens = 0
        steps_completed = 0

        print(f"  Branch {token_choice_idx+1}/{self.top_n}...")

        while num_tokens < self.max_new_tokens:
            next_tokens, attention = self._get_next_tokens(input_ids)
            token, prob = next_tokens[token_choice_idx]

            generated_text += token
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([[self.tokenizer.encode(token)[-1]]]),
                ],
                dim=-1,
            )

            num_tokens += 1
            print(
                f"    Generated {num_tokens} tokens, current token: [{token}]",
                end="\r",
                flush=True,
            )

            if "." in token:
                steps_completed += 1
                if steps_completed >= self.allow_steps:
                    # Get probabilities for next tokens after completing all steps
                    next_tokens, attention = self._get_next_tokens(input_ids)
                    break

        print()  # New line after completion
        node_id = f"d{depth}_{''.join(str(hash(generated_text))[:8])}"
        return ThoughtNode(
            text=generated_text,
            token_probs=next_tokens,
            attention_weights=attention,
            depth=depth,
            parent_id=parent_id,
            node_id=node_id,
        )

    def analyze(self) -> nx.DiGraph:
        """Build the thought process graph."""
        # First generate until the specified number of full stops
        input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt")
        initial_text = self.prompt
        print("\nGenerating initial sentences:")
        print("-------------------------")

        steps_completed = 0
        while steps_completed < self.allow_steps:
            next_tokens, attention = self._get_next_tokens(input_ids)
            token, prob = next_tokens[
                0
            ]  # Use most probable token for initial generation

            initial_text += token
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([[self.tokenizer.encode(token)[-1]]]),
                ],
                dim=-1,
            )

            print(f"Generated token: [{token}]", end="\r", flush=True)

            if "." in token:
                steps_completed += 1
                print(f"\nCompleted {steps_completed}/{self.allow_steps} sentences")
                if steps_completed < self.allow_steps:
                    continue
                # Get probabilities for branching
                next_tokens, attention = self._get_next_tokens(input_ids)
                break

        print(f"\nInitial text: {initial_text}")
        print("\nBuilding thought process graph:")
        print("------------------------------")

        root_node = ThoughtNode(
            text=initial_text,
            token_probs=next_tokens,
            attention_weights=attention,
            depth=0,
            parent_id=None,
            node_id="root",
        )

        self.graph.add_node(root_node.node_id, data=root_node)
        queue = [(initial_text, 0, root_node.node_id)]

        while queue:
            current_text, depth, parent_id = queue.pop(0)

            if depth >= self.max_depth:
                node = self._generate_path(current_text, depth, parent_id)
                self.graph.add_node(node.node_id, data=node)
                self.graph.add_edge(
                    parent_id, node.node_id, weight=node.token_probs[0][1]
                )
                continue

            # Generate multiple paths
            print(f"\nDepth {depth}: Generating {self.top_n} branches...")
            for i in range(self.top_n):
                node = self._generate_path(
                    current_text, depth, parent_id, token_choice_idx=i
                )
                self.graph.add_node(node.node_id, data=node)
                self.graph.add_edge(
                    parent_id, node.node_id, weight=node.token_probs[i][1]
                )

                queue.append((current_text + node.text, depth + 1, node.node_id))
            print()  # New line after completing all branches at this depth

        print("\nGraph construction complete!\n")
        return self.graph

    def visualize_text(self):
        """Visualize the thought process graph using text indentation."""

        def _print_node(node_id, depth=0, visited=None):
            if visited is None:
                visited = set()

            if node_id in visited:
                return
            visited.add(node_id)

            node = self.graph.nodes[node_id]["data"]
            indent = "  " * depth
            prob_str = ""

            # Add probability for non-root nodes
            if node.parent_id is not None:
                edge_prob = self.graph[node.parent_id][node_id]["weight"]
                prob_str = f" (p={edge_prob:.3f})"

            print(f"{indent}├─ {node.text.strip()}{prob_str}")

            # Sort children by probability
            children = list(self.graph.successors(node_id))
            children.sort(key=lambda x: self.graph[node_id][x]["weight"], reverse=True)

            for child in children:
                _print_node(child, depth + 1, visited)

        print("Thought Process Tree:")
        print("--------------------")
        _print_node("root")

    def export_to_json(self, output_file: str = None):
        """Export the thought process to a JSON structure with just text and probabilities."""

        def _process_node(node_id, visited=None):
            if visited is None:
                visited = set()

            if node_id in visited:
                return None
            visited.add(node_id)

            node = self.graph.nodes[node_id]["data"]
            children = list(self.graph.successors(node_id))
            children.sort(key=lambda x: self.graph[node_id][x]["weight"], reverse=True)

            node_data = {"text": node.text.strip(), "continuations": []}

            # Process children
            for child in children:
                child_data = _process_node(child, visited)
                if child_data:
                    prob = self.graph[node_id][child]["weight"]
                    node_data["continuations"].append(
                        {"probability": prob, "continuation": child_data}
                    )

            return node_data

        json_data = {"prompt": self.prompt, "reasoning": _process_node("root")}

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

        return json_data


def example_usage():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2"
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        "gpt2"
    )
    analyzer = ThoughtPathAnalyzer(
        model=model,
        tokenizer=tokenizer,
        prompt="The cat sat on the map.",
        top_n=2,
        max_depth=2,
        max_new_tokens=1000,  # Control generation length
        allow_steps=1,  # Generate 2 sentences before branching
        apply_chat_template=False,
    )
    graph = analyzer.analyze()
    analyzer.visualize_text()
    analyzer.export_to_json(output_file="temp.json")


if __name__ == "__main__":
    example_usage()
