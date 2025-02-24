import json
import random
import numpy as np
from utils import visz_score, create_folder, visz_attn
from config import MODELS


def test_run():
    import numpy as np

    text = "This is a very long sample text that should automatically wrap to multiple lines when it exceeds the maximum width. It demonstrates how the text wrapping works while maintaining proper spacing and probability coloring for each character."
    i = 0
    tokens = []
    while i < len(text):
        next_token_len = random.randint(3, 10)
        tokens.append(text[i : min(i + next_token_len, len(text))])
        i = i + next_token_len

    scores = np.linspace(0, 1, len(tokens))
    np.random.shuffle(scores)

    visz_score(tokens, scores, write_to="test.html")


def main_attn():
    import pickle

    for model_name, model_path in MODELS.items():
        with open(f"results/{model_name}_attn.pkl", "rb") as f:
            attn_scores = pickle.load(f)

        tokens = attn_scores["tokens"]
        scores = attn_scores["attentions"]

        # take a random sample of layers and heads of scores
        selected_layers = [0, 5, 10, 15, 20]
        selected_heads = [0, 5, 10]
        scores = scores[np.ix_(selected_layers, selected_heads)]

        visz_attn(
            tokens,
            scores,
            write_to=f"figures/{model_name}_attn.html",
            layer_indices=selected_layers,
            head_indices=selected_heads,
        )


def main():
    create_folder("figures")
    for model_name, model_path in MODELS.items():
        print(f"Processing {model_name}")

        analysis_result = json.load(open(f"results/{model_name}.json"))
        tokens = [r["generated_token"] for r in analysis_result]
        target_words = list(analysis_result[0]["target_token_probs"].keys())

        for target_word in target_words:
            scores = []
            for r in analysis_result:
                rr = r["target_token_probs"][target_word]
                scores.append(sum(rrr["probability"] for rrr in rr))

            visz_score(
                tokens,
                scores,
                write_to=f"figures/{model_name}_{target_word}.html",
                as_image=False,
            )


if __name__ == "__main__":
    main_attn()
