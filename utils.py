import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mat_colors
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def get_token_variants(tokenizer, base_words):
    """
    Get all variants of the given words including capitalization and whitespace.

    Args:
        tokenizer: The tokenizer to use for encoding
        base_words: List of base words to generate variants for

    Returns:
        Dict mapping base words to their token IDs and decoded forms
    """
    variants = {}
    for word in base_words:
        variants[word] = []
        # Generate variants with different capitalizations and whitespace
        word_variants = [
            word,
            " " + word,
            word.lower(),
            " " + word.lower(),
            word.capitalize(),
            " " + word.capitalize(),
        ]

        # Get token IDs for each variant
        for variant in word_variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            # Only include single-token variants to avoid ambiguity
            if len(token_ids) == 1:
                decoded = tokenizer.decode(token_ids[0])
                variants[word].append({"token_id": token_ids[0], "decoded": decoded})
    return variants


def analyze_specific_tokens(
    model, tokenizer, input_text, target_words=None, apply_chat_template=True
):
    """
    Analyzes probabilities of specific tokens at each generation step.

    Args:
        model: The language model
        tokenizer: The tokenizer
        input_text: Input prompt text
        target_words: List of words to track (default: ["But", "Alternatively", "Wait"])
        apply_chat_template: Whether to apply chat template

    Returns:
        List of dictionaries containing token analysis for each step
    """
    if target_words is None:
        target_words = ["But", "Alternatively", "Wait"]

    # Get variants for target words
    token_variants = get_token_variants(tokenizer, target_words)

    model.to(device)
    if apply_chat_template:
        messages = [
            {
                "role": "user",
                "content": f"Here is a problem: {input_text}. What is the answer to this problem?",
            }
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )["input_ids"]
    else:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")["input_ids"]

    analysis = []
    generated_text = ""

    pbar = tqdm(desc="Generating tokens", unit=" tokens")
    steps = 0

    while True:
        with torch.no_grad():
            outputs = model(input_ids.to(device))
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            # Get top token and its probability
            top_prob, top_idx = torch.max(probs, dim=-1)
            top_token = tokenizer.decode(top_idx)
            generated_text += top_token

            # Track probabilities of target tokens and their variants
            target_probs = {}
            for word, variants in token_variants.items():
                variant_probs = []
                for var in variants:
                    token_id = var["token_id"]
                    prob = probs[0, token_id].item()
                    variant_probs.append(
                        {"variant": var["decoded"], "probability": prob}
                    )
                target_probs[word] = variant_probs

            # Store the analysis
            analysis.append(
                {
                    "step": steps,
                    "generated_token": top_token,
                    "generated_text": generated_text,
                    "target_token_probs": target_probs,
                }
            )

            # Update progress
            pbar.update(1)
            pbar.set_postfix({"current_token": top_token})

            # Append the chosen token to input_ids
            input_ids = torch.cat([input_ids.to(device), top_idx.unsqueeze(0)], dim=-1)

            steps += 1
            if steps % 100 == 0:
                print(f"\n\nCurrent generation at step {steps}:")
                print(generated_text)
                print("-" * 50)

            if top_idx.item() == tokenizer.eos_token_id or steps > 500:
                break

    pbar.close()

    return analysis


def get_attention_scores(
    model,
    tokenizer,
    input_text,
    apply_chat_template=True,
):
    """
    Analyzes probabilities of specific tokens at each generation step.

    Args:
        model: The language model
        tokenizer: The tokenizer
        input_text: Input prompt text
        apply_chat_template: Whether to apply chat template

    Returns:
    """
    model.config.output_attentions = True
    model.to(device)
    if apply_chat_template:
        messages = [{"role": "user", "content": input_text}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )["input_ids"]
    else:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")["input_ids"]

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=2000,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        output = model(generated_ids.to(model.device))
        next_token_logits = output.logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        generated_ids = torch.cat(
            [generated_ids, next_token_probs.argmax(dim=-1).unsqueeze(0)], dim=-1
        )
    attentions = output.attentions
    attentions = np.array([a.squeeze(0).cpu().numpy() for a in attentions])

    return {
        "tokens": [tokenizer.decode(t) for t in generated_ids[0]],
        "attentions": attentions,
    }


def visz_score(
    tokens,
    scores,
    alpha=0.2,
    cmap="BuGn",
    write_to=None,
):
    off = (sum(scores) / len(scores)) * alpha
    color_map = cm.get_cmap(cmap)
    normer = mat_colors.Normalize(vmin=min(scores) - off, vmax=max(scores) + off)
    colors = [mat_colors.to_hex(color_map(normer(x))) for x in scores]

    if len(tokens) != len(colors):
        raise ValueError("number of tokens and colors don't match")

    style_elems = []
    span_elems = []
    for i in range(len(tokens)):
        style_elems.append(f".c{i} {{ background-color: {colors[i]}; }}")
        span_elems.append(f'<span class="c{i}">{tokens[i]}</span>')

    result = f"""<!DOCTYPE html><html><head><link href="https://fonts.googleapis.com/css?family=Roboto+Mono&display=swap" rel="stylesheet"><style>body {{ max-width: 800px; margin: 0 auto; }} span {{ font-family: "Roboto Mono", monospace; font-size: 15px; }} {''.join(style_elems)}</style></head><body>{''.join(span_elems)}</body></html>"""

    if write_to:
        with open(write_to, "w") as f:
            f.write(result)

    return result


def visz_attn(
    tokens,
    scores,
    write_to=None,
    alpha=0,
    layer_indices=None,
    head_indices=None,
    min_attention=0.01,
):
    """
    Creates an interactive visualization of attention scores using sparse representation.
    Only stores attention values above min_attention threshold.
    """
    L, H, Tm1, _ = scores.shape
    assert (
        len(tokens) == Tm1 + 1
    ), "Scores shape should be (L, H, T-1, T-1) where T is len(tokens)"

    # Pre-compute sparse normalized scores for all layers and heads
    normalized_scores = {}
    for l in range(L):
        normalized_scores[l] = {}
        for h in range(H):
            normalized_scores[l][h] = {}
            for t in range(Tm1):
                scores_t = scores[l, h, t].tolist()
                rest_scores = scores_t[1:]  # Skip first token
                if rest_scores:
                    min_score = min(rest_scores)
                    max_score = max(rest_scores)
                    offset = (max_score - min_score) * alpha

                    # Create sparse representation with significant values only
                    sparse_scores = {}
                    for idx, score in enumerate(rest_scores, start=1):
                        normalized = int(
                            255
                            * (score - min_score + offset)
                            / (max_score - min_score + 2 * offset + 1e-6)
                        )
                        if (
                            normalized > min_attention * 255
                        ):  # Only store significant values
                            sparse_scores[str(idx)] = normalized

                    if sparse_scores:  # Only store non-empty attention patterns
                        normalized_scores[l][h][t] = sparse_scores

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css?family=Roboto+Mono&display=swap" rel="stylesheet">
        <style>
            body { max-width: 1000px; margin: 0 auto; font-family: "Roboto Mono", monospace; }
            .controls { margin: 20px 0; }
            .token { 
                display: inline-block; 
                padding: 2px 2px; 
                margin: 0;
                cursor: pointer; 
                border: 1px solid transparent;
            }
            .token:hover { opacity: 0.8; }
            .selected { border: 1px solid #808080 !important; }
        </style>
    </head>
    <body>
        <div class="controls">
            <label for="layer">Layer:</label>
            <select id="layer" onchange="clearAttention()">
    """

    for l in range(L):
        layer_num = layer_indices[l] if layer_indices else l + 1
        html += f'<option value="{l}">Layer {layer_num}</option>'

    html += """
            </select>
            <label for="head">Head:</label>
            <select id="head" onchange="clearAttention()">
    """

    for h in range(H):
        head_num = head_indices[h] if head_indices else h + 1
        html += f'<option value="{h}">Head {head_num}</option>'

    html += """
            </select>
            <div>Click on any token to visualize its attention:</div>
        </div>
        <div id="tokens">
    """

    for i, token in enumerate(tokens):
        clickable = ' onclick="showAttention(this.dataset.idx)"' if i > 0 else ""
        html += f'<span class="token" data-idx="{i}"{clickable}>{token}</span>'

    html += (
        """
        </div>
        
        <script>
            const normalizedScores = %s;
            const tokens = document.querySelectorAll('.token');
            
            function getColor(normalizedScore) {
                const whiteComponent = 255 - (normalizedScore || 0);
                return `rgba(${whiteComponent}, 255, ${whiteComponent}, 0.3)`;
            }

            function clearAttention() {
                tokens.forEach(t => {
                    t.classList.remove('selected');
                    t.style.backgroundColor = '';
                });
            }

            function showAttention(tokenIdx) {
                const layerIdx = document.getElementById('layer').value;
                const headIdx = document.getElementById('head').value;
                
                clearAttention();
                
                // Get sparse normalized scores
                const scores = normalizedScores[layerIdx]?.[headIdx]?.[tokenIdx-1] || {};
                
                // Reset all tokens to default (no attention)
                tokens.forEach(t => t.style.backgroundColor = '');
                
                // Only update tokens with significant attention
                for (const [idx, score] of Object.entries(scores)) {
                    tokens[parseInt(idx)].style.backgroundColor = getColor(score);
                }
                
                // Highlight selected token
                tokens[tokenIdx].classList.add('selected');
            }
        </script>
    </body>
    </html>
    """
        % normalized_scores
    )

    if write_to:
        with open(write_to, "w", encoding="utf-8") as f:
            f.write(html)
        return None

    return html
