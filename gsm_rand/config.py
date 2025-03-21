MODELS = {
    "deepseek_qwen_1_5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen_1_5B_it": "Qwen/Qwen2.5-Math-1.5B-Instruct",
}

GSM_SYMBOLIC_MODELS = {
    "gemma_9B_it": "google/gemma-2-9b-it",
    "deepseek_qwen_1_5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen_1_5B_it": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "gemma_9B": "google/gemma-2-9b",
    "llama_3_8B": "meta-llama/Meta-Llama-3-8B",
    # "llama_3_8B_it": "meta-llama/Meta-Llama-3-8B-Instruct",
}

EVAL_MODELS = {
    "gemma_9B_it": "google/gemma-2-9b-it",
    # "deepseek_qwen_1_5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "qwen_1_5B_it": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    # "gemma_9B": "google/gemma-2-9b",
    "llama_3_8B": "meta-llama/Meta-Llama-3-8B",
    # "llama_3_8B_it": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma_9B": "google/gemma-2-9b",
}
EVAL_QUESTION_NAMES = [
    "butcher_sales",
    "water_left",
    "magician_cards",
    "cleaner_work",
]

FEW_SHOTS_INDICES = {
    "butcher_sales": [
        "tree_logging_calculation",
        "fruit_rollup_contest",
        "batting_cages",
    ],
    "water_left": [
        "tree_logging_calculation",
        "fruit_rollup_contest",
        "batting_cages",
    ],
    "magician_cards": [
        "tree_logging_calculation",
        "fruit_rollup_contest",
        "batting_cages",
    ],
    "fruit_rollup_contest": [
        "tree_logging_calculation",
        "batting_cages",
        "waterslide",
    ],
    "cleaner_work": [
        "tree_logging_calculation",
        "batting_cages",
        "waterslide",
    ],
}


"""
0 Tree Logging Calculation
1 Fruit RollUp Contest
2 Batting cages
3 Waterslide
4 Water left
5 Butcher Sales
6 Pencil Pairs
7 Total cards
8 Boat Rentals
9 Playlist Hours
10 Doughnuts
11 TV Buying
12 Item Buying
13 Magician Card
"""
