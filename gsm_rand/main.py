import random
from data import TaskTemplate


def magician_cards_variable_generator(setting):
    def is_integer_solution(vars):
        green_cards = (
            vars["red_cards"] + vars["red_cards"] * vars["green_multiplier"] / 100
        )
        return green_cards.is_integer()

    var_dict = {
        "magician": random.choice(["magician", "athelete", "artist", "scientist"]),
        "red_color": random.choice(["red", "blue"]),
        "green_color": random.choice(["green", "purple"]),
        "yellow_color": random.choice(["yellow", "orange"]),
        "red_cards": random.randint(10, 300),
        "green_multiplier": random.choice([25, 30, 50, 80]),
    }
    while not is_integer_solution(var_dict):
        var_dict["red_cards"] = random.randint(10, 300)
        var_dict["green_multiplier"] = random.choice([25, 30, 50, 80])
    green_cards_more = var_dict["red_cards"] * var_dict["green_multiplier"] // 100
    var_dict["yellow_cards"] = int(var_dict["red_cards"] * 2 + green_cards_more)
    return var_dict


magician_cards_template = TaskTemplate(
    type_name="Magician Cards",
    question_template="Question: In a set of {magician}'s cards, there are {red_cards} {red_color} cards, and {green_multiplier}% more {green_color} cards. {yellow_color} cards are as many, as the sum of {red_color} and {green_color} cards. How many cards of all mentioned colors are there?",
    deduction_template="""Answer: Let's think step by step. First find the number of {green_color} cards: {red_cards} {red_color} cards * {green_multiplier}% = <<{red_cards}*{green_multiplier}/100={green_cards_more}>>{green_cards_more} more {green_color} cards.
So there are {red_cards} + {green_cards_more} = <<{red_cards}+{green_cards_more}={green_cards}>>{green_cards} {green_color} cards.
Then find the number of {yellow_color} cards: {red_cards} {red_color} cards + {green_cards} {green_color} cards = <<{red_cards}+{green_cards}={yellow_cards}>>{yellow_cards} {yellow_color} cards.
Finally, add up all the cards: {red_cards} {red_color} cards + {green_cards} {green_color} cards + {yellow_cards} {yellow_color} cards = <<{red_cards}+{green_cards}+{yellow_cards}={total_cards}>>{total_cards} cards.
#### {total_cards}""",
    variable_generator=magician_cards_variable_generator,
    answer_generator=lambda vars: {
        "green_cards_more": int(vars["red_cards"] * vars["green_multiplier"] / 100),
        "green_cards": int(
            vars["red_cards"] + vars["red_cards"] * vars["green_multiplier"] / 100
        ),
        "total_cards": vars["red_cards"]
        + int(vars["red_cards"] + vars["red_cards"] * vars["green_multiplier"] / 100)
        + vars["yellow_cards"],
    },
)


"""
A maintenance worker has to clean a university with 210 floors. They have 10 days to get it done. It takes them 20 minutes per floor. If they work 10 hours each day, what percentage of their day, on average, is spent cleaning floors?
"""


def cleaner_work_variable_generator(setting):
    def is_integer_solution(vars):
        floors_per_day = vars["floors"] / vars["days"]
        minutes_per_day = floors_per_day * vars["minutes_per_floor"]
        clean_hours_per_day = minutes_per_day / 60
        percentage_of_day = clean_hours_per_day / vars["hours_per_day"] * 100

        return (
            floors_per_day.is_integer()
            and minutes_per_day.is_integer()
            and clean_hours_per_day.is_integer()
            and percentage_of_day.is_integer()
            and percentage_of_day < 100
        )

    var_dict = {
        "floors": random.randint(100, 500),
        "days": random.randint(5, 10),
        "minutes_per_floor": random.randint(20, 80),
        "hours_per_day": random.randint(7, 10),
    }
    while not is_integer_solution(var_dict):
        var_dict["floors"] = random.randint(100, 500)
        var_dict["minutes_per_floor"] = random.randint(20, 80)
        var_dict["hours_per_day"] = random.randint(8, 10)
        var_dict["days"] = random.randint(5, 10)

    return var_dict


cleaner_work_template = TaskTemplate(
    type_name="Cleaner Work",
    question_template="Question: A maintenance worker has to clean a university with {floors} floors. They have {days} days to get it done. It takes them {minutes_per_floor} minutes per floor. If they work {hours_per_day} hours each day, what percentage of their day, on average, is spent cleaning floors?",
    deduction_template="""Answer: Let's think step by step. First, find the number of floors the worker needs to clean per day: {floors} / {days}=<<{floors}/{days}={floors_per_day}>>{floors_per_day}. Then find the number of minutes to work per day: {floors_per_day} * {minutes_per_floor} =<<{floors_per_day}*{minutes_per_floor}={minutes_per_day}>>{minutes_per_day}. Next convert the minutes to hours: {minutes_per_day} / 60 = <<{minutes_per_day}/60={clean_hours_per_day}>>{clean_hours_per_day}. Hence the percentage of time spent cleaning: {clean_hours_per_day} / {hours_per_day} * 100 = <<{clean_hours_per_day}/{hours_per_day}*100={percentage_of_day}>>{percentage_of_day}%
    #### {percentage_of_day}""",
    variable_generator=cleaner_work_variable_generator,
    answer_generator=lambda vars: {
        "floors_per_day": int(vars["floors"] / vars["days"]),
        "minutes_per_day": int(
            vars["floors"] / vars["days"] * vars["minutes_per_floor"]
        ),
        "clean_hours_per_day": int(
            vars["floors"] / vars["days"] * vars["minutes_per_floor"] / 60
        ),
        "percentage_of_day": int(
            vars["floors"]
            / vars["days"]
            * vars["minutes_per_floor"]
            / 60
            / vars["hours_per_day"]
            * 100
        ),
    },
)

########################################################################################
########################################################################################
########################################################################################
########################################################################################

from data import TaskTemplate, task_templates

TASK_TEMPLATES = {}
for t in task_templates:
    t_name = "_".join([c.lower() for c in t.type_name.split()])
    TASK_TEMPLATES[t_name] = t
TASK_TEMPLATES["magician_cards"] = magician_cards_template
TASK_TEMPLATES["cleaner_work"] = cleaner_work_template


def generate_few_shots(setting, num_shots, seed=None, few_shots_indices=None):
    if few_shots_indices is None:
        indices = list(range(len(task_templates)))
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)
        few_shots_indices = indices[:num_shots]

    full_prompt = ""
    for index in few_shots_indices:
        example = TASK_TEMPLATES[index].generate(setting)
        full_prompt += f"{example['question']}\n{example['deduction']}\n\n"

    return full_prompt


def collect_questions(
    setting="original",
    num_shots=3,
    num_samples=10,
    seed=None,
    few_shots_indices=None,
    question_name=None,
):
    if seed is not None:
        random.seed(seed)

    if question_name is None:
        question_name = random.choice(list(TASK_TEMPLATES.keys()))

    if few_shots_indices is None:
        # select other than question_index
        few_shots_indices = [i for i in TASK_TEMPLATES.keys() if i != question_name]
        few_shots_indices = random.sample(few_shots_indices, num_shots)

    print(question_name, few_shots_indices)

    few_shots = generate_few_shots(setting, num_shots, seed, few_shots_indices)
    questions = []
    for i in range(num_samples):
        result = TASK_TEMPLATES[question_name].generate(setting)
        question = result["question"]
        deduction = result["deduction"]

        for j in range(1, len(deduction.split(".")) - 1):
            deduction_util = ".".join(deduction.split(".")[:j])
            questions.append(
                {
                    "sample_id": i,
                    "deduction_step": j,
                    "prompt": f"{few_shots}{question}\n{deduction_util}.",
                    "answer": result["answer"],
                }
            )

    return questions


if __name__ == "__main__":
    import json
    import os
    from config import FEW_SHOTS_INDICES, EVAL_QUESTION_NAMES

    os.makedirs("data/questions", exist_ok=True)

    for question_name in EVAL_QUESTION_NAMES:
        questions = collect_questions(
            "original",
            3,
            50,
            seed=42,
            question_name=question_name,
            few_shots_indices=FEW_SHOTS_INDICES[question_name],
        )
        with open(f"data/questions/{question_name}_questions.json", "w") as f:
            json.dump(questions, f, indent=4, ensure_ascii=False)
