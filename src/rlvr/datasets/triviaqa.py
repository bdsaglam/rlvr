from datasets import Dataset

PROMPT_TEMPLATE = "Question: {question}"


def preprocess_example(x: dict) -> dict:
    answers = [
        x["answer"]["value"],
        x["answer"]["normalized_value"],
        *x["answer"]["aliases"],
        *x["answer"]["normalized_aliases"],
    ]
    prompt = PROMPT_TEMPLATE.format(question=x["question"])
    return {
        "id": x["question_id"],
        "prompt": [{"role": "user", "content": prompt}],
        "answer": answers[0],
        "answers": list(set(answers)),
        "docs": [],
        "n_hops": 1,
    }


def preprocess_dataset(dataset: Dataset) -> Dataset:
    columns_to_remove = list(
        set(dataset.column_names)
        - {
            "id",
            "prompt",
            "docs",
            "answer",
            "answers",
            "n_hops",
        }
    )
    new_dataset = dataset.map(preprocess_example, load_from_cache_file=False).remove_columns(columns_to_remove)
    return new_dataset
