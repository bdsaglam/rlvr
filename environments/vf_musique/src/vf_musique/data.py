import random
from typing import TypedDict

from datasets import Dataset, concatenate_datasets, load_dataset


def preprocess_answer(answer: str) -> str:
    """Preprocess answer to handle digit and ordinal number conversions."""
    answer = answer.lower().strip()

    # Convert digits to numbers
    digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    if answer in digits:
        return str(digits.index(answer))

    # Convert ordinal numbers to numbers
    mapping = {
        "zeroth": "0th",
        "first": "1st",
        "second": "2nd",
        "third": "3rd",
        "fourth": "4th",
        "fifth": "5th",
        "sixth": "6th",
        "seventh": "7th",
        "eighth": "8th",
        "ninth": "9th",
    }
    for k, v in mapping.items():
        answer = answer.replace(k, v)
    return answer


class MuSiQueDocument(TypedDict):
    id: str
    title: str
    body: str
    is_supporting: bool
    text: str


def _make_doc(p: dict) -> MuSiQueDocument:
    """Convert MuSiQue paragraph to document format (matches official verifiers)."""
    return {
        "id": str(p["idx"]),
        "title": p["title"],
        "body": p["paragraph_text"],
        "is_supporting": p.get("is_supporting", False),
        "text": f"# {p['title']}\n{p['paragraph_text']}",
    }


def preprocess_example(x: dict) -> dict:
    """Preprocess MuSiQue example for verifiers format (matches official verifiers)."""
    answers = [x["answer"], *x["answer_aliases"]]
    answers += [preprocess_answer(a) for a in answers]
    docs = [_make_doc(p) for p in x["paragraphs"]]

    n_hops = sum(doc["is_supporting"] for doc in docs)
    return {
        "question": x["question"],
        "answer": x["answer"],
        "info": {
            "id": x["id"],
            "docs": docs,
            "answers": list(set(answers)),
            "n_hops": n_hops,
        },
    }


def preprocess_dataset(dataset: Dataset) -> Dataset:
    """Preprocess dataset with proper column cleanup (matches official verifiers)."""
    columns_to_remove = list(
        set(dataset.column_names)
        - {
            "question",
            "answer",
            "info",
        }
    )
    new_dataset = dataset.map(preprocess_example, load_from_cache_file=False).remove_columns(columns_to_remove)
    return new_dataset


def load_datasets(dataset_str: str) -> Dataset:
    """
    Prepare a dataset from a string of the form "path,name,split".
    """
    ds_list = []
    for s in dataset_str.split(";"):
        path, name, split = s.split(",")
        ds = load_dataset(path, name, split=split)
        ds = preprocess_dataset(ds)
        ds_list.append(ds)

    return concatenate_datasets(ds_list).shuffle(seed=89)


def prepare_dataset(dataset_str: str, noise_rate: float = 1.0, **kwargs) -> Dataset:
    ds = load_datasets(dataset_str)
    ds = ds.filter(lambda x: x["info"]["n_hops"] < 3)

    if noise_rate != 1.0:

        def adjust_noise(x):
            x["info"]["docs"] = [
                doc for doc in x["info"]["docs"] if doc["is_supporting"] or random.random() < noise_rate
            ]
            return x

        ds = ds.map(adjust_noise)

    # Sort by number of hops (fewer hops first)
    ds = (
        ds.map(lambda x: {"n_hops": x["info"]["n_hops"]})
        .sort("n_hops", load_from_cache_file=False)
        .remove_columns(["n_hops"])
    )
    return ds
