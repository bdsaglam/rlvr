from datasets import Dataset, concatenate_datasets, load_dataset

PROMPT_TEMPLATE = """\
{question}

# Available Documents
{docs}
"""


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


def _make_doc(p: dict) -> dict:
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
    prompt = PROMPT_TEMPLATE.format(question=x["question"], docs="\n".join([f"{d['id']}. {d['title']}" for d in docs]))
    supporting_doc_slugs = [f"{doc['id']}. {doc['title']}" for doc in docs if doc["is_supporting"]]
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "docs": docs,
        "answer": x["answer"],
        "answers": list(set(answers)),
        "supporting_doc_slugs": supporting_doc_slugs,
        "n_hops": len(supporting_doc_slugs),
    }


def preprocess_dataset(dataset: Dataset) -> Dataset:
    """Preprocess dataset with proper column cleanup (matches official verifiers)."""
    columns_to_remove = list(
        set(dataset.column_names)
        - {
            "id",
            "prompt",
            "docs",
            "answer",
            "answers",
            "supporting_doc_slugs",
            "n_hops",
        }
    )
    new_dataset = dataset.map(preprocess_example, load_from_cache_file=False).remove_columns(columns_to_remove)
    return new_dataset


def prepare_dataset(path: str, name: str, split: str) -> Dataset:
    """Prepare a single dataset."""
    ds = load_dataset(path, name, split=split)
    return preprocess_dataset(ds)


def prepare_datasets(dataset_str: str) -> Dataset:
    """
    Prepare a dataset from a string of the form "path,name,split".
    """
    ds_list = []
    for s in dataset_str.split(";"):
        path, name, split = s.split(",")
        ds = prepare_dataset(path, name, split)
        ds_list.append(ds)

    return concatenate_datasets(ds_list).shuffle(seed=89)
