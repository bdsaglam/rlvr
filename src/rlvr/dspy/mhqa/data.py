import dspy
from vf_musique.data import prepare_dataset


def prepare_musique_dataset(
    datasets_str: str = "bdsaglam/musique,answerable,train",
    noise_rate: float = 1.0,
):
    """Load and prepare MuSiQue dataset using vf_musique data functions."""
    # Use the official vf_musique data preparation
    dataset = prepare_dataset(datasets_str, noise_rate=noise_rate)

    # Convert to DSPy examples
    processed_examples = []
    for x in dataset:
        # Get supporting document IDs
        supporting_doc_ids = [doc["id"] for doc in x["info"]["docs"] if doc.get("is_supporting")]

        # Create DSPy example
        example = dspy.Example(  # noqa: F821
            question=x["question"],
            answer=x["answer"],
            answers=x["info"]["answers"],  # All valid answer forms
            docs=x["info"]["docs"],  # All documents
            supporting_ids=supporting_doc_ids,  # IDs of supporting docs
            n_hops=x["info"]["n_hops"],  # Number of hops
        ).with_inputs("question", "docs")

        processed_examples.append(example)

    return processed_examples
