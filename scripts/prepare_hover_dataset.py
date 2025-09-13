#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "httpx",
#   "typer",
#   "datasets",
#   "tqdm",
# ]
# ///

"""
Script to prepare the HOVER dataset and push it to Hugging Face Hub.
Since HuggingFace no longer allows datasets with custom scripts, this converts
the HOVER dataset to a standard format and uploads it directly.
"""

from typing import Any, Dict, List

import httpx
import typer
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from tqdm import tqdm

# URLs for the HOVER dataset files
TRAIN_URL = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_train_release_v1.1.json"
VALID_URL = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_dev_release_v1.1.json"
TEST_URL = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_test_release_v1.1.json"

DESCRIPTION = """
HoVer is an open-domain, many-hop fact extraction and claim verification dataset built upon the Wikipedia corpus. 
The original 2-hop claims are adapted from question-answer pairs from HotpotQA. 
It is collected by a team of NLP researchers at UNC Chapel Hill and Verisk Analytics.
"""

CITATION = """
@inproceedings{jiang2020hover,
  title={{HoVer}: A Dataset for Many-Hop Fact Extraction And Claim Verification},
  author={Yichen Jiang and Shikha Bordia and Zheng Zhong and Charles Dognin and Maneesh Singh and Mohit Bansal.},
  booktitle={Findings of the Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year={2020}
}
"""


def download_json(url: str) -> List[Dict[str, Any]]:
    """Download JSON data from URL using httpx."""
    print(f"Downloading from {url}")
    resp = httpx.get(url)
    resp.raise_for_status()
    return resp.json()


def process_split(data: List[Dict[str, Any]], split_name: str) -> Dict[str, List]:
    """Process a single data split into the format expected by datasets library."""
    processed = {
        "id": [],
        "uid": [],
        "claim": [],
        "supporting_facts": [],
        "label": [],
        "num_hops": [],
        "hpqa_id": [],
    }

    is_test = split_name == "test"

    for idx, item in enumerate(tqdm(data, desc=f"Processing {split_name}")):
        processed["id"].append(idx)
        processed["uid"].append(item["uid"])
        processed["claim"].append(item["claim"])

        if not is_test:
            # For train/validation, include all fields
            supporting_facts = [{"key": fact[0], "value": fact[1]} for fact in item.get("supporting_facts", [])]
            processed["supporting_facts"].append(supporting_facts)
            processed["label"].append(item["label"])
            processed["num_hops"].append(item["num_hops"])
            processed["hpqa_id"].append(item.get("hpqa_id", ""))
        else:
            # For test, hide labels and supporting facts
            processed["supporting_facts"].append([])
            processed["label"].append("NOT_SUPPORTED")  # Default label for test
            processed["num_hops"].append(-1)
            processed["hpqa_id"].append("None")

    return processed


def create_dataset_dict() -> DatasetDict:
    """Download and process all splits of the HOVER dataset."""

    # Download data
    print("Downloading HOVER dataset...")
    train_data = download_json(TRAIN_URL)
    valid_data = download_json(VALID_URL)
    test_data = download_json(TEST_URL)

    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(valid_data)}")
    print(f"Test examples: {len(test_data)}")

    # Process each split
    train_processed = process_split(train_data, "train")
    valid_processed = process_split(valid_data, "validation")
    test_processed = process_split(test_data, "test")

    # Define features schema
    features = Features(
        {
            "id": Value("int32"),
            "uid": Value("string"),
            "claim": Value("string"),
            "supporting_facts": [
                {
                    "key": Value("string"),
                    "value": Value("int32"),
                }
            ],
            "label": ClassLabel(names=["NOT_SUPPORTED", "SUPPORTED"]),
            "num_hops": Value("int32"),
            "hpqa_id": Value("string"),
        }
    )

    # Create datasets
    train_dataset = Dataset.from_dict(train_processed, features=features)
    valid_dataset = Dataset.from_dict(valid_processed, features=features)
    test_dataset = Dataset.from_dict(test_processed, features=features)

    # Create dataset dictionary
    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "validation": valid_dataset,
            "test": test_dataset,
        }
    )

    return dataset_dict


def push_to_hub(dataset_dict: DatasetDict, repo_id: str, private: bool = False):
    """Push the dataset to Hugging Face Hub."""
    print(f"Pushing dataset to hub: {repo_id}")

    # Push to hub
    dataset_dict.push_to_hub(
        repo_id,
        private=private,
    )

    print(f"Dataset successfully pushed to https://huggingface.co/datasets/{repo_id}")


def main(
    repo_id: str = typer.Option("hover", help="Repository ID on HuggingFace Hub (e.g., 'username/hover-dataset')"),
    private: bool = typer.Option(False, help="Make the dataset private on HuggingFace Hub"),
    dry_run: bool = typer.Option(False, help="Only process a small sample for testing"),
):
    """
    Prepare and upload HOVER dataset to HuggingFace Hub.
    """
    dataset_dict = create_dataset_dict()

    if not dry_run:
        push_to_hub(dataset_dict, repo_id, private)


if __name__ == "__main__":
    typer.run(main)
