# /// script
# dependencies = [
#   "pandas",
# ]

import argparse
import json
import sys
from pathlib import Path
from typing import Generator

import pandas as pd


def parse_params(record):
    params_node = record.get("data", {}).get("params", {})
    params = {}
    for v in params_node.values():
        params.update(v.get("data", {}))
    return params


def parse_metrics(record):
    metrics_node = record.get("data", {}).get("metrics", {})
    metrics = {}
    for v in metrics_node.values():
        metrics.update(v.get("data", {}))
    return metrics


def parse_experiment(record):
    return {
        "id": record["rev"],
        "name": record["name"],
        "params": parse_params(record),
        "metrics": parse_metrics(record),
    }


def parse_experiments(data: list[dict]) -> Generator[dict, None, None]:
    for node in data:
        if node.get("error"):
            continue
        commit = node.get("rev")
        if experiments := (node.get("experiments") or []):
            for experiment in experiments:
                for rev in experiment.get("revs") or []:
                    if not rev.get("error"):
                        yield {"commit": commit, **parse_experiment(rev)}
        else:
            yield {"commit": commit, **parse_experiment(node)}


def load_experiments(json_filepath=None):
    if json_filepath:
        with open(json_filepath, "r") as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)
    return list(parse_experiments(data))


def main():
    parser = argparse.ArgumentParser(description="Process DVC experiment data")
    parser.add_argument("--input", "-i", help="Input JSON file (optional, defaults to stdin)")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for processed experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments(args.input)
    print(f"{len(experiments)} experiments loaded")

    df = pd.json_normalize(experiments)
    if df.empty:
        return

    for _, row in df.iterrows():
        name = row.get("name")
        if not name or name == "main":
            continue
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(dict(row), f)


if __name__ == "__main__":
    main()
