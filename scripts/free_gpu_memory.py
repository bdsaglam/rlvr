import gc
import os

import torch
import typer
from dotenv import load_dotenv

load_dotenv()


def clear():
    """Clear GPU memory"""
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    print(f"Clearing GPU memory for {visible_devices}")
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    typer.run(clear)
