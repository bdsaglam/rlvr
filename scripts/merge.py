import logging
from pathlib import Path

import torch
import typer
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from verifiers.utils.cuda import get_half_precision_dtype

log = logging.getLogger(__name__)

app = typer.Typer()


def load_tokenizer_model(model_name_or_path: str, **model_kwargs):
    # Load model
    model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    # Load tokenizer
    tokenizer_id = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    return tokenizer, model


def save_artifacts(model, tokenizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving model and tokenizer locally: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


@app.command()
def merge_adapters_and_publish(
    model_id: str = typer.Argument(..., help="Model name or path to merge adapters from"),
    out: Path = typer.Option(..., help="Output directory to save model and tokenizer"),
    torch_dtype: str = get_half_precision_dtype(),
):
    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        torch_dtype = getattr(torch, torch_dtype)

    log.info(f"Loading model and tokenizer for {model_id}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map={"": torch.cuda.current_device()},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    log.info("Merging adapters to model...")
    model = model.merge_and_unload()

    log.info(f"Saving model and tokenizer locally: {out}")
    save_artifacts(model, tokenizer, out)


if __name__ == "__main__":
    app()
