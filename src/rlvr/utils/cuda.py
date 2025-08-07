import torch


def get_half_precision_dtype() -> str:
    """Get appropriate half precision dtype based on GPU capability.

    Returns:
        str: "bfloat16" or "float16"
    """

    major, _ = torch.cuda.get_device_capability()
    return "bfloat16" if major >= 8 else "float16"
