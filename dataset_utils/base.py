"""
Dataset loading utilities.

This module provides dataset loading functions that work with the
existing dataset.py module but provide a cleaner interface.
"""

from src_coconut_multimode.dataset import get_dataset as _get_dataset_original
from typing import Optional
from transformers import PreTrainedTokenizer


def load_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    max_size: Optional[int] = None,
    split: str = "test"
):
    """
    Load a dataset from JSON file.

    This wraps the original get_dataset function from dataset.py.

    Args:
        dataset_path: Path to dataset JSON file
        tokenizer: Tokenizer to use for tokenization
        max_size: Maximum number of samples to load
        split: Dataset split (currently unused, for compatibility)

    Returns:
        Dataset with tokenized samples
    """
    max_size = max_size or 1000000000  # Default from original

    dataset = _get_dataset_original(
        path=dataset_path,
        tokenizer=tokenizer,
        max_size=max_size
    )

    return dataset


def get_dataset_info(dataset_path: str) -> dict:
    """
    Get information about a dataset without loading it.

    Args:
        dataset_path: Path to dataset JSON file

    Returns:
        Dictionary with dataset metadata
    """
    import json

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    return {
        "num_samples": len(data),
        "path": dataset_path,
        "format": "coconut_json"  # Standard format with question/steps/answer
    }
