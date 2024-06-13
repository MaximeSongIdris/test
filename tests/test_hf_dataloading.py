from datasets import load_dataset


def test_load_dataset():
    """Simple test."""

    DATA_DIR = "DRUNET_preprocessed"

    # download from Internet
    # https://huggingface.co/datasets/deepinv/drunet_dataset
    dataset = load_dataset("deepinv/denoising", split="train")
    assert (
        len(dataset) == 6
    ), f"Dataset should have been of len 6, instead got {len(dataset)}."

