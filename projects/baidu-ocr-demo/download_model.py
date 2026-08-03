"""
Download Baidu Unlimited-OCR model from Hugging Face.
Run this script first to cache the model locally before inference.

Usage:
    python download_model.py
    python download_model.py --model_dir ./models/Unlimited-OCR
"""

import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download Unlimited-OCR model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Local directory to save the model. If not set, uses HF cache.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="baidu/Unlimited-OCR",
        help="Hugging Face model ID",
    )
    args = parser.parse_args()

    print(f"Downloading model: {args.model_name}")
    path = snapshot_download(
        repo_id=args.model_name,
        local_dir=args.model_dir,
    )
    print(f"Model downloaded to: {path}")


if __name__ == "__main__":
    main()
