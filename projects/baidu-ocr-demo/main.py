"""
Baidu Unlimited-OCR Demo
========================
Demonstrates single-image, multi-image, and PDF OCR using the
Unlimited-OCR model via HuggingFace Transformers.

References:
  - https://github.com/baidu/Unlimited-OCR
  - https://huggingface.co/baidu/Unlimited-OCR

Usage:
    # Single image (gundam config — best for single-page docs)
    python main.py --image path/to/image.jpg

    # Single image (base config — larger tile, no cropping)
    python main.py --image path/to/image.jpg --mode base

    # Multiple images
    python main.py --images page1.png page2.png page3.png

    # PDF file
    python main.py --pdf path/to/document.pdf

    # Specify output directory
    python main.py --image path/to/image.jpg --output_dir ./results
"""

import argparse
import os
import sys
import tempfile

import torch


# ──────────────────────────────────────────────────────────────
# PDF → images helper
# ──────────────────────────────────────────────────────────────
def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    """Convert every page of a PDF to a PNG image via PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    print(f"[PDF] Converted {len(paths)} page(s) → {tmp_dir}")
    return paths


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────
def load_model(model_name: str = "baidu/Unlimited-OCR"):
    """Load tokenizer + model onto GPU with bfloat16."""
    from transformers import AutoModel, AutoTokenizer

    print(f"[Model] Loading tokenizer from {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"[Model] Loading model from {model_name} …")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    print("[Model] Ready ✓")
    return tokenizer, model


# ──────────────────────────────────────────────────────────────
# Inference wrappers
# ──────────────────────────────────────────────────────────────
def infer_single(model, tokenizer, image_path: str, output_dir: str, mode: str = "gundam"):
    """
    Run single-image OCR.

    Modes:
      gundam — base_size=1024, image_size=640, crop_mode=True  (default, best quality)
      base   — base_size=1024, image_size=1024, crop_mode=False
    """
    if mode == "gundam":
        base_size, image_size, crop_mode = 1024, 640, True
    else:
        base_size, image_size, crop_mode = 1024, 1024, False

    print(f"\n[OCR] Single image — mode={mode}, file={image_path}")
    model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=image_path,
        output_path=output_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=128,
        save_results=True,
    )
    print(f"[OCR] Results saved → {output_dir}")


def infer_multi(model, tokenizer, image_files: list[str], output_dir: str):
    """Run multi-image OCR (base config only)."""
    print(f"\n[OCR] Multi-image — {len(image_files)} file(s)")
    model.infer_multi(
        tokenizer,
        prompt="<image>Multi page parsing.",
        image_files=image_files,
        output_path=output_dir,
        image_size=1024,
        max_length=32768,
        no_repeat_ngram_size=35,
        ngram_window=1024,
        save_results=True,
    )
    print(f"[OCR] Results saved → {output_dir}")


def infer_pdf(model, tokenizer, pdf_path: str, output_dir: str, dpi: int = 300):
    """Convert a PDF to images, then run multi-page OCR."""
    image_files = pdf_to_images(pdf_path, dpi=dpi)
    infer_multi(model, tokenizer, image_files, output_dir)


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Baidu Unlimited-OCR — single image / multi-image / PDF demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file")
    group.add_argument("--images", type=str, nargs="+", help="Paths to multiple image files")
    group.add_argument("--pdf", type=str, help="Path to a PDF file")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["gundam", "base"],
        default="gundam",
        help="Single-image config: gundam (crop, 640px) or base (no crop, 1024px). Default: gundam",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save OCR results. Default: ./output",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="baidu/Unlimited-OCR",
        help="HuggingFace model ID or local path. Default: baidu/Unlimited-OCR",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF-to-image conversion. Default: 300",
    )
    args = parser.parse_args()

    # Validate inputs
    if args.image and not os.path.isfile(args.image):
        sys.exit(f"Error: image file not found — {args.image}")
    if args.images:
        for p in args.images:
            if not os.path.isfile(p):
                sys.exit(f"Error: image file not found — {p}")
    if args.pdf and not os.path.isfile(args.pdf):
        sys.exit(f"Error: PDF file not found — {args.pdf}")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, model = load_model(args.model_name)

    if args.image:
        infer_single(model, tokenizer, args.image, args.output_dir, args.mode)
    elif args.images:
        infer_multi(model, tokenizer, args.images, args.output_dir)
    elif args.pdf:
        infer_pdf(model, tokenizer, args.pdf, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()
