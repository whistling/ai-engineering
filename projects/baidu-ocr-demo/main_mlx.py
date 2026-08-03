"""
Baidu Unlimited-OCR Demo — Apple Silicon (MLX) version
======================================================
Run Unlimited-OCR locally on M-series Mac using mlx-vlm with
community-quantized models.

References:
  - https://github.com/baidu/Unlimited-OCR
  - https://huggingface.co/sahilchachra/unlimited-ocr-8bit-mlx
  - https://huggingface.co/sahilchachra/unlimited-ocr-4bit-mlx

Usage:
    # Single image (default: Int4 model, lightest on memory)
    python main_mlx.py --image path/to/image.jpg

    # Use Int8 model for better accuracy
    python main_mlx.py --image path/to/image.jpg --model_variant 8bit

    # PDF file (each page parsed separately)
    python main_mlx.py --pdf path/to/document.pdf

    # Custom prompt
    python main_mlx.py --image path/to/image.jpg --prompt "Extract all text from this document."
"""

import argparse
import os
import sys
import tempfile
import time


# ──────────────────────────────────────────────────────────────
# PDF → images helper
# ──────────────────────────────────────────────────────────────
def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
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
# MLX model registry
# ──────────────────────────────────────────────────────────────
MLX_MODELS = {
    "4bit":  "sahilchachra/unlimited-ocr-4bit-mlx",    # ~2.3 GB, fastest
    "8bit":  "sahilchachra/unlimited-ocr-8bit-mlx",    # ~3.7 GB, higher accuracy
    "mxfp4": "sahilchachra/unlimited-ocr-mxfp4-mlx",   # ~2.2 GB
    "mxfp8": "sahilchachra/unlimited-ocr-mxfp8-mlx",   # ~3.6 GB
}


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────
def load_model(variant: str = "4bit"):
    """Load the MLX-quantized model + processor."""
    from mlx_vlm import load

    model_id = MLX_MODELS.get(variant)
    if model_id is None:
        sys.exit(f"Unknown variant '{variant}'. Choose from: {list(MLX_MODELS.keys())}")

    print(f"[Model] Loading {model_id} …")
    model, processor = load(model_id)
    print("[Model] Ready ✓")
    return model, processor


# ──────────────────────────────────────────────────────────────
# Output post-processing
# ──────────────────────────────────────────────────────────────
import re


def clean_output(raw: str, keep_det_tags: bool = False) -> str:
    """
    Clean up raw model output:
    1. Fix BPE artifacts (Ġ → space, Ċ → newline)
    2. Optionally strip <|det|>...<|/det|> bounding-box tags
    3. Remove hallucinated preamble before first detection
    4. Clean up HTML table to readable markdown
    """
    text = raw

    # Fix byte-level BPE artifacts
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")

    # Remove everything before the first <|det|> tag (hallucination preamble)
    first_det = text.find("<|det|>")
    if first_det > 0:
        text = text[first_det:]

    if not keep_det_tags:
        # Strip detection tags but keep the text content
        text = re.sub(r"<\|det\|>[^<]*<\|/det\|>", "", text)

    # Clean up extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────
def run_ocr(model, processor, image_path: str, prompt: str,
            max_tokens: int = 4096, keep_det_tags: bool = False) -> str:
    """Run OCR on a single image and return the generated text."""
    from mlx_vlm import generate

    print(f"\n[OCR] Processing: {image_path}")
    t0 = time.time()

    result = generate(
        model,
        processor,
        prompt,
        image=image_path,
        max_tokens=max_tokens,
        verbose=False,
    )

    # generate() may return a GenerationResult object or a string
    raw = result.text if hasattr(result, "text") else str(result)
    output = clean_output(raw, keep_det_tags=keep_det_tags)

    elapsed = time.time() - t0
    print(f"[OCR] Done in {elapsed:.1f}s ({len(output)} chars)")
    return output


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Unlimited-OCR on Apple Silicon via mlx-vlm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file")
    group.add_argument("--images", type=str, nargs="+", help="Paths to multiple images (parsed one by one)")
    group.add_argument("--pdf", type=str, help="Path to a PDF file")

    parser.add_argument(
        "--model_variant",
        type=str,
        choices=list(MLX_MODELS.keys()),
        default="4bit",
        help="Quantization variant. Default: 4bit (~2.3 GB). Use 8bit for better accuracy (~3.7 GB).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>document parsing.",
        help="Prompt to send to the model. Must contain <image> token. Default: '<image>document parsing.'",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max tokens to generate per image. Default: 4096",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save results. Default: ./output",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF-to-image conversion. Default: 200 (lower = faster, less memory)",
    )
    parser.add_argument(
        "--keep_det_tags",
        action="store_true",
        help="Keep <|det|> bounding-box tags in output. Default: strip them.",
    )
    args = parser.parse_args()

    # Validate inputs
    if args.image and not os.path.isfile(args.image):
        sys.exit(f"Error: file not found — {args.image}")
    if args.images:
        for p in args.images:
            if not os.path.isfile(p):
                sys.exit(f"Error: file not found — {p}")
    if args.pdf and not os.path.isfile(args.pdf):
        sys.exit(f"Error: file not found — {args.pdf}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect image paths
    if args.image:
        image_paths = [args.image]
    elif args.images:
        image_paths = args.images
    else:
        image_paths = pdf_to_images(args.pdf, dpi=args.dpi)

    # Load model
    model, processor = load_model(args.model_variant)

    # Process each image
    all_results = []
    for i, img_path in enumerate(image_paths):
        result = run_ocr(model, processor, img_path, args.prompt, args.max_tokens, args.keep_det_tags)
        all_results.append(result)

        # Save individual result
        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_file = os.path.join(args.output_dir, f"{basename}.md")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"[Save] {out_file}")

    # Save combined result if multiple pages
    if len(all_results) > 1:
        combined_file = os.path.join(args.output_dir, "combined_output.md")
        with open(combined_file, "w", encoding="utf-8") as f:
            for i, text in enumerate(all_results):
                f.write(f"\n\n---\n## Page {i + 1}\n\n")
                f.write(text)
        print(f"\n[Save] Combined → {combined_file}")

    print(f"\n✅ Processed {len(all_results)} page(s). Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
