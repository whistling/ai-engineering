# PaddleOCR Demo

A simple demo showing how to use PaddleOCR to perform text detection and recognition on images/documents.

## Features
- Generates a sample receipt image with both Chinese and English text.
- Initializes PaddleOCR with the Chinese model (`lang="ch"` which supports both Chinese and English).
- Performs text detection, classification (direction classification), and recognition.
- Prints the results (text, confidence, and bounding boxes) to the console.
- Generates a visual output image with bounding boxes and recognized text overlaid.

## Prerequisites
Ensure Python 3.9+ is installed.

## Setup and Running

1. **Activate virtual environment & Install dependencies**:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the demo**:
   ```bash
   python main.py
   ```

3. **Check the outputs**:
   - The sample image is saved at `samples/sample_receipt.png`.
   - The OCR output image with bounding boxes is saved at `output/ocr_result.png`.
