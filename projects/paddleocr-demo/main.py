import os
import sys
import json
import subprocess


def generate_sample_image():
    sample_path = os.path.join(os.path.dirname(__file__), "samples", "sample_receipt.png")
    if not os.path.exists(sample_path):
        print("Generating sample receipt image...")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "create_sample.py")], check=True)
    return sample_path


def get_bbox(box):
    """
    Safely parses bounding box coordinates in multiple formats:
    - [x_min, y_min, x_max, y_max]
    - [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    - numpy ndarray representation of the above
    """
    import numpy as np
    if hasattr(box, 'tolist'):
        box = box.tolist()

    if not isinstance(box, (list, tuple)):
        return 0, 0, 0, 0

    if len(box) == 4 and not isinstance(box[0], (list, tuple)):
        # Format: [x_min, y_min, x_max, y_max]
        return float(box[0]), float(box[1]), float(box[2]), float(box[3])
    elif len(box) == 4 and isinstance(box[0], (list, tuple)) and len(box[0]) == 2:
        # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
    else:
        try:
            arr = np.array(box)
            if arr.ndim == 1 and arr.size == 4:
                return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
            elif arr.ndim == 2 and arr.shape == (4, 2):
                return float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())
        except Exception:
            pass
        return 0, 0, 0, 0


def reconstruct_layout(texts, scores, boxes, y_threshold=15):
    """
    Reconstructs the reading layout by grouping bounding boxes that are vertically close,
    and sorting them left-to-right horizontally.
    """
    parsed_items = []
    for t, s, b in zip(texts, scores, boxes):
        x_min, y_min, x_max, y_max = get_bbox(b)
        parsed_items.append({
            'text': t,
            'score': s,
            'bbox': (x_min, y_min, x_max, y_max),
            'center_y': (y_min + y_max) / 2,
            'height': y_max - y_min
        })

    # Sort primarily by vertical center
    sorted_items = sorted(parsed_items, key=lambda x: x['center_y'])

    lines = []
    for item in sorted_items:
        placed = False
        # Try to fit the item in an existing line
        for line in lines:
            line_centers = [li['center_y'] for li in line]
            avg_center_y = sum(line_centers) / len(line_centers)
            avg_height = sum(li['height'] for li in line) / len(line)

            # Group into the same line if within distance relative to average height
            limit = max(y_threshold, avg_height * 0.6)
            if abs(item['center_y'] - avg_center_y) < limit:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    # Sort each line left-to-right (by x_min)
    reconstructed_lines = []
    for line in lines:
        sorted_line = sorted(line, key=lambda x: x['bbox'][0])
        reconstructed_lines.append(sorted_line)

    # Sort the lines themselves by their average center_y
    reconstructed_lines = sorted(reconstructed_lines,
                                 key=lambda line: sum(item['center_y'] for item in line) / len(line))

    return reconstructed_lines


def format_line_with_spaces(line_items, char_width=10):
    """
    Aligns items in the same line visually based on their pixel coordinate gaps.
    """
    if not line_items:
        return ""

    result_str = line_items[0]['text']
    for i in range(1, len(line_items)):
        prev_item = line_items[i - 1]
        curr_item = line_items[i]

        prev_x_max = prev_item['bbox'][2]
        curr_x_min = curr_item['bbox'][0]

        gap = curr_x_min - prev_x_max
        if gap > 0:
            num_spaces = max(2, int(gap / char_width))
            result_str += " " * num_spaces
        else:
            result_str += "  "

        result_str += curr_item['text']

    return result_str


def convert_to_markdown(reconstructed_lines):
    """
    Intelligently converts the reconstructed layout into structured Markdown.
    Groups lines with 3+ items into Markdown tables (joined by \n to preserve table rendering),
    formats 2-item lines as bold key-values, and single-item lines as headings/paragraphs (joined by \n\n).
    """
    md_blocks = []
    current_table = []

    for line in reconstructed_lines:
        num_items = len(line)
        texts = [item['text'] for item in line]

        if num_items >= 3:
            # Table row
            if not current_table:
                # Add header and separator
                current_table.append("| " + " | ".join(texts) + " |")
                current_table.append("| " + " | ".join(["---"] * num_items) + " |")
            else:
                current_table.append("| " + " | ".join(texts) + " |")
        else:
            # Flush existing table to blocks before writing a non-table element
            if current_table:
                md_blocks.append("\n".join(current_table))
                current_table = []

            if num_items == 2:
                # Format as bold key-value pair
                md_blocks.append(f"**{texts[0]}** {texts[1]}")
            elif num_items == 1:
                text = texts[0]
                # Format specific keywords as headers
                if "收据" in text or "Receipt" in text or "发票" in text or "Invoice" in text:
                    md_blocks.append(f"# {text}")
                else:
                    md_blocks.append(text)

    # Flush remaining table if exists
    if current_table:
        md_blocks.append("\n".join(current_table))

    return "\n\n".join(md_blocks)


def run_ocr(img_path):
    print("Initializing PaddleOCR...")
    from paddleocr import PaddleOCR

    # Initialize OCR engine.
    ocr = PaddleOCR(
        lang="ch",
        ocr_version="PP-OCRv6",  # 显式指定 OCR 版本
        use_doc_orientation_classify=False,  # 关闭文档整体方向矫正
        use_doc_unwarping=False,  # 关闭文档折痕/扭曲自动拍平
        use_textline_orientation=True  # 仅保留文字方向检测（如倒字）
    )

    print(f"\nRunning OCR on: {img_path} ...")
    result = ocr.predict(img_path)

    if not result or len(result) == 0:
        print("No text detected.")
        return

    res = result[0]

    texts = res.get('rec_texts', [])
    scores = res.get('rec_scores', [])
    boxes = res.get('rec_boxes', [])

    # 1. Perform layout reconstruction
    print("\n--- Reconstructed Reading Layout ---")
    reconstructed_lines = reconstruct_layout(texts, scores, boxes)

    formatted_lines = []
    for line in reconstructed_lines:
        line_str = format_line_with_spaces(line)
        formatted_lines.append(line_str)
        print(line_str)

    # 2. Output and Save Files
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save the reconstructed text layout
    layout_file_path = os.path.join(output_dir, "ocr_layout.txt")
    with open(layout_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_lines))
    print(f"\nSaved layout text to: {layout_file_path}")

    # Save structured JSON data
    json_file_path = os.path.join(output_dir, "ocr_data.json")
    structured_data = []
    for line_idx, line in enumerate(reconstructed_lines):
        line_data = []
        for item in line:
            line_data.append({
                'text': item['text'],
                'confidence': float(item['score']),
                'bbox': [float(coord) for coord in item['bbox']]
            })
        structured_data.append({
            'line_number': line_idx + 1,
            'items': line_data
        })

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)
    print(f"Saved structured JSON data to: {json_file_path}")

    # 3. Generate and Save Markdown
    markdown_content = convert_to_markdown(reconstructed_lines)
    markdown_file_path = os.path.join(output_dir, "ocr_layout.md")
    with open(markdown_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Saved Markdown file to: {markdown_file_path}")

    # Save the visualized image using native PaddleOCR helper
    res.save_to_img(output_dir)
    print(f"Saved visualized images to: {output_dir}")


if __name__ == "__main__":
    sample_img = generate_sample_image()
    run_ocr(sample_img)
