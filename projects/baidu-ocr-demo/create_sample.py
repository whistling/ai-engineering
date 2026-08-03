"""Generate a sample document image for OCR testing."""
from PIL import Image, ImageDraw, ImageFont
import os

width, height = 800, 600
img = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(img)

# Try to use a readable font, fall back to default
try:
    font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    font_normal = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
except (OSError, IOError):
    font_large = ImageFont.load_default()
    font_normal = font_large
    font_small = font_large

# Title
draw.text((50, 30), "Invoice #2024-0731", fill="black", font=font_large)
draw.line([(50, 75), (750, 75)], fill="gray", width=2)

# Info block
draw.text((50, 100), "Date: 2024-07-05", fill="black", font=font_normal)
draw.text((50, 130), "Company: Acme Corp.", fill="black", font=font_normal)
draw.text((50, 160), "Address: 123 Main Street, San Francisco, CA 94102", fill="black", font=font_normal)

# Table header
y = 220
draw.rectangle([(50, y), (750, y + 30)], fill="#eeeeee")
draw.text((60, y + 5), "Item", fill="black", font=font_normal)
draw.text((300, y + 5), "Qty", fill="black", font=font_normal)
draw.text((450, y + 5), "Unit Price", fill="black", font=font_normal)
draw.text((620, y + 5), "Total", fill="black", font=font_normal)

# Table rows
items = [
    ("Widget A", "10", "$25.00", "$250.00"),
    ("Widget B", "5", "$42.50", "$212.50"),
    ("Service Fee", "1", "$100.00", "$100.00"),
]
for i, (item, qty, price, total) in enumerate(items):
    row_y = y + 35 + i * 30
    draw.text((60, row_y), item, fill="black", font=font_normal)
    draw.text((300, row_y), qty, fill="black", font=font_normal)
    draw.text((450, row_y), price, fill="black", font=font_normal)
    draw.text((620, row_y), total, fill="black", font=font_normal)

# Total line
draw.line([(50, 360), (750, 360)], fill="gray", width=1)
draw.text((450, 370), "Subtotal:", fill="black", font=font_normal)
draw.text((620, 370), "$562.50", fill="black", font=font_normal)
draw.text((450, 400), "Tax (8%):", fill="black", font=font_normal)
draw.text((620, 400), "$45.00", fill="black", font=font_normal)
draw.text((450, 435), "TOTAL:", fill="black", font=font_large)
draw.text((620, 435), "$607.50", fill="black", font=font_large)

# Footer
draw.line([(50, 500), (750, 500)], fill="gray", width=1)
draw.text((50, 520), "Thank you for your business!", fill="gray", font=font_small)
draw.text((50, 545), "Payment due within 30 days. Please reference invoice number.", fill="gray", font=font_small)

out_path = os.path.join(os.path.dirname(__file__), "samples", "sample_invoice.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
img.save(out_path, "PNG")
print(f"Saved: {out_path}")
