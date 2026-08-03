from PIL import Image, ImageDraw, ImageFont
import os

width, height = 800, 600
img = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(img)

# Try to find a font supporting Chinese
font_path = "/System/Library/Fonts/PingFang.ttc"
if not os.path.exists(font_path):
    font_path = "/System/Library/Fonts/STHeiti Light.ttc"
if not os.path.exists(font_path):
    font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
if not os.path.exists(font_path):
    font_path = None

try:
    if font_path:
        font_large = ImageFont.truetype(font_path, 32)
        font_normal = ImageFont.truetype(font_path, 20)
        font_small = ImageFont.truetype(font_path, 16)
    else:
        raise OSError
except OSError:
    font_large = ImageFont.load_default()
    font_normal = font_large
    font_small = font_large

# Draw text
draw.text((50, 30), "测试收据 / Test Receipt", fill="black", font=font_large)
draw.line([(50, 75), (750, 75)], fill="gray", width=2)

draw.text((50, 100), "日期 / Date: 2026-07-06", fill="black", font=font_normal)
draw.text((50, 130), "商户 / Merchant: 极客咖啡馆 / Geek Cafe", fill="black", font=font_normal)
draw.text((50, 160), "地址 / Address: 北京市海淀区中关村南大街", fill="black", font=font_normal)

# Table Header
y = 220
draw.rectangle([(50, y), (750, y + 30)], fill="#eeeeee")
draw.text((60, y + 5), "品名 / Description", fill="black", font=font_normal)
draw.text((350, y + 5), "数量 / Qty", fill="black", font=font_normal)
draw.text((500, y + 5), "单价 / Price", fill="black", font=font_normal)
draw.text((650, y + 5), "金额 / Amount", fill="black", font=font_normal)

# Rows
items = [
    ("美式咖啡 / Americano", "2", "￥25.00", "￥50.00"),
    ("芝士蛋糕 / Cheesecake", "1", "￥38.00", "￥38.00"),
    ("牛角包 / Croissant", "3", "￥15.00", "￥45.00"),
]
for i, (item, qty, price, total) in enumerate(items):
    row_y = y + 35 + i * 30
    draw.text((60, row_y), item, fill="black", font=font_normal)
    draw.text((350, row_y), qty, fill="black", font=font_normal)
    draw.text((500, row_y), price, fill="black", font=font_normal)
    draw.text((650, row_y), total, fill="black", font=font_normal)

# Total
draw.line([(50, 360), (750, 360)], fill="gray", width=1)
draw.text((450, 370), "小计 / Subtotal:", fill="black", font=font_normal)
draw.text((650, 370), "￥133.00", fill="black", font=font_normal)
draw.text((450, 400), "服务费 / Service Fee:", fill="black", font=font_normal)
draw.text((650, 400), "￥0.00", fill="black", font=font_normal)
draw.text((450, 435), "总计 / TOTAL:", fill="black", font=font_large)
draw.text((650, 435), "￥133.00", fill="black", font=font_large)

# Footer
draw.line([(50, 500), (750, 500)], fill="gray", width=1)
draw.text((50, 520), "谢谢惠顾 / Thank you for your visit!", fill="gray", font=font_small)

out_dir = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "sample_receipt.png")
img.save(out_path, "PNG")
print(f"Saved: {out_path}")
