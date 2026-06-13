from ultralytics import YOLO
import glob
import os
import torch

# Load model
model = YOLO("yolov8n.pt")  

# Collect images
train_images = glob.glob("images/train/*.png")
val_images = glob.glob("images/val/*.png")
all_images = train_images + val_images

# Output dir
output_dir = "predictions_clean"
os.makedirs(output_dir, exist_ok=True)

# Thresholds
CONF_THRESHOLD = 0.4
MAX_BOX_AREA = 50000

for img_path in all_images:
    results = model.predict(source=img_path, conf=CONF_THRESHOLD, verbose=False)
    result = results[0]
    boxes = result.boxes

    if boxes is None or boxes.data is None:
        continue

    # Filter boxes
    filtered = []
    for box in boxes.data:
        x1, y1, x2, y2 = box[:4]
        w, h = x2 - x1, y2 - y1
        area = w * h
        if area < MAX_BOX_AREA:
            filtered.append(box)

    
    if filtered:
        result.boxes.data = torch.stack(filtered)
    else:
        result.boxes.data = torch.empty((0, 6))  

    result.save(filename=os.path.join(output_dir, os.path.basename(img_path)))

print("✅ Done saving cleaned predictions.")
