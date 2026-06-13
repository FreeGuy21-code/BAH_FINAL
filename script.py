import cv2
import os


CLASS_ID = 0  
MIN_AREA = 50
SMALL_MAX = 150
MEDIUM_MAX = 400
MAX_AREA = 1000  


input_dirs = {
    "train": "images/train",
    "val": "images/val"
}
output_base = "labels"

os.makedirs(output_base, exist_ok=True)

for split, img_dir in input_dirs.items():
    output_dir = os.path.join(output_base, split)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(img_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            continue

        image_path = os.path.join(img_dir, filename)
        label_path = os.path.join(output_dir, filename.rsplit(".", 1)[0] + ".txt")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Skipped: {image_path}")
            continue

        height, width = img.shape[:2]

        
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 3)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with open(label_path, "w") as f:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if MIN_AREA < area < MAX_AREA:
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"✅ Saved: {label_path}")
