from ultralytics import YOLO
import glob
import os
import cv2
import numpy as np

# Load trained model
model = YOLO("C:/BAH_FINAL/runs/detect/train_final6/weights/best.pt")

# Collect image paths
train_images = glob.glob("C:/BAH_FINAL/images/train/*.png") + glob.glob("C:/BAH_FINAL/images/train/*.jpg")
val_images = glob.glob("C:/BAH_FINAL/images/val/*.png") + glob.glob("C:/BAH_FINAL/images/val/*.jpg")
all_images = train_images + val_images

# Create output directory
output_dir = "C:/BAH_FINAL/predictions_clean_ellipse"
os.makedirs(output_dir, exist_ok=True)

# --- Enhanced Crater Elimination Helper ---
def is_probable_crater(w, h, img_crop, confidence=None):
    """
    Enhanced crater detection using multiple criteria
    Returns True if object is likely a crater (to be filtered out)
    """
    
    # Rule 1: Size filtering - craters are often larger than boulders
    area = w * h
    if area > 5000:  # Adjust threshold based on your image resolution
        return True  # Likely large crater
    
    # Rule 2: Aspect ratio - craters are more circular than boulders
    aspect_ratio = max(w / h, h / w)
    if aspect_ratio < 1.2:  # Very circular objects are likely craters
        return True
    
    # Rule 3: Texture analysis - craters have smoother, more uniform interiors
    if img_crop.size > 0:
        gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) if len(img_crop.shape) == 3 else img_crop
        
        # Calculate texture variance (craters have lower variance)
        texture_variance = np.var(gray_crop)
        if texture_variance < 200:  # Adjust threshold as needed
            return True  # Smooth texture = likely crater
        
        # Edge density - craters have fewer internal edges
        edges = cv2.Canny(gray_crop, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density < 0.05:  # Low edge density = likely crater
            return True
    
    # Rule 4: Confidence filtering - if your model gives confidence scores
    if confidence is not None and confidence < 0.3:
        return True  # Low confidence detections might be craters
    
    # Rule 5: Shape analysis using contours
    if img_crop.size > 0:
        gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) if len(img_crop.shape) == 3 else img_crop
        
        # Find contours
        contours, _ = cv2.findContours(gray_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate circularity
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:  # High circularity = likely crater
                    return True
    
    return False  # Likely a boulder

# --- Alternative: Simple size-based filtering ---
def simple_crater_filter(w, h, min_boulder_size=100, max_crater_size=3000):
    """
    Simple filtering based on size assumptions:
    - Very small objects: noise
    - Very large objects: craters
    - Medium objects: boulders
    """
    area = w * h
    
    if area < min_boulder_size:  # Too small
        return True
    if area > max_crater_size:   # Too large (likely crater)
        return True
    
    aspect_ratio = max(w / h, h / w)
    if aspect_ratio < 1.1 and area > 1000:  # Large circular objects
        return True
    
    return False

# --- Loop through all images ---
for img_path in all_images:
    results = model.predict(
        source=img_path,
        conf=0.1,
        device='cpu',
        save=False,
        show_labels=False,
        show_conf=False
    )
    
    img = results[0].orig_img.copy()
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        
        # Get confidence score if available
        confidence = float(box.conf[0]) if box.conf is not None else None
        
        # Ensure valid crop region
        if w <= 0 or h <= 0 or y2 > img.shape[0] or x2 > img.shape[1]:
            continue
        
        # Get cropped image region
        img_crop = img[y1:y2, x1:x2]
        
        # Choose your filtering method:
        # Method 1: Enhanced filtering
        if is_probable_crater(w, h, img_crop, confidence):
            continue  # Skip drawing this box
        
        # Method 2: Simple filtering (comment out Method 1 and uncomment this)
        # if simple_crater_filter(w, h):
        #     continue
        
        # Draw ellipse for detected boulders
        center = (x1 + w // 2, y1 + h // 2)
        axes = (w // 2, h // 2)
        angle = 0
        color = (0, 255, 255)  # Yellow
        thickness = 1
        cv2.ellipse(img, center, axes, angle, 0, 360, color, thickness)
        
        # Optional: Add size information as text
        # cv2.putText(img, f"{w}x{h}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # Save the image
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, img)

print("✅ Ellipse-only filtered predictions saved in:", output_dir)