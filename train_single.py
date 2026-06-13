import os
import shutil
import yaml
from ultralytics import YOLO

def main():
    print("Step 1: Setting up a temporary dataset for a single image...")
    
    # Define paths
    base_dir = "C:/BAH_FINAL"
    temp_dir = os.path.join(base_dir, "temp_single_image_dataset")
    
    # Subdirectories
    train_img_dir = os.path.join(temp_dir, "images", "train")
    val_img_dir = os.path.join(temp_dir, "images", "val")
    train_lbl_dir = os.path.join(temp_dir, "labels", "train")
    val_lbl_dir = os.path.join(temp_dir, "labels", "val")
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    
    # Pick a single image and label
    src_image = os.path.join(base_dir, "images", "train", "tile_000.png")
    src_label = os.path.join(base_dir, "labels", "train", "tile_000.txt")
    
    if not os.path.exists(src_image) or not os.path.exists(src_label):
        print(f"Error: Source image or label not found!")
        print(f"Image exists: {os.path.exists(src_image)}")
        print(f"Label exists: {os.path.exists(src_label)}")
        return

    # Copy files to train and val folders
    shutil.copy(src_image, os.path.join(train_img_dir, "tile_000.png"))
    shutil.copy(src_image, os.path.join(val_img_dir, "tile_000.png"))
    shutil.copy(src_label, os.path.join(train_lbl_dir, "tile_000.txt"))
    shutil.copy(src_label, os.path.join(val_lbl_dir, "tile_000.txt"))
    
    # Create data.yaml
    data_yaml = {
        "path": temp_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["boulder"]
    }
    
    yaml_path = os.path.join(temp_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
        
    print("Dataset setup complete! YAML path:", yaml_path)
    print("\nStep 2: Loading trained model best.pt...")
    # Load trained model
    model = YOLO(os.path.join(base_dir, "runs/detect/train_final6/weights/best.pt"))
    
    print("\nStep 3: Fine-tuning on the single image for 5 epochs on CPU...")
    # Train model (fine-tuning)
    results = model.train(
        data=yaml_path,
        epochs=5,           # Train for 5 epochs
        imgsz=512,
        batch=1,
        device="cpu",       # CPU training for this single image is extremely fast
        workers=0,          # Single process to avoid overhead on Windows
        plots=False,
        augment=False,      # Disable augmentations
        val=False,          # Skip validation during training
        verbose=False       # Reduce verbose logs to avoid clutter
    )
    
    print("\nStep 4: Evaluating accuracy on the trained single image...")
    # Validate
    metrics = model.val(data=yaml_path, device="cpu", imgsz=512)
    
    print("\n" + "="*50)
    print("SINGLE IMAGE TRAINING & EVALUATION RESULTS")
    print("="*50)
    
    results_dict = metrics.results_dict
    for key, value in results_dict.items():
        print(f"{key}: {value:.6f}")
        
    print("="*50)
    
    # Cleanup temporary dataset to keep space clean
    try:
        shutil.rmtree(temp_dir)
        print("Temporary dataset directory cleaned up successfully.")
    except Exception as e:
        print("Could not clean up temporary directory:", e)

if __name__ == "__main__":
    main()
