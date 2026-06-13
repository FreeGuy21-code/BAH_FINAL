from ultralytics import YOLO
import sys

def main():
    print("Loading YOLO model...")
    # Load trained model
    model = YOLO("C:/BAH_FINAL/runs/detect/train_final6/weights/best.pt")
    
    print("Running validation on dataset defined in C:/BAH_FINAL/data.yaml...")
    # Run validation
    metrics = model.val(data="C:/BAH_FINAL/data.yaml", device="cpu") # run on CPU to avoid CUDA cache issues
    
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    
    # print metrics
    results_dict = metrics.results_dict
    for key, value in results_dict.items():
        print(f"{key}: {value:.6f}")
        
    print("="*50)
    print("Individual class metrics:")
    for i, name in enumerate(metrics.names.values()):
        print(f"Class '{name}':")
        print(f"  Precision: {metrics.box.p[i]:.6f}")
        print(f"  Recall:    {metrics.box.r[i]:.6f}")
        print(f"  AP50:      {metrics.box.ap50[i]:.6f}")
        print(f"  AP50-95:   {metrics.box.ap[i]:.6f}")
    print("="*50)

if __name__ == "__main__":
    main()
