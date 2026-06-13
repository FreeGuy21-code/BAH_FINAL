from ultralytics import YOLO

def main():
    # model
    model = YOLO("yolov8n.pt")  

    # Training of the model
    model.train(
        data="C:/BAH_FINAL/data.yaml",
        epochs=50,
        imgsz=512,
        batch=4,
        device=0,  # Use GPU
        name="train_final",
        augment=True,
        patience=5
    )
cache = False
persist = True
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
