from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="data/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        patience=15,
        project="runs/fridge",
        name="yolo26_baseline",
        plots=True
    )

if __name__ == "__main__":
    main()