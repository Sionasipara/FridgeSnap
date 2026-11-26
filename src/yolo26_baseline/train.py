from ultralytics import YOLO

model = YOLO("models/yolo26/weights/yolo26n.pt")

model.train(
    data="data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=9,
    patience=15,
    project="yolo26_baseline",
    name="train",
    plots=True,
    exist_ok=True
)
