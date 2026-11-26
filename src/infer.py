from ultralytics import YOLO

img_dir = "test/images/"
fname = ""

# Load updated YOLO
model = YOLO("runs/detect/yolo26_baseline/train/weights/best.pt")

# Run detection on source images
results = model.predict(
    source="data/"+img_dir+fname,
    conf=0.25,
    project="yolo26_baseline",
    name="infer",
    save=True,
    verbose=False,
    exist_ok=True
)

for result in results:
    print('---')
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(model.names[class_id], confidence)
