from ultralytics import YOLO

# Load pretrained YOLO
model = YOLO("yolo26n.pt")

# Run detection on a fridge image
results = model.predict(
    source="data/test_images/fridge.jpg",
    conf=0.25,
    save=True
)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(model.names[class_id], confidence)
