from ultralytics import YOLO

img_dir = "test/images/"
fname = ""

# Load pretrained YOLO
model = YOLO("yolo26n.pt")

# Run detection on a fridge image
results = model.predict(
    source="data/"+img_dir+fname,
    conf=0.25,
    save=False,
    verbose=False
)

for result in results:
    print('---')
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(model.names[class_id], confidence)
