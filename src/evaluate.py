from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/fridge/yolo26_baseline/weights/best.pt"
)

metrics = model.val(
    data="data/data.yaml",
    split="test",
    imgsz=640,
    plots=True,
    exist_ok=True,
    project="runs/fridge",
)

print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# Per-class mAP50-95
for class_id in metrics.box.ap_class_index:
    name = model.names[int(class_id)]
    score = metrics.box.maps[int(class_id)]
    print(f"{name}: {score:.3f}")
