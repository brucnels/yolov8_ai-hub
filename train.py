from ultralytics import YOLO

model = YOLO("yolov8s.pt")
results = model.train(data="coco8.yaml", epochs=10, imgsz=640, batch=-1, workers=8, project="./")