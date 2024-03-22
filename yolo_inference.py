from ultralytics import YOLO 

model = YOLO('yolov8n')

result = model.predict('input_videos/bar_realm.mp4',conf=0.2, save=True)