from ultralytics import YOLO 
import cv2
import pickle
import sys
from utils import get_center_of_bbox
import os
sys.path.append('../')

class YardTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        yard_detections = []

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                yard_detections = pickle.load(f)
            return yard_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            yard_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(yard_detections, f)
        
        return yard_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        yard_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            border = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            yard = {}
            yard['box'] = border
            yard['cls'] = object_cls_id
            yard_dict[track_id] = yard
        
        return yard_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, yard_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            # Đỏ
            for track_id, info in yard_dict.items():
                id = info["cls"]
                x1, y1, x2, y2 = info["box"]
                cv2.putText(frame, f"{id}",(int(x1),int(y1-10)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
                cv2.circle(frame, (get_center_of_bbox(info["box"])), 5, (0, 0, 255),-1)
                # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            output_video_frames.append(frame)
        
        return output_video_frames