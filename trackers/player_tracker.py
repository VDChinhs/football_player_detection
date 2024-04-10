from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            border = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            if (id_name_dict[object_cls_id] != "ball"):
                player = {}
                player['box'] = border
                player['cls'] = id_name_dict[object_cls_id]
                player_dict[track_id] = player
            
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        color = {
            "player": (0, 255, 0), #Xanh cây
            "goalkeeper": (0,0,255), # Xanh nước biển
            "referee" :(255,0,255), # Hồng
        }
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, info in player_dict.items():
                name = info["cls"]
                x1, y1, x2, y2 = info["box"]
                if(name == "player"):
                    cv2.putText(frame, f"{name}: {track_id}",(int(x1),int(y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, color[name], 1)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color[name], 1)
                elif( name == "goalkeeper"):
                    cv2.putText(frame, f"{name}: {track_id}",(int(x1),int(y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, color[name], 1)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color[name], 1)
                elif( name == "referee"):
                    cv2.putText(frame, f"{name}: {track_id}",(int(x1),int(y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, color[name], 1)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color[name], 1)

            output_video_frames.append(frame)
        
        return output_video_frames