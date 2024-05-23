from ultralytics import YOLO
import cv2
import pickle
import sys
from utils import get_center_of_bbox
import os

sys.path.append("../")


class YardTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def findRL(self, text):
        for char in text:
            if char == "R" or char == "L":
                return char

    def clean_data(self, frames):
        output_detection_yard = []
        for frame in frames:
            processed_keys = []
            for key1, value1 in frame.items():
                if key1 in processed_keys:
                    continue
                for key2, value2 in frame.items():
                    if key1 != key2 and value1["cls"] == value2["cls"]:
                        if (
                            get_center_of_bbox(value1["bbox"])[0]
                            < get_center_of_bbox(value2["bbox"])[0]
                        ):
                            if self.findRL(value1["cls_name"]) == "L":
                                if value1["cls"] == 0 or value1["cls"] == 16:
                                    value2["cls"] += 1
                                else:
                                    if value2["cls"] == 15:
                                        value2["cls"] -= 3
                                    else:
                                        value2["cls"] -= 1
                            else:
                                if value1["cls"] == 1 or value1["cls"] == 17:
                                    value2["cls"] -= 1
                                else:
                                    if value1["cls"] == 12:
                                        value1["cls"] += 3
                                    else:
                                        value1["cls"] += 1
                            processed_keys.append(key2)

                        elif (
                            get_center_of_bbox(value1["bbox"])[0]
                            > get_center_of_bbox(value2["bbox"])[0]
                        ):
                            if self.findRL(value1["cls_name"]) == "R":
                                if value1["cls"] == 1 or value1["cls"] == 17:
                                    value2["cls"] -= 1
                                else:
                                    if value2["cls"] == 12:
                                        value2["cls"] += 3
                                    else:
                                        value2["cls"] += 1
                            else:
                                if value1["cls"] == 0 or value1["cls"] == 16:
                                    value2["cls"] += 1
                                else:
                                    if value1["cls"] == 15:
                                        value1["cls"] -= 3
                                    else:
                                        value1["cls"] -= 1
                            processed_keys.append(key2)
            output_detection_yard.append(frame)
        return output_detection_yard

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        yard_detections = []

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                yard_detections = pickle.load(f)
            return yard_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            yard_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(yard_detections, f)

        return yard_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        yard_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            border = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            yard = {}
            yard["bbox"] = border
            yard["cls"] = object_cls_id
            yard["cls_name"] = id_name_dict[object_cls_id]
            yard_dict[track_id] = yard

        return yard_dict

    def draw_bboxes(self, video_frames, yard_detections):
        output_video_frames = []
        for frame, yard_dict in zip(video_frames, yard_detections):
            # Draw Bounding Boxes
            # Đỏ
            for track_id, info in yard_dict.items():
                id = info["cls"]
                x1, y1, x2, y2 = info["bbox"]
                cv2.putText(
                    frame,
                    f"{id}",
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                cv2.circle(
                    frame, (get_center_of_bbox(info["bbox"])), 5, (0, 0, 255), -1
                )
                # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            output_video_frames.append(frame)

        return output_video_frames

    def get_center_point_yard_frames(self, frames):
        out_center_point_yard_frames = []
        for frame in frames:
            sorted_data = dict(sorted(frame.items(), key=lambda item: item[1]["cls"]))
            center_yard_frame = [0] * 56
            for i in sorted_data:
                index = int(sorted_data[i]["cls"])
                center_yard_frame[index * 2], center_yard_frame[index * 2 + 1] = (
                    get_center_of_bbox(sorted_data[i]["bbox"])
                )
            out_center_point_yard_frames.append(center_yard_frame)

        return out_center_point_yard_frames
