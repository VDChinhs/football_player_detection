import cv2
import time


def read_video(video_path):
    new_timeframe = 0
    pre_timeframe = 0

    frames = {
        "video_frame": [],
        "FPS": [],
    }

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames["video_frame"].append(frame)

        new_timeframe = time.time()
        fps = 1 / (new_timeframe - pre_timeframe)
        pre_timeframe = new_timeframe
        fps = int(fps)
        frames["FPS"].append(fps)

    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()
