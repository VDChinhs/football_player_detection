from utils import read_video,save_video
from trackers import PlayerTracker,BallTracker

def main():
    print("Hello World")
    #Read Video
    input_video_path = "input_videos/bar_realm.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8m.pt')
    # ball_tracker = BallTracker(model_path='models/yolo5_last.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                    #  stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    
    save_video(output_video_frames, "output_videos/output_video.avi")
    
if __name__ == "__main__":
    main()