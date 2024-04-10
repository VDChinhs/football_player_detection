from utils import read_video,save_video
from trackers import PlayerTracker,BallTracker, YardTracker
import cv2
from mini_map import MiniMap

def main():
    print("Start")
    
    #Read Video
    input_video_path = "input_videos/women.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    print("Detection....")
    player_tracker = PlayerTracker(model_path='models/best_predict.pt')
    ball_tracker = BallTracker(model_path='models/player_predict_best.pt')
    yard_tracker = YardTracker(model_path='models/keypoint_yard_best.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_women.pkl"
                                                     )
    
    
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/ball_women.pkl"
                                                     )
    
    yard_detections = yard_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/yard_women.pkl"
                                                     )
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)


    yard_custom = [1550, 70, 1850, 70, 1550, 510.0, 1850, 510.0, 1620.0, 70, 1780.0, 70, 1620.0, 510.0, 1780.0, 510.0, 1664.0, 70, 1736.0, 70, 1664.0, 510.0, 1736.0, 510.0, 1620.0, 136.0, 1780.0, 136.0, 1620.0, 444.0, 1780.0, 444.0, 1664.0, 92.0, 1736.0, 92.0, 1664.0, 488.0, 1736.0, 488.0, 1550, 290.0, 1850, 290.0, 487.0, 494.0, 1122.0, 489.0, 619.0, 236.0, 992.0, 234.0, 1668.0, 444.0, 1732.0, 444.0]

    mini_map = MiniMap(video_frames[0])

    # Convert position to minimap
    player_mini_map_dectection, ball_mini_map_dectection = mini_map.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          yard_custom)
    
    print("DrawBboxes...")
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames= ball_tracker.draw_bboxes(video_frames, ball_detections)
    output_video_frames= yard_tracker.draw_bboxes(video_frames, yard_detections)

    output_video_frames = mini_map.draw_points_on_mini_court(output_video_frames,player_mini_map_dectection)
    output_video_frames = mini_map.draw_points_on_mini_court(output_video_frames,ball_mini_map_dectection, color=(0,255,255))  

    # Draw MiniMap
    output_video_frames = mini_map.draw_mini_court(output_video_frames)

    #Frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    save_video(output_video_frames, "output_videos/output_video_women.avi")
    print("Finish...")
    
if __name__ == "__main__":
    main()