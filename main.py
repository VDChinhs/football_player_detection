from utils import read_video, save_video
from trackers import PlayerTracker, YardTracker
from team_assigner import TeamAssigner
import cv2
from mini_map import MiniMap


def main():
    print("Read Video...")

    # Read Video
    keyword = "women"

    input_video_path = f"input_videos/{keyword}.mp4"
    # stub_path_tracker = f"tracker_stubs/test_man.pkl"
    # stub_path_yard = f"tracker_stubs/test_yard_man.pkl"
    # output_video_path = f"output_videos/test_man.avi"
    stub_path_tracker = f"tracker_stubs/PJ_tracker_{keyword}.pkl"
    stub_path_yard = f"tracker_stubs/PJ_yard_{keyword}.pkl"
    output_video_path = f"output_videos/test_{keyword}.avi"

    video_frames = read_video(input_video_path)

    # Detection Players and Ball
    print("Detection Players, Ball and Keypoint Yard...")
    tracker = PlayerTracker(model_path="models/YOLOv8xEp50.pt")
    yard_tracker = YardTracker(model_path="models/keypoint_yard_best.pt")
    mini_map = MiniMap(video_frames["video_frame"][0])

    tracker_detections = tracker.get_object_tracks(
        video_frames["video_frame"],
        read_from_stub=True,
        stub_path=stub_path_tracker,
    )
    tracker_detections["ball"] = tracker.interpolate_ball_positions(
        tracker_detections["ball"]
    )

    yard_detections = yard_tracker.detect_frames(
        video_frames["video_frame"],
        read_from_stub=True,
        stub_path=stub_path_yard,
    )
    yard_detections = yard_tracker.clean_data(yard_detections)

    # Assign Player Teams
    print('Assign player teams...')
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(
        video_frames["video_frame"][0], tracker_detections["players"][0]
    )
    for frame_num, player_track in enumerate(tracker_detections["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames["video_frame"][frame_num], track["bbox"], player_id
            )
            tracker_detections["players"][frame_num][player_id]["team"] = team
            tracker_detections["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    # Convert position to minimap
    print('Convert position to minimap...')
    # player_mini_map_dectection, ball_mini_map_dectection = (
    #     mini_map.convert_bounding_boxes_to_mini_court_coordinates(
    #         tracker_detections["players"], tracker_detections["ball"], center_yard
    #     )
    # )
    player_mini_map_dectection, ball_mini_map_dectection = mini_map.homography_matrix(tracker_detections["players"], tracker_detections["ball"], yard_detections)


    print("DrawBboxes...")
    output_video_frames = tracker.draw_bboxes(
        video_frames["video_frame"], tracker_detections
    )
    output_video_frames = yard_tracker.draw_bboxes(output_video_frames, yard_detections)

    print("MiniMap...")
    output_video_frames = mini_map.draw_mini_court(output_video_frames)

    output_video_frames = mini_map.draw_points_on_mini_court(
        output_video_frames, player_mini_map_dectection
    )
    output_video_frames = mini_map.draw_points_on_mini_court(
        output_video_frames, ball_mini_map_dectection
    )

    # Frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(
            frame,
            f"Frame: {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"FPS: {video_frames['FPS'][i]}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    # Save Video
    save_video(output_video_frames, output_video_path)
    print("Finish...")

if __name__ == "__main__":
    main()
