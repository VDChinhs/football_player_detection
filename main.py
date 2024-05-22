from utils import read_video, save_video
from trackers import PlayerTracker, YardTracker
from team_assigner import TeamAssigner
import cv2
from mini_map import MiniMap


def main():
    print("Start")

    # Read Video
    input_video_path = "input_videos/ManCity-Bayer.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    print("Detection....")
    tracker = PlayerTracker(model_path="models/YOLOv8xEp50.pt")
    yard_tracker = YardTracker(model_path="models/keypoint_yard_best.pt")

    tracker_detections = tracker.get_object_tracks(
        video_frames["video_frame"],
        read_from_stub=True,
        stub_path="tracker_stubs/test_man.pkl",
    )
    tracker_detections["ball"] = tracker.interpolate_ball_positions(
        tracker_detections["ball"]
    )

    yard_detections = yard_tracker.detect_frames(
        video_frames["video_frame"],
        read_from_stub=True,
        stub_path="tracker_stubs/test_yard_man.pkl",
    )

    # Assign Player Teams
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

    yard_custom = [
        1550,
        70,
        1850,
        70,
        1550,
        510.0,
        1850,
        510.0,
        1620.0,
        70,
        1780.0,
        70,
        1620.0,
        510.0,
        1780.0,
        510.0,
        1664.0,
        70,
        1736.0,
        70,
        1664.0,
        510.0,
        1736.0,
        510.0,
        1620.0,
        136.0,
        1780.0,
        136.0,
        1620.0,
        444.0,
        1780.0,
        444.0,
        1664.0,
        92.0,
        1736.0,
        92.0,
        1664.0,
        488.0,
        1736.0,
        488.0,
        1550,
        290.0,
        1850,
        290.0,
        487.0,
        494.0,
        1122.0,
        489.0,
        619.0,
        236.0,
        992.0,
        234.0,
        1668.0,
        444.0,
        1732.0,
        444.0,
    ]

    mini_map = MiniMap(video_frames["video_frame"][0])

    # Convert position to minimap
    player_mini_map_dectection, ball_mini_map_dectection = (
        mini_map.convert_bounding_boxes_to_mini_court_coordinates(
            tracker_detections["players"], tracker_detections["ball"], yard_custom
        )
    )

    print("DrawBboxes...")
    output_video_frames = tracker.draw_bboxes(
        video_frames["video_frame"], tracker_detections
    )
    output_video_frames = yard_tracker.draw_bboxes(output_video_frames, yard_detections)

    output_video_frames = mini_map.draw_points_on_mini_court(
        output_video_frames, player_mini_map_dectection
    )
    output_video_frames = mini_map.draw_points_on_mini_court(
        output_video_frames, ball_mini_map_dectection, color=(0, 255, 255)
    )

    # Draw MiniMap
    output_video_frames = mini_map.draw_mini_court(output_video_frames)

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
    save_video(output_video_frames, "output_videos/test_man.avi")
    print("Finish...")


if __name__ == "__main__":
    main()
