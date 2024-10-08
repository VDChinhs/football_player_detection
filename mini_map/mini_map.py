import cv2
import sys
from sklearn.metrics import mean_squared_error
import constants
import numpy as np
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_center_of_bbox,
    get_closest_keypoint_index,
    measure_distance,
    get_height_of_bbox,
    measure_xy_distance,
)
sys.path.append("../")


class MiniMap:
    def __init__(self, frame):
        self.drawing_rectangle_width = 340
        self.drawing_rectangle_height = 490
        self.buffer = 50
        self.padding_map = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_map_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(
            meters, constants.RONG_SAN, self.court_drawing_width
        )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0] * 56

        # point 0
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(
            self.court_start_y
        )
        # point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(
            self.court_start_y
        )
        # point 16
        drawing_key_points[32] = int(self.court_start_x)
        drawing_key_points[33] = self.court_start_y + self.convert_meters_to_pixels(
            constants.NUA_SAN_DAI * 2
        )
        # point 17
        drawing_key_points[34] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[35] = drawing_key_points[33]
        # point 9
        drawing_key_points[18] = drawing_key_points[0] + self.convert_meters_to_pixels(
            constants.GOC_CAM
        )
        drawing_key_points[19] = drawing_key_points[1]
        # point 8
        drawing_key_points[16] = drawing_key_points[2] - self.convert_meters_to_pixels(
            constants.GOC_CAM
        )
        drawing_key_points[17] = drawing_key_points[3]
        # point 25
        drawing_key_points[50] = drawing_key_points[32] + self.convert_meters_to_pixels(
            constants.GOC_CAM
        )
        drawing_key_points[51] = drawing_key_points[33]
        # point 24
        drawing_key_points[48] = drawing_key_points[34] - self.convert_meters_to_pixels(
            constants.GOC_CAM
        )
        drawing_key_points[49] = drawing_key_points[35]
        # point 5
        drawing_key_points[10] = drawing_key_points[18] + self.convert_meters_to_pixels(
            constants.CAM_GOAL
        )
        drawing_key_points[11] = drawing_key_points[1]
        # point 4
        drawing_key_points[8] = drawing_key_points[16] - self.convert_meters_to_pixels(
            constants.CAM_GOAL
        )
        drawing_key_points[9] = drawing_key_points[1]
        # point 21
        drawing_key_points[42] = drawing_key_points[50] + self.convert_meters_to_pixels(
            constants.CAM_GOAL
        )
        drawing_key_points[43] = drawing_key_points[33]
        # point 20
        drawing_key_points[40] = drawing_key_points[48] - self.convert_meters_to_pixels(
            constants.CAM_GOAL
        )
        drawing_key_points[41] = drawing_key_points[33]
        # point 7
        drawing_key_points[14] = drawing_key_points[18]
        drawing_key_points[15] = drawing_key_points[19] + self.convert_meters_to_pixels(
            constants.CAM_DAI
        )
        # point 6
        drawing_key_points[12] = drawing_key_points[16]
        drawing_key_points[13] = drawing_key_points[17] + self.convert_meters_to_pixels(
            constants.CAM_DAI
        )
        # point 23
        drawing_key_points[46] = drawing_key_points[50]
        drawing_key_points[47] = drawing_key_points[51] - self.convert_meters_to_pixels(
            constants.CAM_DAI
        )
        # point 22
        drawing_key_points[44] = drawing_key_points[48]
        drawing_key_points[45] = drawing_key_points[49] - self.convert_meters_to_pixels(
            constants.CAM_DAI
        )
        # point 3
        drawing_key_points[6] = drawing_key_points[10]
        drawing_key_points[7] = drawing_key_points[11] + self.convert_meters_to_pixels(
            constants.GOAL_DAI
        )
        # point 2
        drawing_key_points[4] = drawing_key_points[8]
        drawing_key_points[5] = drawing_key_points[9] + self.convert_meters_to_pixels(
            constants.GOAL_DAI
        )
        # point 19
        drawing_key_points[38] = drawing_key_points[10]
        drawing_key_points[39] = drawing_key_points[43] - self.convert_meters_to_pixels(
            constants.GOAL_DAI
        )
        # point 18
        drawing_key_points[36] = drawing_key_points[8]
        drawing_key_points[37] = drawing_key_points[41] - self.convert_meters_to_pixels(
            constants.GOAL_DAI
        )
        # point 15
        drawing_key_points[30] = drawing_key_points[0]
        drawing_key_points[31] = drawing_key_points[1] + self.convert_meters_to_pixels(
            constants.NUA_SAN_DAI
        )
        # point 12
        drawing_key_points[24] = drawing_key_points[2]
        drawing_key_points[25] = drawing_key_points[3] + self.convert_meters_to_pixels(
            constants.NUA_SAN_DAI
        )
        # point 14
        drawing_key_points[28] = drawing_key_points[0] + self.convert_meters_to_pixels(
            constants.MID
        )
        drawing_key_points[29] = drawing_key_points[31]
        # point 13
        drawing_key_points[26] = drawing_key_points[2] - self.convert_meters_to_pixels(
            constants.MID
        )
        drawing_key_points[27] = drawing_key_points[31]
        # point 11
        drawing_key_points[22] = drawing_key_points[14] + self.convert_meters_to_pixels(
            constants.BAU_DUC
        )
        drawing_key_points[23] = drawing_key_points[13]
        # point 10
        drawing_key_points[20] = drawing_key_points[12] - self.convert_meters_to_pixels(
            constants.BAU_DUC
        )
        drawing_key_points[21] = drawing_key_points[13]
        # point 27
        drawing_key_points[54] = drawing_key_points[22]
        drawing_key_points[55] = drawing_key_points[47]
        # point 26
        drawing_key_points[52] = drawing_key_points[20]
        drawing_key_points[53] = drawing_key_points[45]

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        self.lines = [
            # dọc
            (0, 16),
            (1, 17),
            (5, 3),
            (4, 2),
            (21, 19),
            (20, 18),
            (9, 7),
            (8, 6),
            (25, 23),
            (24, 22),
            # ngang
            (0, 1),
            (16, 17),
            (7, 6),
            (23, 22),
            (3, 2),
            (19, 18),
        ]

    def set_mini_map_position(self):
        self.court_start_x = self.start_x + self.padding_map
        self.court_start_y = self.start_y + self.padding_map
        self.court_end_x = self.end_x - self.padding_map
        self.court_end_y = self.end_y - self.padding_map
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        # draw Lines
        for line in self.lines:
            start_point = (
                int(self.drawing_key_points[line[0] * 2]),
                int(self.drawing_key_points[line[0] * 2 + 1]),
            )
            end_point = (
                int(self.drawing_key_points[line[1] * 2]),
                int(self.drawing_key_points[line[1] * 2 + 1]),
            )
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (
            self.drawing_key_points[0],
            int((self.drawing_key_points[1] + self.drawing_key_points[33]) / 2),
        )
        net_end_point = (
            self.drawing_key_points[2],
            int((self.drawing_key_points[1] + self.drawing_key_points[33]) / 2),
        )
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # Draw the rectangle
        cv2.rectangle(
            shapes,
            (self.start_x, self.start_y),
            (self.end_x, self.end_y),
            (255, 255, 255),
            cv2.FILLED,
        )
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_mini_court_coordinates(
        self,
        object_position,
        closest_key_point,
        closest_key_point_index,
        player_height_in_pixels,
        player_height_in_meters,
    ):

        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = (
            measure_xy_distance(object_position, closest_key_point)
        )

        # Conver pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(
            distance_from_keypoint_x_pixels,
            player_height_in_meters,
            player_height_in_pixels,
        )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(
            distance_from_keypoint_y_pixels,
            player_height_in_meters,
            player_height_in_pixels,
        )

        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(
            distance_from_keypoint_x_meters
        )
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(
            distance_from_keypoint_y_meters
        )
        closest_mini_coourt_keypoint = (
            self.drawing_key_points[closest_key_point_index * 2],
            self.drawing_key_points[closest_key_point_index * 2 + 1],
        )

        mini_court_player_position = (
            closest_mini_coourt_keypoint[0] + mini_court_x_distance_pixels,
            closest_mini_coourt_keypoint[1] + mini_court_y_distance_pixels,
        )

        return mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(
        self, player_boxes, ball_boxes, original_court_key_points
    ):

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            output_player_bboxes_dict = []
            output_ball_bboxes_dict = []

            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)

            for player_id, info in player_bbox.items():
                player_obj = {}
                ball_obj = {}
                
                foot_position = get_foot_position(info["bbox"])
                closest_player_id_to_ball = min(
                    player_bbox.keys(),
                    key=lambda x: measure_distance(
                        ball_position, get_center_of_bbox(info["bbox"])
                    ),
                )

                # Get The closest keypoint in pixels
                closest_key_point_index = get_closest_keypoint_index(
                    foot_position,
                    original_court_key_points[frame_num],
                    [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                    ],
                )
                closest_key_point = (
                    original_court_key_points[frame_num][closest_key_point_index * 2],
                    original_court_key_points[frame_num][
                        closest_key_point_index * 2 + 1
                    ],
                )

                # Get Player height in pixels
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                bboxes_heights_in_pixels = [
                    get_height_of_bbox(info["bbox"])
                    for i in range(frame_index_min, frame_index_max)
                ]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(
                    foot_position,
                    closest_key_point,
                    closest_key_point_index,
                    max_player_height_in_pixels,
                    constants.PLAYER_HIGHT,
                )

                player_obj['position'] = mini_court_player_position
                player_obj['color'] = info['team_color']
                output_player_bboxes_dict.append(player_obj)

                if closest_player_id_to_ball == player_id:
                    # Get The closest keypoint in pixels
                    closest_key_point_index = get_closest_keypoint_index(
                        ball_position,
                        original_court_key_points[frame_num],
                        [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                            19,
                            20,
                            21,
                            22,
                            23,
                            24,
                            25,
                            26,
                            27,
                        ],
                    )
                    closest_key_point = (
                        original_court_key_points[frame_num][
                            closest_key_point_index * 2
                        ],
                        original_court_key_points[frame_num][
                            closest_key_point_index * 2 + 1
                        ],
                    )

                    mini_court_player_position = self.get_mini_court_coordinates(
                        ball_position,
                        closest_key_point,
                        closest_key_point_index,
                        max_player_height_in_pixels,
                        constants.PLAYER_HIGHT,
                    )
                    ball_obj['position'] = mini_court_player_position
                    ball_obj['color'] = (0, 255, 255)
                    output_ball_bboxes_dict.append(ball_obj)

                    # output_ball_boxes.append({1: mini_court_player_position})
                output_player_boxes.append(output_player_bboxes_dict)
                output_ball_boxes.append(output_ball_bboxes_dict)
        return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(self, frames, postions):
        for frame_num, frame in enumerate(frames):
            for info in postions[frame_num]:
                x, y = info['position']
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, info['color'], -1)
        return frames
    

    def homography_matrix(self, player_boxes, ball_boxes, original_court_key_points):
        pred_dst_pts = []                                                           
        pred_dst_pos = []      
        index = 0                                                   
        for frame_num, player_bbox in enumerate(player_boxes):

            detected_labels = list(map(lambda x: int(x['cls']), original_court_key_points[frame_num].values())) # Nhãn chữ keypoint
            detected_labels_src_pts = np.array([get_center_of_bbox(obj['bbox']) for obj in original_court_key_points[frame_num].values()]) #Tâm keypoint
            detected_labels_dst_pts = np.array([[self.drawing_key_points[a*2], self.drawing_key_points[a*2+1]] for a in detected_labels]) #Vị trí các điểm trên minimap

            if len(detected_labels) > 3:
                    if frame_num + 1 > 1:
                        common_labels = set(detected_labels_prev) & set(detected_labels)
                        if len(common_labels) > 3:
                            common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]   
                            common_label_idx_curr = [detected_labels.index(i) for i in common_labels]        
                            coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]     
                            coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]          
                            coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)  
                            update_homography = coor_error > 5                                         
                        else:
                            update_homography = True                                                         
                    else:
                        update_homography = True

                    if  update_homography:
                        homog, mask = cv2.findHomography(detected_labels_src_pts,                   
                                                    detected_labels_dst_pts)                  
            if 'homog' in locals():
                detected_labels_prev = detected_labels.copy()                               
                detected_labels_src_pts_prev = detected_labels_src_pts.copy()    

                bboxes_p_c_0 = np.array([obj['bbox']for obj in player_bbox.values()])  
                bboxes_p_c_0_color = np.array([obj['team_color']for obj in player_bbox.values()])  
                bboxes_p_c_2 = np.array([ball_boxes[frame_num][1]]) 
                                    
                detected_ppos_src_pts = bboxes_p_c_0[:,:2]  + np.array([[0]*bboxes_p_c_0.shape[0], bboxes_p_c_0[:,3]/2]).transpose()
                detected_ball_src_pos = bboxes_p_c_2[0,:2] if bboxes_p_c_2.shape[0]>0 else None

                pred_dst_pts_player = []
                for nb,pt in enumerate(detected_ppos_src_pts):    
                    player_obj = {}                                      
                    pt = np.append(np.array(pt), np.array([1]), axis=0)                     
                    dest_point = np.matmul(homog, np.transpose(pt))                              
                    dest_point = dest_point/dest_point[2]   
                    player_obj['position'] = tuple(np.transpose(dest_point)[:2])                   
                    player_obj['color'] = bboxes_p_c_0_color[nb]                   
                    pred_dst_pts_player.append(player_obj) 
                # while frame_num == 69 and index == 0:
                #     height, width, channels = video.shape
                #     img1_warped = cv2.warpPerspective(video, homog, (width, height))
                #     img1_warped = self.draw_points_on_mini_court([img1_warped], [pred_dst_pts_player])
                #     cv2.imwrite(f'output_videos/hog1.jpg',img1_warped[0])
                #     index = index + 1 
                pred_dst_pts.append(pred_dst_pts_player)
                
                pred_dst_pts_ball = []
                if detected_ball_src_pos is not None:
                    ball_obj = {}                                      
                    pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
                    dest_point = np.matmul(homog, np.transpose(pt))
                    dest_point = dest_point/dest_point[2]
                    ball_obj['position'] = tuple(np.transpose(dest_point)[:2])                   
                    ball_obj['color'] = (0, 255, 255)
                    pred_dst_pts_ball.append(ball_obj)
                pred_dst_pos.append(pred_dst_pts_ball)

                
        return pred_dst_pts, pred_dst_pos 
 
