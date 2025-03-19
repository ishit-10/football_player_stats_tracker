from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        # Safely extract ball positions
        ball_positions = []
        for frame_positions in ball_positions:
            if 1 in frame_positions and 'bbox' in frame_positions[1]:
                ball_positions.append(frame_positions[1]['bbox'])
            else:
                ball_positions.append([])  # Empty list for missing detections

        # Convert to DataFrame only if we have valid positions
        if any(ball_positions):
            df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
            # Interpolate missing values
            df_ball_positions = df_ball_positions.interpolate()
            df_ball_positions = df_ball_positions.bfill()
            ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        else:
            ball_positions = [{1: {"bbox": []}} for _ in range(len(ball_positions))]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            print(f"Processing frames {i} to {min(i+batch_size, len(frames))}")
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.05)  # Lower confidence threshold
            detections += detections_batch
            # Print detailed detection information
            for det in detections_batch:
                print(f"Number of detections: {len(det.boxes)}")
                if len(det.boxes) > 0:
                    print("Classes detected:", [det.names[int(box.cls)] for box in det.boxes])
                    print("Confidence scores:", [float(box.conf) for box in det.boxes])
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            print(f"\nFrame {frame_num} class names:", cls_names)
            cls_names_inv = {v:k for k,v in cls_names.items()}
            print("Inverse class mapping:", cls_names_inv)

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            print(f"Number of detections in frame {frame_num}: {len(detection_supervision)}")

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            print(f"Number of tracked objects in frame {frame_num}: {len(detection_with_tracks)}")

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # First pass: Identify potential referees based on position and size
            frame = frames[frame_num]
            frame_height, frame_width = frame.shape[:2]
            
            # Store all person detections for analysis
            person_detections = []
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                # Extract confidence from the detection object properly
                if isinstance(frame_detection[5], dict):
                    confidence = float(frame_detection[5].get('confidence', 0.0))
                else:
                    confidence = float(frame_detection[5])

                # Map COCO classes to our categories
                class_name = cls_names[cls_id]
                print(f"Detected class: {class_name} with ID: {cls_id}")

                # Calculate bbox properties
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = height / width
                center_y = (y1 + y2) / 2
                center_x = (x1 + x2) / 2

                if class_name == 'person':
                    person_detections.append({
                        'bbox': bbox,
                        'track_id': track_id,
                        'width': width,
                        'height': height,
                        'aspect_ratio': aspect_ratio,
                        'center_x': center_x,
                        'center_y': center_y,
                        'confidence': confidence
                    })

            # Analyze person detections to identify referees
            if person_detections:
                # Calculate average player size and position
                avg_width = np.mean([p['width'] for p in person_detections])
                avg_height = np.mean([p['height'] for p in person_detections])
                avg_center_x = np.mean([p['center_x'] for p in person_detections])

                for person in person_detections:
                    # Referee detection based on relative size and position
                    is_referee = False
                    
                    # Size-based criteria
                    size_diff = abs(person['width'] - avg_width) / avg_width
                    height_diff = abs(person['height'] - avg_height) / avg_height
                    
                    # Position-based criteria
                    x_pos_diff = abs(person['center_x'] - avg_center_x) / frame_width
                    
                    # Referee is likely if:
                    # 1. Size is significantly different from average player
                    # 2. Position is away from the main group of players
                    # 3. High confidence in detection
                    if (size_diff > 0.3 or height_diff > 0.3) and \
                       (x_pos_diff > 0.3) and \
                       person['confidence'] > 0.7:
                        is_referee = True
                        tracks["referees"][frame_num][person['track_id']] = {"bbox": person['bbox']}
                    else:
                        tracks["players"][frame_num][person['track_id']] = {"bbox": person['bbox']}

            # Second pass: Identify ball using multiple features
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                # Extract confidence from the detection object properly
                if isinstance(frame_detection[5], dict):
                    confidence = float(frame_detection[5].get('confidence', 0.0))
                else:
                    confidence = float(frame_detection[5])
                class_name = cls_names[cls_id]

                # Calculate bbox properties
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = height / width
                center_y = (y1 + y2) / 2

                # Identify ball based on multiple features
                is_ball = False
                if class_name == 'sports ball':
                    # For sports ball class, use a lower confidence threshold
                    if confidence > 0.5:
                        is_ball = True
                elif class_name == 'person':
                    # For person class, use stricter criteria
                    if (width < frame_width * 0.03 and height < frame_height * 0.03 and 
                        abs(aspect_ratio - 1.0) < 0.1 and confidence > 0.8):
                        # Check if it's near the ground or in the air
                        if y2 > frame_height * 0.8 or (y2 > frame_height * 0.6 and y1 < frame_height * 0.4):
                            # Additional check: should be near players
                            is_near_player = False
                            for player in tracks["players"][frame_num].values():
                                player_bbox = player["bbox"]
                                player_center = get_center_of_bbox(player_bbox)
                                ball_center = get_center_of_bbox(bbox)
                                
                                # Calculate distance between ball and player
                                distance = np.sqrt((player_center[0] - ball_center[0])**2 + 
                                                 (player_center[1] - ball_center[1])**2)
                                
                                # If ball is within 100 pixels of a player
                                if distance < 100:
                                    is_near_player = True
                                    break
                            
                            if is_near_player:
                                is_ball = True

                if is_ball:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

            print(f"Players in frame {frame_num}: {len(tracks['players'][frame_num])}")
            print(f"Referees in frame {frame_num}: {len(tracks['referees'][frame_num])}")
            print(f"Balls in frame {frame_num}: {len(tracks['ball'][frame_num])}")

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        try:
            # Convert bbox coordinates to integers
            bbox = [int(x) for x in bbox]
            y2 = bbox[3]
            x_center, _ = get_center_of_bbox(bbox)
            width = get_bbox_width(bbox)

            # Convert all parameters to integers
            x_center = int(x_center)
            width = int(width)
            height = int(0.35 * width)

            # Ensure coordinates are within frame bounds
            frame_height, frame_width = frame.shape[:2]
            x_center = max(0, min(x_center, frame_width))
            y2 = max(0, min(y2, frame_height))
            width = min(width, frame_width)
            height = min(height, frame_height)

            cv2.ellipse(
                frame,
                center=(x_center, y2),
                axes=(width, height),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=2,
                lineType=cv2.LINE_4
            )

            if track_id is not None:
                rectangle_width = 40
                rectangle_height = 20
                x1_rect = max(0, x_center - rectangle_width//2)
                x2_rect = min(frame_width, x_center + rectangle_width//2)
                y1_rect = max(0, (y2 - rectangle_height//2) + 15)
                y2_rect = min(frame_height, (y2 + rectangle_height//2) + 15)

                cv2.rectangle(frame,
                             (x1_rect, y1_rect),
                             (x2_rect, y2_rect),
                             color,
                             cv2.FILLED)
                
                x1_text = x1_rect + 12
                if track_id > 99:
                    x1_text -= 10
                
                cv2.putText(
                    frame,
                    f"{track_id}",
                    (x1_text, y1_rect + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
        except Exception as e:
            print(f"Warning: Error drawing ellipse: {str(e)}")
        return frame

    def draw_traingle(self,frame,bbox,color):
        try:
            # Convert bbox coordinates to integers
            bbox = [int(x) for x in bbox]
            y = bbox[1]
            x, _ = get_center_of_bbox(bbox)
            x = int(x)  # Ensure x is an integer

            # Ensure coordinates are within frame bounds
            frame_height, frame_width = frame.shape[:2]
            x = max(0, min(x, frame_width))
            y = max(0, min(y, frame_height))

            # Create triangle points with integer coordinates
            triangle_points = np.array([
                [x, y],
                [max(0, x-10), max(0, y-20)],
                [min(frame_width, x+10), max(0, y-20)]
            ], dtype=np.int32)  # Specify dtype as int32 for OpenCV

            # Draw filled triangle
            cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
            # Draw triangle outline
            cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        except Exception as e:
            print(f"Warning: Error drawing triangle: {str(e)}")
        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        try:
            # Draw a semi-transparent rectangle 
            overlay = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(1350, frame_width - 550))
            y1 = max(0, min(850, frame_height - 120))
            x2 = min(frame_width, x1 + 550)
            y2 = min(frame_height, y1 + 120)
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255,255,255), -1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            team_ball_control_till_frame = team_ball_control[:frame_num+1]
            
            # Get the number of times each team had ball control
            team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
            team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
            unknown_frames = team_ball_control_till_frame[team_ball_control_till_frame==0].shape[0]
            
            total_frames = team_1_num_frames + team_2_num_frames + unknown_frames
            if total_frames > 0:
                team_1 = team_1_num_frames/total_frames
                team_2 = team_2_num_frames/total_frames
                unknown = unknown_frames/total_frames
                
                # Adjust text position based on frame size
                text_y = y1 + 50
                cv2.putText(frame, f"Team 1: {team_1*100:.1f}%", (x1 + 50, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
                text_y += 50
                cv2.putText(frame, f"Team 2: {team_2*100:.1f}%", (x1 + 50, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
                if unknown > 0:
                    text_y += 50
                    cv2.putText(frame, f"Unknown: {unknown*100:.1f}%", (x1 + 50, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            else:
                cv2.putText(frame, "No ball possession data", (x1 + 50, y1 + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        except Exception as e:
            print(f"Warning: Error drawing team ball control: {str(e)}")
        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            try:
                frame = frame.copy()

                player_dict = tracks["players"][frame_num]
                ball_dict = tracks["ball"][frame_num]
                referee_dict = tracks["referees"][frame_num]

                # Draw Players
                for track_id, player in player_dict.items():
                    if 'bbox' in player and len(player['bbox']) == 4:
                        color = player.get("team_color",(0,0,255))
                        frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                        if player.get('has_ball',False):
                            frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

                # Draw Referee
                for _, referee in referee_dict.items():
                    if 'bbox' in referee and len(referee['bbox']) == 4:
                        frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
                
                # Draw ball 
                for track_id, ball in ball_dict.items():
                    if 'bbox' in ball and len(ball['bbox']) == 4:
                        frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

                # Draw Team Ball Control
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

                output_video_frames.append(frame)
            except Exception as e:
                print(f"Warning: Error processing frame {frame_num}: {str(e)}")
                output_video_frames.append(frame)  # Add original frame if processing fails

        return output_video_frames