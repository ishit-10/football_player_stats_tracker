from video_utils import read_video, save_video
from tracker import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # Read Video
    video_frames = read_video('input videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('yolov8n.pt')  # smaller model first to test

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,  # Set to False to force new detection
                                       stub_path='stubs/track_stubs.pkl')
    
    # Check if any players were detected
    if not any(tracks['players']):
        print("No players were detected in the video. Please check if the video contains football players.")
        return

    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                        read_from_stub=False,  # Changed to False to force new estimation
                                                                        stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    try:
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    except Exception as e:
        print(f"Warning: Could not assign team colors: {str(e)}")
        print("Using default team colors")
        # Set default team colors
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                track['team'] = 1 if player_id % 2 == 0 else 2
                track['team_color'] = (0,0,255) if track['team'] == 1 else (255,0,0)
    else:
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                     track['bbox'],
                                                     player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    last_team = 1  # Default to team 1 if no ball possession is detected
    frames_since_last_ball = 0
    max_frames_without_ball = 30  # Maximum frames to keep last known team
    consecutive_unknown_frames = 0
    max_consecutive_unknown = 10  # Maximum frames to keep unknown state
    
    # Ensure tracks length matches video frames length
    num_frames = len(video_frames)
    if len(tracks['ball']) < num_frames:
        # Pad tracks with empty frames if needed
        while len(tracks['ball']) < num_frames:
            tracks['ball'].append({})
    
    for frame_num, player_track in enumerate(tracks['players']):
        # Safely get ball bbox with bounds checking
        ball_bbox = None
        if frame_num < len(tracks['ball']) and 1 in tracks['ball'][frame_num]:
            ball_bbox = tracks['ball'][frame_num][1].get('bbox')
        
        if ball_bbox is not None and len(ball_bbox) == 4:  # Ensure valid bbox
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                last_team = tracks['players'][frame_num][assigned_player]['team']
                team_ball_control.append(last_team)
                frames_since_last_ball = 0
                consecutive_unknown_frames = 0
            else:
                frames_since_last_ball += 1
                if frames_since_last_ball > max_frames_without_ball:
                    consecutive_unknown_frames += 1
                    if consecutive_unknown_frames > max_consecutive_unknown:
                        team_ball_control.append(0)  # Unknown possession
                    else:
                        team_ball_control.append(last_team)
                else:
                    team_ball_control.append(last_team)
        else:
            frames_since_last_ball += 1
            if frames_since_last_ball > max_frames_without_ball:
                consecutive_unknown_frames += 1
                if consecutive_unknown_frames > max_consecutive_unknown:
                    team_ball_control.append(0)  # Unknown possession
                else:
                    team_ball_control.append(last_team)
            else:
                team_ball_control.append(last_team)
    
    team_ball_control = np.array(team_ball_control)

    # Draw output with error handling
    try:
        # Draw object Tracks
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

        # Draw Camera movement
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

        # Draw Speed and Distance
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

        # Save video
        save_video(output_video_frames, 'output_videos/output_video.avi')
        print("Video processing completed successfully!")
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        print("Attempting to save partial results...")
        try:
            save_video(output_video_frames, 'output_videos/output_video_partial.avi')
            print("Partial results saved successfully.")
        except Exception as save_error:
            print(f"Failed to save partial results: {str(save_error)}")

if __name__ == '__main__':
    main()