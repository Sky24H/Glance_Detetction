import os
import sys
sys.path.append('./DeepHeadPose')
from head_pose_estimator import initialize_model, head_pose_estimation

sys.path.append('./GazeTracking')
from Gaze_Traking import init_gaze_tracking, track_gaze

import cv2
from util import initial_face_detection_model, get_coords, get_face




if __name__ == '__main__':
    # initialize head pose estimation model 
    head_model, transformations = initialize_model()

    # initialize gaze tracking model
    gaze_model = init_gaze_tracking()

    # initialize face detection model
    face_detector = initial_face_detection_model()
    # Set max frames
    max_frames = 100

    # Set up video parameters, just for testing
    video_path = '../base.mp4'
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_FPS, 25)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)

    # get frame and detect head pose
    frame_num = 0
    while frame_num < max_frames:
        try:
            frame_num = 1
            _, frame = capture.read()
            if frame is not None:
                frame = cv2.flip(frame, 1)
                coords = get_coords(face_detector, frame)
                face, current_size = get_face(coords, frame)

                head_results = head_pose_estimation(head_model, transformations, face)
                gaze_results = track_gaze(gaze_model, face)
                yaw_predicted, pitch_predicted, roll_predicted = head_results
                print(head_results, gaze_results)
                frame_num += 1
            else:
                break
        except Exception as e:
            print(e)
            break
    capture.release()