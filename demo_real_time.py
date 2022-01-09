import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './DeepHeadPose'))
from head_pose_estimator import initialize_model, head_pose_estimation
from DeepHeadPose.utils import draw_axis

sys.path.append(os.path.join(os.path.dirname(__file__),'./GazeTracking'))
from Gaze_Traking import init_gaze_tracking, track_gaze

import cv2
from util import initial_face_detection_model, get_coords, get_face, smooth_coords, get_faces_fast
import numpy as np
import torch
import dlib
from skimage import img_as_ubyte
import imageio


if __name__ == '__main__':
    # initialize head pose estimation model
    head_model, transformations = initialize_model()

    # initialize gaze tracking model
    gaze_model = init_gaze_tracking()

    # initialize face detection model
    face_detector = initial_face_detection_model()
    dlib_face_detector = dlib.get_frontal_face_detector()

    # Set max frames
    use_camera = False

    # Set up video parameters, just for testing
    video_path = 0 if use_camera else '../test.mp4'
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_FPS, 25)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)

    # set window
    if use_camera:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        max_frames = 10000000
    else:
        max_frames=600
    # get frame and detect head pose
    frame_num = 0
    last_coord = None
    looking_flag = False
    output_frames = []

    while frame_num < max_frames:
        try:
            _, frame = capture.read()
            if frame is not None:
                frame = cv2.flip(frame, 1)
                # frame = cv2.vconcat([frame, frame])
                # faces = get_faces_fast(dlib_face_detector, frame)
                try:
                    coords_ = get_coords(face_detector, frame)
                    if last_coord is None:
                        last_coord = coords_
                    coords = smooth_coords(last_coord, coords_)
                    face_ori, current_size = get_face(coords, frame)
                    head_results = head_pose_estimation(head_model, transformations, face_ori)
                    gaze_res = False
                    for _ in range(5):
                        gaze_results, face = track_gaze(gaze_model, face_ori, annotate=True)
                        if gaze_results == 'Looking center':
                            gaze_res = True
                            face_annotated = face
                            frame[coords[0]:coords[1], coords[2]:coords[3]] = face_annotated
                    yaw_predicted, pitch_predicted, roll_predicted = head_results
                    if torch.abs(yaw_predicted) < 45 and torch.abs(pitch_predicted) < 30 and gaze_res:
                        looking_flag = True
                        image = cv2.rectangle(frame, (coords[2], coords[0]),  (coords[3], coords[1]), (0,255,0), 2)
                    else:
                        looking_flag = False
                    # print(looking_flag)
                    frame_num += 1
                    draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (coords[2] + coords[3]) / 2, tdy= (coords[0] + coords[1]) / 2, size = (coords[1] - coords[0])/2)
                    last_coord = coords
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(e)
                    continue
                if use_camera:
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)
                else:
                    output_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print('exit', e)
            break

    # relase resources
    cv2.destroyAllWindows()
    capture.release()
    if not use_camera:
        imageio.mimsave('test.mp4', [img_as_ubyte(frame) for frame in output_frames], fps=25)
