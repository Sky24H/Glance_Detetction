import numpy as np
import cv2, os, sys, glob
import face_detection
device = "cuda"
from tqdm import tqdm
import time
import dlib
from imutils import face_utils
import cv2


def initial_face_detection_model():
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    return detector

def get_coords(detector, image, coordinates=None):
    if coordinates is not None:
        coords = coordinates
        face = image[coords[0]:coords[1], coords[2]:coords[3]]
        rect = detector.get_detections_for_batch(np.array([face]))[0]
        if rect is None:
            # check this frame where the face was not detected.
            # cv2.imwrite('temp/faulty_frame.jpg', image)
            print('Face not detected! Ensure the video contains a face in all the frames.')
            return []
        return face
    else:
        rect = detector.get_detections_for_batch(np.array([image]))[0]
        if rect is None:
            # check this frame where the face was not detected.
            # cv2.imwrite('temp/faulty_frame.jpg', image)
            print('Face not detected! Ensure the video contains a face in all the frames.')
            return []
        y1 = max(0, rect[1])
        y2 = min(image.shape[0], rect[3])
        x1 = max(0, rect[0])
        x2 = min(image.shape[1], rect[2])
        # y_gap, x_gap = (y2 - y1)//2, (x2 - x1)//2
        # y_gap = min((y2 - y1)//6, y1, image.shape[0] - y2)
        # x_gap = (((y2-y1)+y_gap*2) - (x2-x1))//2
        # coords = y1-y_gap, y2+y_gap, x1-x_gap, x2+x_gap
        coords = y1, y2, x1, x2
        #print(coords)
        # coords = [coords_[0], coords_[0]+1024, coords_[2], coords_[2]+1024]
        #results.append(image[y1-y_gap: y2+y_gap, x1-x_gap:x2+x_gap])
        # face = image[coords[0]:coords[1], coords[2]:coords[3]]
        return coords

# def get_face(detector, image, coordinates=None):

#     if coordinates is not None:
#         coords = coordinates
#         face = image[coords[0]:coords[1], coords[2]:coords[3]]
#         rect = detector.get_detections_for_batch(np.array([face]))[0]
#         if rect is None:
#             # check this frame where the face was not detected.
#             # cv2.imwrite('temp/faulty_frame.jpg', image)
#             print('Face not detected! Ensure the video contains a face in all the frames.')
#             return []
#         return face
#     else:
#         rect = detector.get_detections_for_batch(np.array([image]))[0]
#         if rect is None:
#             # check this frame where the face was not detected.
#             # cv2.imwrite('temp/faulty_frame.jpg', image)
#             print('Face not detected! Ensure the video contains a face in all the frames.')
#             return []
#         y1 = max(0, rect[1])
#         y2 = min(image.shape[0], rect[3])
#         x1 = max(0, rect[0])
#         x2 = min(image.shape[1], rect[2])
#         # y_gap, x_gap = (y2 - y1)//2, (x2 - x1)//2
#         y_gap = min((y2 - y1)//4, y1, image.shape[0] - y2)
#         x_gap = (((y2-y1)+y_gap*2) - (x2-x1))//2
#         coords = y1-y_gap, y2+y_gap, x1-x_gap, x2+x_gap
#         #print(coords)
#         # coords = [coords_[0], coords_[0]+1024, coords_[2], coords_[2]+1024]
#         #results.append(image[y1-y_gap: y2+y_gap, x1-x_gap:x2+x_gap])
#         face = image[coords[0]:coords[1], coords[2]:coords[3]]
#         return face, coords

def smooth_coords(last_coord, current_coord):
    change = np.array(current_coord) - np.array(last_coord)
    change = change * 0.8
    return (np.array(last_coord) + np.array(change)).astype(int).tolist()

def get_face(coords, image):
    y1, y2, x1, x2 = coords
    w, h = x2 - x1, y2 - y1
    # center = (x1 + w//2, y1 + h//2)
    size = (w+h)//2

    return image[y1:y2, x1:x2], size

def get_faces_fast(face_detecor, frame):
    landmarks = []
    faces = face_detecor(frame, 1)
    # landmark = face_predictor(frame, faces[0])
    print(faces)

    return faces


# if __name__ == '__main__':
#     # test_image = cv2.imread('./output.png')
#     detector = initil_face_detection_model()

#     all_images = glob.glob('../.driving_test/*.png')
#     all_images.sort()
#     results = []
#     n = 0
#     for path in tqdm(all_images):
#         try:
#             face = get_face(detector, cv2.imread(path))
#             face = cv2.resize(face, (256, 256))
#             # results.append(face)
#             save_path = '../driving_test/'+str(n).zfill(4)+'.png'
#             cv2.imwrite(save_path, face)
#             n+=1
#         except Exception as e:
#             print(e)
#             continue