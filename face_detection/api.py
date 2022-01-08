from __future__ import print_function
import os
import torch
from torch.utils.model_zoo import load_url
from enum import Enum
import numpy as np
import cv2
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value

ROOT = os.path.dirname(os.path.abspath(__file__))

class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__('face_detection.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)
            
            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))
        return results

    def get_landmarks_from_image(self, image_or_path, detected_faces=None):
        """Predict the landmarks for each face present in the image.
        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.
         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.
        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        image = get_image(image_or_path)

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_image(image.copy())

        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            return None

        landmarks = []
        for i, d in enumerate(detected_faces):
            center = torch.tensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()

            inp = inp.to(self.device)
            inp.div_(255.0).unsqueeze_(0)

            out = self.face_alignment_net(inp).detach()
            if self.flip_input:
                out += flip(self.face_alignment_net(flip(inp)).detach(), is_label=True)
            out = out.cpu().numpy()

            pts, pts_img = get_preds_fromhm(out, center.numpy(), scale)
            pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

            if self.landmarks_type == LandmarksType._3D:
                heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                for i in range(68):
                    if pts[i, 0] > 0 and pts[i, 1] > 0:
                        heatmaps[i] = draw_gaussian(
                            heatmaps[i], pts[i], 2)
                heatmaps = torch.from_numpy(
                    heatmaps).unsqueeze_(0)

                heatmaps = heatmaps.to(self.device)
                depth_pred = self.depth_prediciton_net(
                    torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
                pts_img = torch.cat(
                    (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

            landmarks.append(pts_img.numpy())

        return landmarks

    def get_landmarks_from_batch(self, image_batch, detected_faces=None):
        """Predict the landmarks for each face present in the image.
        This function predicts a set of 68 2D or 3D images, one for each image in a batch in parallel.
        If detect_faces is None the method will also run a face detector.
         Arguments:
            image_batch {torch.tensor} -- The input images batch
        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_batch(image_batch)

        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            return None

        landmarks = []
        # A batch for each frame
        for i, faces in enumerate(detected_faces):
            landmark_set = self.get_landmarks_from_image(
                image_batch[i].cpu().numpy().transpose(1, 2, 0),
                detected_faces=faces
            )
            # Bacward compatibility
            if landmark_set is not None:
                landmark_set = np.concatenate(landmark_set, axis=0)
            else:
                landmark_set = []
            landmarks.append(landmark_set)
        return landmarks