import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import time
import statistics

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets as datasets, hopenet, utils

from skimage import io
# import dlib


def initialize_model():
    snapshot_path = os.path.join(os.path.dirname(__file__), './hopenet_robust_alpha1.pkl')
    cudnn.enabled = True
    gpu = 0

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    model.cuda(gpu)

    print ('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    print ('model initialized.')

    return model, transformations



def head_pose_estimation(model, transformations, cv2_face):
    gpu = 0
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    # Transform
    img = Image.fromarray(cv2_face)
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).cuda(gpu)

    yaw, pitch, roll = model(img)

    yaw_predicted = F.softmax(yaw)
    pitch_predicted = F.softmax(pitch)
    roll_predicted = F.softmax(roll)
    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    return yaw_predicted, pitch_predicted, roll_predicted

    # # Print new frame with cube and axis
    # txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
    # # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
    # utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
    # # Plot expanded bounding box
    # # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
    # tock_inner = time.time()
    # print(f'inference time for one face detection is {tock_inner-tick_inner}')


# if __name__ == '__main__':
#     # Initialize model
#     model, transformations = initialize_model()
    
#     # Set max frames
#     max_frames = 100

#     # Set up video parameters, just for testing
#     video_path = '../../base.mp4'
#     capture = cv2.VideoCapture(video_path)
#     capture.set(cv2.CAP_PROP_FPS, 25)
#     capture.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
#     capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)

#     # get frame and detect head pose
#     frame_num = 0
#     while frame_num < max_frames:
#         try:
#             frame_num = 1
#             _, frame = capture.read()
#             if frame is not None:
#                 frame = cv2.flip(frame, 1)
#                 results = head_pose_estimation(model, transformations, frame)
#                 yaw_predicted, pitch_predicted, roll_predicted = results
#                 print(yaw_predicted, pitch_predicted, roll_predicted)
#                 frame_num += 1
#             else:
#                 break
#         except Exception as e:
#             print(e)
#             break
#     capture.release()