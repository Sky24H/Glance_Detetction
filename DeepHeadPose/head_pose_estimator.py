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

def initialize_model():
    snapshot_path = os.path.join(os.path.dirname(__file__), './hopenet_robust_alpha1.pkl')
    cudnn.enabled = True
    gpu = 0

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    model.cuda(gpu)

    # print ('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    transformations = transforms.Compose([transforms.Resize([224, 224]),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    # print ('model initialized.')

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

    yaw_predicted = F.softmax(yaw, dim=1)
    pitch_predicted = F.softmax(pitch, dim=1)
    roll_predicted = F.softmax(roll, dim=1)
    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    return yaw_predicted, pitch_predicted, roll_predicted
