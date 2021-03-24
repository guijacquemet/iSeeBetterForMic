from __future__ import print_function
import argparse

import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rbpn import Net as RBPN
from data import get_test_set
import numpy as np
import utils
import time
import cv2
import math
import logger

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('-m', '--model', default="weights/netG_epoch_4_1.pth", help="Model")
parser.add_argument('-o', '--output', default='Results/', help="Location to save test results")
parser.add_argument('--data_dir', type=str, default="./Vid4")
parser.add_argument('--file_list', type=str, default="foliage_test.txt")
parser.add_argument('--nFrames', type=int, default=7, help="")