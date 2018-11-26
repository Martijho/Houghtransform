from houghspace import HoughSpace
from util import *
from time import time
import cv2
from scipy.misc import imsave


original = cv2.imread('images/original/test.jpg')

size = (200, 200)
original = cv2.resize(original, size)

HoughSpace(original)
