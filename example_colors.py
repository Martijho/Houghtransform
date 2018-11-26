from houghspace import HoughSpace
from util import *
from time import time
import cv2
from scipy.misc import imsave
import numpy as np
from matplotlib import pyplot as plt

# Edit a endge image into segments e.g: object 1 and object 2
obj1 = cv2.imread('images/edge/segments/obj1.png')
obj2 = cv2.imread('images/edge/segments/obj2.png')
background = cv2.imread('images/edge/test.png') - obj1 - obj2

background_hough = HoughSpace(background, save=False).hough
obj1_hough = HoughSpace(obj1, save=False).hough
obj2_hough = HoughSpace(obj2, save=False).hough


hough_color = np.array(background_hough)
hough_color[:, :, 0][...] = obj1_hough[:, :, 0]
hough_color[:, :, 1][...] = obj2_hough[:, :, 0]


imsave('images/hough/color.png', hough_color)

plt.subplot(221)
plt.imshow(background_hough/255)
plt.title('Background')

plt.subplot(222)
plt.imshow(hough_color/255)
plt.title('Hough color')

plt.subplot(223)
plt.imshow(obj1_hough/255)
plt.title('Object 1')

plt.subplot(224)
plt.imshow(obj2_hough/255)
plt.title('Object 2')

plt.show()
