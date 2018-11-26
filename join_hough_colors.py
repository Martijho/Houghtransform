from houghspace import HoughSpace
from util import *
from time import time
import cv2
from scipy.misc import imsave
from matplotlib import pyplot as plt

r = np.array([255, 0, 0])
g = np.array([0, 255, 0])
b = np.array([0, 0, 255])

def img_to_color(img):
    red = np.array(img)
    green = np.array(img)
    blue = np.array(img)

    red[:, :, 1:] *= 0
    green[:, :, 0] *= 0
    green[:, :, 2] *= 0

    blue[:, :, :2] *= 0

    # red green blue, white, teal, pink, yellow
    return red, green, blue, red+green+blue, green+blue, red+blue, blue+green

def add(a, b, c):
    background = np.array(a)


rmamma, gmamma, bmamma, wmamma, tmamma, pmamma, ymamma = img_to_color(cv2.imread('mamma_separated_hough_large.png'))
rpappa, gpappa, bpappa, wpappa, tpappa, ppappa, ypappa = img_to_color(cv2.imread('pappa_separated_hough_large.png'))
rjohannes, gjohannes, bjohannes, wjohannes, tjohannes, pjohannes, yjohannes = img_to_color(cv2.imread('johannes_separated_hough_large.png'))
rmeg, gmeg, bmeg, wmeg, tmeg, pmeg, ymeg = img_to_color(cv2.imread('meg_separated_hough_large.png'))

'''
plt.subplot(221)
plt.imshow(mamma, cmap='gray')
plt.subplot(222)
plt.imshow(pappa, cmap='gray')
plt.subplot(223)
plt.imshow(johannes, cmap='gray')
plt.subplot(224)
plt.imshow(meg, cmap='gray')
plt.show()
'''


comb = [
    bmeg + np.flip(tjohannes, axis=1) + np.flip(wpappa, axis=0), #Til mamma
    rmeg + np.flip(bjohannes, axis=0) + np.flip(wmamma, axis=1), #Til pappa
    gmeg + np.flip(wmamma, axis=0) + np.flip(ypappa, axis=1), #Til johannes

    np.flip(wmeg, axis=1) + bjohannes + np.flip(gpappa, axis=0),  # Til mamma
    np.flip(ymeg, axis=0) + np.flip(rjohannes, axis=1) + wmamma,  # Til pappa
    np.flip(rmeg, axis=1) + np.flip(gmamma, axis=0) + bpappa,  # Til johannes

    rmeg+gmeg+bmeg + np.flip(bjohannes, axis=1) + np.flip(gpappa, axis=0),  # Til mamma
    gmeg+bmeg+rmeg + np.flip(rjohannes, axis=0) + np.flip(bmamma, axis=1),  # Til pappa
    bmeg+rmeg+gmeg + np.flip(rmamma, axis=0) + np.flip(bpappa, axis=1),  # Til johannes
]

for i in range(9):
    imsave(f'images/test_output/test{i}.png', comb[i])

'''
for i in range(9):
    plt.subplot(331+i)
    plt.imshow(comb[i])

plt.show()


'''