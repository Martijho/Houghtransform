import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

def scale(image, scale):
    h, w, _ = image.shape
    image = cv2.resize(image, (w//scale, h//scale))
    return image

def load(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def edge_detection(image, sigma=.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return cv2.Canny(image, lower, upper)

def grayscale(image):
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show(image, block=True):

    matplotlib.rc('figure', figsize=(8, 6))
    plt.figure()
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show(block=block)

def invert(image):
    return (image-255)*(255)
