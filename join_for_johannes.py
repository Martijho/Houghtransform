from houghspace import HoughSpace
from util import *
from time import time
import cv2
from scipy.misc import imsave

prefix = 'alle'
pappa = cv2.imread('images/hough/pappa.png')
mamma = cv2.imread('images/hough/mamma.png')
meg = cv2.imread('images/hough/johannes.png')

to_size = (4000, 3000)#(edge.shape[0]*2,  edge.shape[1]*2)

mamma = cv2.resize(mamma, to_size)
pappa = cv2.resize(pappa, to_size)
meg = cv2.resize(meg, to_size)

mamma = np.flip(mamma, axis=0)
pappa = np.flip(pappa, axis=1)

image = np.zeros([3000, 4000, 3])
image[:, :, 0] = mamma[:, :, 0]
image[:, :, 1] = pappa[:, :, 0]
image[:, :, 2] = meg[:, :, 0]


plt.imshow(image/255)
plt.show()

imsave('images/to_johannes.png', image)