from houghspace import HoughSpace
from util import *
from time import time
import cv2
from scipy.misc import imsave


whole = cv2.imread('images/edge/edge2.png')
pappa = cv2.imread('images/edge/mamma.png')
johannes = cv2.imread('images/edge/meg.png')
whole = whole - pappa - johannes

edge = np.array(whole)
edge[:, :, 1] = pappa[:, :, 0]
edge[:, :, 2] = johannes[:, :, 0]

#cv2.imshow('', edge)
#cv2.waitKey()
cv2.imwrite('test.jpg', edge)

to_size = (3000, 4000)#(edge.shape[0]*2,  edge.shape[1]*2)

edge = cv2.resize(edge, to_size)
pappa = cv2.resize(pappa, to_size)
johannes = cv2.resize(johannes, to_size)

hough_background = HoughSpace(edge).hough#, out_shape=(200, 200))
hough_pappa = HoughSpace(pappa).hough#, out_shape=(200, 200))
hough_johannes = HoughSpace(johannes).hough#, out_shape=(200, 200))

hough_background[:, :, 1] = hough_pappa[:, :, 0]
hough_background[:, :, 2] = hough_johannes[:, :, 0]

imsave('images/output/pappa_johannes_color_hough_large.png', hough_background)
#hs.transform()
#hs.show()
#hs.alternate_transform(hs.gray)
