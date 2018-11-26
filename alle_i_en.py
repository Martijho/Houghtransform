from houghspace import HoughSpace
from util import *
from time import time
import cv2
from scipy.misc import imsave

prefix = 'alle'
back1 = cv2.imread('images/edge/edge2.png')
mamma = cv2.imread('images/edge/mamma.png')
meg = cv2.imread('images/edge/meg.png')
#background1 = np.array(back1)
#background1[mamma] = 0
#background1[meg] = 0

back2 = cv2.imread('images/edge/edge1.png')
pappa = cv2.imread('images/edge/pappa.png')
johannes = cv2.imread('images/edge/johannes.png')
#background2 = np.array(back2)
#background2[pappa] = 0
#background2[johannes] = 0


to_size = (4000, 3000)#(edge.shape[0]*2,  edge.shape[1]*2)

mamma = cv2.resize(mamma, to_size)
pappa = cv2.resize(pappa, to_size)
johannes = cv2.resize(johannes, to_size)
meg = cv2.resize(meg, to_size)

hough_mamma = HoughSpace(mamma).hough#, out_shape=(200, 200))
hough_pappa = HoughSpace(pappa).hough#, out_shape=(200, 200))
hough_johannes = HoughSpace(johannes).hough#, out_shape=(200, 200))
hough_meg = HoughSpace(meg).hough

#hough_background[:, :, 1] = hough_chan1[:, :, 0]
#hough_background[:, :, 2] = hough_chan2[:, :, 0]

for p, mat in zip(['mamma', 'pappa', 'johannes', 'meg'], [hough_mamma, hough_pappa, hough_johannes, hough_meg]):
    imsave(f'images/hough/{p}.png', mat)
#hs.transform()
#hs.show()
#hs.alternate_transform(hs.gray)
