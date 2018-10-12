from houghspace import HoughSpace
from util import *

hs = HoughSpace('C:/Users/Martin/Desktop/Houghtransform/images/pappa.jpg', out_shape=(50, 50))
#hs = HoughSpace('C:/Users/Martin/Desktop/dot.png')

hs.show()
#hs.show()
'''
img = cv2.resize(img, (w//scale, h//scale))
cv2.imwrite('gray.jpg', img)
cv2.imshow('grayscale', cv2.resize(np.array(img), (800, 500)))
cv2.waitKey()
cv2.destroyAllWindows()

img = img/255
print(180/w)
accumulator, thetas, rhos = hough_line(img, 180/w)

#accumulator = cv2.resize(accumulator, (w, h))
cv2.imwrite('houghlines.jpg', accumulator*255)
cv2.imshow('grayscale', accumulator)
cv2.waitKey()
cv2.destroyAllWindows()
'''
