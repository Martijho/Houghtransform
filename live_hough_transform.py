from houghspace import hough_live
from util import load
import cv2


edge_image = cv2.resize(load('images/edge/segments/person.png'), None, fx=0.5, fy=0.5)
hough_live(edge_image[:, :, 0])