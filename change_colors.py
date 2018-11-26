import cv2
from matplotlib import pyplot as plt
import numpy as np


prefix = 'mamma_meg'
image = cv2.imread(f'images/output/{prefix}_color_hough_large.png')
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#img = (255-img)
img[img==0] = 255
print(np.min(img), np.mean(img), np.max(img))

plt.imshow(img/255)
plt.show()