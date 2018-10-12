import numpy as np
from tqdm import tqdm
from typing import Union
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

from util import load, grayscale, edge_detection

class HoughSpace:
    def __init__(self, data: Union[Path, str, np.ndarray], out_shape=None):
        if type(data) == str or type(data) == Path:
            image = load(str(data))
        else:
            image = data

        self.image = image
        self.gray = grayscale(np.array(image))
        self.edges = edge_detection(np.array(self.gray))
        self.hough = self.transform(out_shape=out_shape)

        if type(data) == str or type(data) == Path:
            prefix = Path('images/output') / Path(data).stem
            self.save_images(str(prefix))

    def transform(self, out_shape=None):
        img = self.edges

        height, width = img.shape
        #if out_shape:
        #    width, height = out_shape

        # Rho and Theta ranges
        diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
        thetas = np.deg2rad(np.arange(0.0, 180.0, 180/height))
        rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

        # Cache some resuable values
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        num_thetas = len(thetas)

        # Hough accumulator array of theta vs rho
        #accumulator = np.zeros((diag_len, diag_len))#len(thetas)))
        accumulator = np.zeros((width, height))
        #accumulator = np.zeros((width, height))#, dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

        # Vote in the hough accumulator
        for x, y in tqdm(zip(x_idxs, y_idxs), total=len(x_idxs)):
            for i, (cos, sin)  in enumerate(zip(cos_t, sin_t)):
                # Calculate rho. diag_len is added for a positive index
                rho = x * cos + y * sin
                rho_coor = int(round(rho*width/diag_len))
                theta_coor = int(round(i*height/num_thetas))
                try:
                    accumulator[rho_coor][theta_coor] += 1
                except IndexError as e:
                    print(e)
        accumulator = accumulator/np.max(accumulator.astype(np.float))
        return np.rot90(accumulator)

    def show(self):
        plt.figure()

        plt.subplot(221)
        plt.title('Original')
        plt.imshow(self.image)

        plt.subplot(222)
        plt.title('Grayscale')
        plt.imshow(self.gray, cmap='gray')

        plt.subplot(223)
        plt.title('Edges')
        plt.imshow(self.edges, cmap='gray')

        plt.subplot(224)
        plt.title('Hough transform')
        plt.imshow(self.hough, cmap='gray')

        for i in range(1, 5):
            plt.subplot(220+i)
            plt.xticks([])
            plt.yticks([])

        plt.show()

    def save_images(self, prefix):
        cv2.imwrite(prefix +'_gray.jpg', self.gray)
        cv2.imwrite(prefix +'_edges.jpg', self.edges)
        cv2.imwrite(prefix +'_hough.jpg', self.hough)
