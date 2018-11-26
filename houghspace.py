import numpy as np
from tqdm import tqdm
from typing import Union
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
from scipy.misc import imsave
from util import load, grayscale, edge_detection
from datetime import datetime

class HoughSpace:
	def __init__(self, data: Union[Path, str, np.ndarray], save=True):
		if type(data) == str or type(data) == Path:
			image = load(str(data))
		else:
			image = data
			self.image = image
			self.gray = grayscale(np.array(image))
			self.edges = edge_detection(np.array(self.gray))
			self.hough = self.transform()
			
			if type(data) == str or type(data) == Path:
				prefix = Path(data).stem
			else: 
				prefix = datetime.now().strftime("%Y-%m-%d-H%M")
			if save:
				self.save_images(str(prefix))
				self.show()

	def transform(self):
		img = self.edges

		height, width = img.shape


		# Rho and Theta ranges
		diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
		thetas = np.deg2rad(np.arange(0.0, 180.0, 180/height))
		#rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

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

		accumulator = accumulator/np.max(accumulator)
		return np.stack((np.rot90(accumulator),)*3, axis=-1)

	def quick_transform(self, out_shape=None):
		raise DeprecationWarning('Old implementation')
		img = self.edges

		height, width = img.shape
		#if out_shape:
		#    width, height = out_shape

		# Rho and Theta ranges
		diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
		thetas = np.deg2rad(np.arange(0.0, 180.0, 180/height))
		#rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

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
			rho = x*cos_t + y*sin_t
			rho_scaled = rho*width/diag_len

			theta = np.arange(len(cos_t))*height/num_thetas

			theta_coor = np.round(theta).astype(np.int8)
			rho_coor = np.round(rho_scaled).astype(np.int8)

			accumulator[rho_coor][theta_coor] = accumulator[rho_coor][theta_coor] + 1

			#for i, (cos, sin)  in enumerate(zip(cos_t, sin_t)):
			#    # Calculate rho. diag_len is added for a positive index
			#    rho = x * cos + y * sin
			#    rho_coor = int(round(rho*width/diag_len))
			#    theta_coor = int(round(i*height/num_thetas))
			#    try:
			#        accumulator[rho_coor][theta_coor] += 1
			#    except IndexError as e:
			#        print(e)

		print(accumulator.shape)
		print(np.max(accumulator))
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
		plt.imshow(self.hough)

		for i in range(1, 5):
			plt.subplot(220+i)
			plt.xticks([])
			plt.yticks([])

		plt.show()

	def save_images(self, prefix):
		print('saving to', prefix)
		imsave(f'images/gray/{prefix}.png', self.gray)
		imsave(f'images/edge/{prefix}.png', self.edges)
		imsave(f'images/hough/{prefix}.png', self.hough*255)

	@staticmethod
	def alternate_transform(image):
		raise DeprecationWarning('Old implementation')

		import numpy as np

		from skimage.transform import (hough_line, hough_line_peaks,
									   probabilistic_hough_line)
		from skimage.feature import canny
		from skimage import data

		import matplotlib.pyplot as plt
		from matplotlib import cm


		# Classic straight-line Hough transform
		h, theta, d = hough_line(image)

		# Generating figure 1
		fig, axes = plt.subplots(1, 3, figsize=(15, 6))
		ax = axes.ravel()

		ax[0].imshow(image, cmap=cm.gray)
		ax[0].set_title('Input image')
		ax[0].set_axis_off()

		ax[1].imshow(np.log(1 + h),
					 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
					 cmap=cm.gray, aspect=1 / 1.5)
		ax[1].set_title('Hough transform')
		ax[1].set_xlabel('Angles (degrees)')
		ax[1].set_ylabel('Distance (pixels)')
		ax[1].axis('image')

		ax[2].imshow(image, cmap=cm.gray)
		for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
			y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
			y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
			ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
		ax[2].set_xlim((0, image.shape[1]))
		ax[2].set_ylim((image.shape[0], 0))
		ax[2].set_axis_off()
		ax[2].set_title('Detected lines')

		plt.tight_layout()
		plt.show()

		# Line finding using the Probabilistic Hough Transform


		# Generating figure 2
		fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
		ax = axes.ravel()

		ax[0].imshow(image, cmap=cm.gray)
		ax[0].set_title('Input image')

		ax[1].imshow(edges, cmap=cm.gray)
		ax[1].set_title('Canny edges')

		ax[2].imshow(edges * 0)
		for line in lines:
			p0, p1 = line
			ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
		ax[2].set_xlim((0, image.shape[1]))
		ax[2].set_ylim((image.shape[0], 0))
		ax[2].set_title('Probabilistic Hough')

		for a in ax:
			a.set_axis_off()

		plt.tight_layout()
		plt.show()
