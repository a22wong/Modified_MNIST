import numpy as np
import math


def resize_images(data_set, resolution):
	# data_length is the number of images, image_size is the total number of pixels in one image (m = width * height)
	(data_length, image_size) = data_set.shape

	# n is the width and height of the squared images
	n = int(math.sqrt(image_size))

	factor = n / resolution
	rest = (n / factor) % resolution

	reshaped_data_set = np.empty((data_length, resolution*resolution))

	for index, image in enumerate(data_set):
		# make matrix not flat
		reshaped_matrix = image.reshape((n, n))

		# create a new matrix with new dimensions
		factored_matrix = np.empty((n / factor, n / factor))

		# store a factor of the resolution in the new matrix
		for row in xrange(0, n, factor):
			factored_matrix[row/factor] = reshaped_matrix[row][0::factor]

		# cutoff excess rows/cols
		for i in range(rest/2):
			factored_matrix = np.delete(factored_matrix, [0, factored_matrix[0].size - 1], 0)
			factored_matrix = np.delete(factored_matrix, [0, factored_matrix[0].size - 1], 1)

		reshaped_data_set[index] = factored_matrix.flatten()

	return reshaped_data_set

