import numpy as np


def resize_image(image, resolution):
	(n, m) = image.shape
	factor = n / resolution
	rest = n % resolution

	resized = np.empty((resolution, resolution))

	for i in range(resolution):
		for j in range(resolution):
			resized[i][j] = image[rest + i*factor][rest + j*factor]

	return resized

