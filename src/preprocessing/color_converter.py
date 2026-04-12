import numpy as np
from skimage.color import rgb2lab, rgb2hsv, separate_stains, hdx_from_rgb


HE_MATRIX = np.array([
	[0.644211, 0.716556, 0.266844],
	[0.092789, 0.954111, 0.283111],
	[0.63718,  0.00000,  0.000103],
])


def convert_image(image_uint8):
	"""
	Convert RGB image to LAB, HSV, and HED color spaces.

	Parameters
	----------
	image_uint8 : np.ndarray, shape (256,256,3), dtype uint8

	Returns
	-------
	dict with keys: 'rgb', 'lab', 'hsv', 'hed'
	Each value is np.ndarray shape (256,256,3), float32
	"""
	img_float = image_uint8.astype(np.float32)
	img_01 = img_float / 255.0

	lab = rgb2lab(img_01).astype(np.float32)
	hsv = rgb2hsv(img_01).astype(np.float32)

	_ = hdx_from_rgb
	hed = separate_stains(img_01, HE_MATRIX).astype(np.float32)

	return {
		'rgb': img_float,
		'lab': lab,
		'hsv': hsv,
		'hed': hed,
	}
