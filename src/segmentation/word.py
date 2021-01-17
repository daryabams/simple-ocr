import cv2 as cv
import numpy as np


class WordSegmentation():
	
	def __init__(self, binary):
		self.kernel_size = 11
		self.sigma = 11
		self.theta = 7
		self.min_area = 100

	def set_kernel(self, kernel_size, sigma, theta):
		self.kernel_size = kernel_size
		self.sigma = sigma
		self.theta = theta


	def generate_regions(self, img):
		"""Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
		
		Args:
			img: grayscale uint8 image of the text-line to be segmented.
			kernel_size: size of filter kernel, must be an odd integer.
			sigma: standard deviation of Gaussian function used for filter kernel.
			theta: approximated width/height ratio of words, filter function is distorted by this factor.
			min_area: ignore word candidates smaller than specified area.
			
		Returns:
			List of tuples. Each tuple contains the bounding box and the image of the segmented word.
		"""
		# apply filter kernel
		kernel = self.__create_kernel(self.kernel_size, self.sigma, self.theta)
		imgFiltered = cv.filter2D(img, -1, kernel, borderType=cv.BORDER_REPLICATE).astype(np.uint8)
		(_, imgThres) = cv.threshold(imgFiltered, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
		imgThres = 255 - imgThres

		# find connected components. OpenCV: return type differs between Opencv and 3
		if cv.__version__.startswith('3.'):
			(_, components, _) = cv.findContours(imgThres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		else:
			(components, _) = cv.findContours(imgThres, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

		# append components to result
		res = []
		for c in components:
			# skip small word candidates
			if cv.contourArea(c) < self.min_area:
				continue
			# append bounding box and image of word to result list
			currBox = cv.boundingRect(c) # returns (x, y, w, h)
			(x, y, w, h) = currBox
			currImg = img[y:y+h, x:x+w]
			res.append((currBox, currImg))

		# return list of words, sorted by x-coordinate
		return sorted(res, key=lambda entry:entry[0][0])


	def prepare_img(self, img, height):
		"""convert given image to grayscale image (if needed) and resize to desired height"""
		assert img.ndim in (2, 3)
		if img.ndim == 3:
			img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		h = img.shape[0]
		factor = height / h
		return cv.resize(img, dsize=None, fx=factor, fy=factor)


	def __create_kernel(self, kernel_size, sigma, theta):
		"""create anisotropic filter kernel according to given parameters"""
		assert kernel_size % 2 # must be odd size
		halfSize = kernel_size // 2
		
		kernel = np.zeros([kernel_size, kernel_size])
		sigmaX = sigma
		sigmaY = sigma * theta
		
		for i in range(kernel_size):
			for j in range(kernel_size):
				x = i - halfSize
				y = j - halfSize
				
				expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
				xTerm = (x**2 - sigmaX**2) / (2 * np.pi * sigmaX**5 * sigmaY)
				yTerm = (y**2 - sigmaY**2) / (2 * np.pi * sigmaY**5 * sigmaX)
				
				kernel[i, j] = (xTerm + yTerm) * expTerm

		kernel = kernel / np.sum(kernel)
		return kernel
