import cv2 as cv
import numpy as np


class CharacterSegmentation():
	
	def __init__(self):
		pass

	def generate_regions(self, img):
		res = []

		contours, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
		for contour in contours:
			epsilon = 0.03 * cv.arcLength(contour, True)
			approx = cv.approxPolyDP(contour, epsilon, True)
			curr_box = cv.boundingRect(approx)
			(x, y, w, h) = curr_box
			if cv.contourArea(contour) < 10:
				continue
			if (x == 0) and (y == 0):
				continue
				
			curr_img = img[y:y+h, x:x+w]
			res.append(curr_img)
		
		return res