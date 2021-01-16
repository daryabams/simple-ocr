from segmentation import binarize, line, word
import environment as env
import util.plot as plot
import numpy as np
import cv2 as cv
import os


class Image():
	def __init__(self, i_path):
		self.name = os.path.basename(i_path).split(".")[0]
		self.file_ext = ".png"

		self.img = cv.imread(i_path, cv.IMREAD_GRAYSCALE)
		self.binary = None

		cv.imwrite(self.file_path_out(), self.img)

	def file_path_out(self, ext=None, *args):
		path = os.path.join(env.OUT_PATH, self.name, *args)
		os.makedirs(path, exist_ok=True)

		ext = "" if ext is None else "_" + ext
		return os.path.join(path, self.name + ext + self.file_ext)

	def threshold(self, method):
		if method == "su":
			self.binary = binarize.su(self.img)
		elif method == "suplus":
			self.binary = binarize.su_plus(self.img)
		elif method == "sauvola":
			self.binary = binarize.sauvola(self.img, [127, 127], 127, 0.1)
		else:
			self.binary = binarize.otsu(self.img)

		cv.imwrite(self.file_path_out("1_binary"), self.binary)

	def segment(self):
		lines = self.segment_lines()
		words = self.segment_words()



	def segment_lines(self):
		l = line.LineSegmentation(self.binary)

	# find letters contours
		l.find_contours()
		plot.rects(self.file_path_out("2_contours"), self.binary, l.contours)

		# divide image into vertical chunks
		l.divide_chunks()
		plot.chunks(self.file_path_out("chunk#", "chunks"), l.chunks)
		plot.chunks_histogram(self.file_path_out("3_histogram"), l.chunks)

		# get initial lines
		l.get_initial_lines()
		plot.image_with_lines(self.file_path_out(
			"4_initial_lines"), self.binary, l.initial_lines)

		try:
			# get initial line regions
			l.generate_regions()

			# repair initial lines and generate the final line regions
			l.repair_lines()

			# generate the final line regions
			l.generate_regions()
		except:
			pass

		plot.image_with_lines(self.file_path_out(
			"5_final_lines"), self.binary, l.initial_lines)

		# get lines to segment
		img_lines = l.get_regions()
		plot.lines(self.file_path_out("line#", "lines"), img_lines)

		return img_lines

	def segment_words(self):
		_w = word.WordSegmentation(self.binary)
		_w.set_kernel(kernel_size=11, sigma=11, theta=7)
		
		# read input images from 'in' directory
		in_dir = os.path.join(env.OUT_PATH, self.name, "lines")
		imgFiles = os.listdir(in_dir)
		print(imgFiles)
		for (i,f) in enumerate(imgFiles):
			print('Segmenting words of sample %s'%f)
			
			# read image, prepare it by resizing it to fixed height and converting it to grayscale
			img = _w.prepare_img(cv.imread(os.path.join(in_dir, f)), 100)
			
			# execute segmentation with given parameters
			res = _w.generate_regions(img)
			
			# iterate over all segmented words
			print('Segmented into %d words'%len(res))
			img_words = []
			for (j, w) in enumerate(res):
				(wordBox, wordImg) = w
				(x, y, w, h) = wordBox
				img_words.append(wordImg)
				#cv.imwrite('../out/%s/%d.png'%(f, j), wordImg) # save word
				# cv.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
			
			plot.lines(self.file_path_out("word#", "words", f), img_words)
			# output summary image with bounding boxes around words
			# cv.imwrite('../out/%s/summary.png'%f, img)

		return img_words
	
	def segment_characters(self, word):
		res = []

		contours, _ = cv.findContours(word, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
		for contour in contours:
			epsilon = 0.03 * cv.arcLength(contour, True)
			approx = cv.approxPolyDP(contour, epsilon, True)
			curr_box = cv.boundingRect(approx)
			(x, y, w, h) = curr_box
			if cv.contourArea(contour) < 10:
				continue
			if (x == 0) and (y == 0):
				continue
				
			curr_img = word[y:y+h, x:x+w]
			res.append(curr_img)
		
		return res
