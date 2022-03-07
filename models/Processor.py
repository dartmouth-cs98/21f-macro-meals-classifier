import cv2
import numpy as np
import sys
import math
from multiprocessing.pool import ThreadPool


class Processor:

	#density - gram / cm^3
	density_dict = {1:0.96, 2:1.09, 3:1.04, 4:0.98, 5:0.97, 6:0.814, 7:0.45, 8:0.57, 9:0.47, 10:0.96, 11:0.59}
	#kcal
	calorie_dict = { 1:52, 2:89, 3:41, 4:15, 5:40, 6:47, 7:40, 8:61, 9:18, 10:30, 11:57}
	protein_dict = { 1:0.3, 2:1.1, 3:0.9, 4:0.66, 5:1.1, 6:0.9, 7:1.9, 8:1.1, 9:0.9, 10:0.6, 11:0.38}
	carb_dict = { 1:14, 2:23, 3:10, 4:3.6, 5:9, 6:12, 7:9, 8:15, 9:3.9, 10:8, 11:15}
	fat_dict = { 1:0.2, 2:0.3, 3:0.2, 4:0.1, 5:0.1, 6:0.1, 7:0.4, 8:0.5, 9:0.2, 10:0.2, 11:0.1}
	#skin of photo to real multiplier
	skin_multiplier = 5*2.3

	def __init__(self):
		pass

	def getShapeFeatures(self, img):
		'''
		The shape features of an image are calculated
		based on the contour of the food item using Hu moments.
		'''
		contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		moments = cv2.moments(contours[0])
		hu = cv2.HuMoments(moments)
		feature = []
		for i in hu:
			feature.append(i[0])
		M = max(feature)
		m = min(feature)

		feature = list(map(lambda x: x * 2, feature))
		feature = (feature - M - m)/(M - m)
		mean=np.mean(feature)
		dev=np.std(feature)
		feature = (feature - mean)/dev
		return feature

	def getAreaOfFood(self, img1):
		img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		img_filt = cv2.medianBlur( img, 5)
		img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		# find contours. sort. and find the biggest contour. the biggest contour corresponds to the plate and fruit.
		mask = np.zeros(img.shape, np.uint8)
		largest_areas = sorted(contours, key=cv2.contourArea)
		cv2.drawContours(mask, [largest_areas[-1]], 0, (255,255,255,255), -1)
		img_bigcontour = cv2.bitwise_and(img1,img1,mask = mask)

		# convert to hsv. otsu threshold in s to remove plate
		hsv_img = cv2.cvtColor(img_bigcontour, cv2.COLOR_BGR2HSV)
		h,s,v = cv2.split(hsv_img)
		mask_plate = cv2.inRange(hsv_img, np.array([0,0,100]), np.array([255,90,255]))
		mask_not_plate = cv2.bitwise_not(mask_plate)
		fruit_skin = cv2.bitwise_and(img_bigcontour,img_bigcontour,mask = mask_not_plate)

		#convert to hsv to detect and remove skin pixels
		hsv_img = cv2.cvtColor(fruit_skin, cv2.COLOR_BGR2HSV)
		skin = cv2.inRange(hsv_img, np.array([0,10,60]), np.array([10,160,255])) #Scalar(0, 10, 60), Scalar(20, 150, 255)
		not_skin = cv2.bitwise_not(skin); #invert skin and black
		fruit = cv2.bitwise_and(fruit_skin,fruit_skin,mask = not_skin) #get only fruit pixels

		fruit_bw = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
		fruit_bin = cv2.inRange(fruit_bw, 10, 255) #binary of fruit

		#erode before finding contours
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		erode_fruit = cv2.erode(fruit_bin,kernel,iterations = 1)

		#find largest contour since that will be the fruit
		img_th = cv2.adaptiveThreshold(erode_fruit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
		largest_areas = sorted(contours, key=cv2.contourArea)
		cv2.drawContours(mask_fruit, [largest_areas[-2]], 0, (255,255,255), -1)
		#dilate now
		kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
		mask_fruit2 = cv2.dilate(mask_fruit,kernel2,iterations = 1)
		res = cv2.bitwise_and(fruit_bin,fruit_bin,mask = mask_fruit2)
		fruit_final = cv2.bitwise_and(img1,img1,mask = mask_fruit2)
		#find area of fruit
		img_th = cv2.adaptiveThreshold(mask_fruit2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		largest_areas = sorted(contours, key=cv2.contourArea)
		fruit_contour = largest_areas[-2]
		fruit_area = cv2.contourArea(fruit_contour)


		#finding the area of skin. find area of biggest contour
		skin2 = skin - mask_fruit2
		#erode before finding contours
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		skin_e = cv2.erode(skin2,kernel,iterations = 1)
		img_th = cv2.adaptiveThreshold(skin_e,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		mask_skin = np.zeros(skin.shape, np.uint8)
		largest_areas = sorted(contours, key=cv2.contourArea)

		# if can't find finger
		try:
			# print(len(largest_areas))
			cv2.drawContours(mask_skin, [largest_areas[-2]], 0, (255,255,255), -1)
			skin_rect = cv2.minAreaRect(largest_areas[-2])
			box = cv2.boxPoints(skin_rect)
			box = np.int0(box)
			mask_skin2 = np.zeros(skin.shape, np.uint8)
			cv2.drawContours(mask_skin2,[box],0,(255,255,255), -1)

			pix_height = max(skin_rect[1])
			if pix_height == 0:
				pix_to_cm_multiplier = 0
			else:
				pix_to_cm_multiplier = 5.0/pix_height
			skin_area = cv2.contourArea(box)
		except IndexError:
			# exit()
			# print("debug: no contours found")
			# return False
			skin_area = None
			pix_to_cm_multiplier = None

		return fruit_area, mask_fruit2, fruit_final, skin_area, fruit_contour, pix_to_cm_multiplier

	''' calorie calculation methods '''

	def getMacros(self, label, volume): #volume in cm^3
		'''
		Inputs are the volume of the foot item and the label of the food item
		so that the food item can be identified uniquely.
		The calorie content in the given volume of the food item is calculated.
		'''
		protein = self.protein_dict[int(label)]
		carb = self.carb_dict[int(label)]
		fat = self.fat_dict[int(label)]
		calorie = self.calorie_dict[int(label)]

		if (volume == None):
			return None, None, calorie
			# return None, None, None, None, None, calorie, None, None, None
		density = self.density_dict[int(label)]
		mass = volume*density*1.0
		calorie_tot = (calorie/100.0)*mass
		protein_tot = (protein/100.0)*mass
		carb_tot = (carb/100.0)*mass
		fat_tot = (fat/100.0)*mass
		# fat_tot = fat

		return mass, calorie_tot, protein_tot, carb_tot, fat_tot #calorie per 100 grams

	def getVolume(self, label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
		'''
		Using callibration techniques, the volume of the food item is calculated using the
		area and contour of the foot item by comparing the foot item to standard geometric shapes
		'''
		# print("skin area")
		# print(skin_area)
		if skin_area is not None:
			area_fruit = (area/skin_area)*self.skin_multiplier #area in cm^2
		else:
			area_fruit = area
			pix_to_cm_multiplier = 1
		label = int(label)
		volume = 100
		if label == 1 or label == 9 or label == 7 or label == 6 or label==12 or label==13: #sphere-apple,tomato,orange,kiwi,onion
			radius = np.sqrt(area_fruit/np.pi)
			volume = (4/3)*np.pi*radius*radius*radius
			#print area_fruit, radius, volume, skin_area

		if label == 2 or label == 10 or (label == 4 and area_fruit > 30): #cylinder like banana, cucumber, carrot
			fruit_rect = cv2.minAreaRect(fruit_contour)
			height = max(fruit_rect[1])*pix_to_cm_multiplier
			radius = area_fruit/(2.0*height)
			volume = np.pi*radius*radius*height

		if (label==4 and area_fruit < 30) or (label==5) or (label==11) or (label==3) or (label==14) or (label==8): #cheese, carrot, sauce
			volume = area_fruit*0.5 #assuming width = 0.5 cm

		return volume

	def build_filters(self):
		'''
		The Gabor kernel is calculated, which is later used to calculate the gabor features of an image
		'''
		filters = []
		ksize = 31
		for theta in np.arange(0, np.pi, np.pi / 8):
			for wav in [ 8.0, 13.0]:
				for ar in [0.8, 2.0]:
					kern = cv2.getGaborKernel((ksize, ksize), 5.0, theta, wav, ar, 0, ktype=cv2.CV_32F)
					filters.append(kern)
		#cv2.imshow('filt', filters[9])
		return filters

	def process_threaded(self, img, filters, threadn = 8):
		accum = np.zeros_like(img)
		def f(kern):
			return cv2.filter2D(img, cv2.CV_8UC3, kern)
		pool = ThreadPool(processes=threadn)
		for fimg in pool.imap_unordered(f, filters):
			np.maximum(accum, fimg, accum)
		return accum

	def EnergySum(self, img):
		mean, dev = cv2.meanStdDev(img)
		return mean[0][0], dev[0][0]

	def process(self, img, filters):
		'''
		Given an image and gabor filters,
		the gabor features of the image are calculated.
		'''
		feature = []
		accum = np.zeros_like(img)
		for kern in filters:
			fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
			a, b = self.EnergySum(fimg)
			feature.append(a)
			feature.append(b)
			np.maximum(accum, fimg, accum)

		M = max(feature)
		m = min(feature)
		feature = list(map(lambda x: x * 2, feature))
		feature = (feature - M - m)/(M - m)
		mean=np.mean(feature)
		dev=np.std(feature)
		feature = (feature - mean)/dev
		return feature

	def getTextureFeature(self, img):
		'''
		Given an image, the gabor filters are calculated and
		then the texture features of the image are calculated
		'''
		filters = self.build_filters()
		gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		res1 = self.process(gray_image, filters)
		return res1

	def getColorFeature(self, img):
		'''
		Computes the color feature vector of the image
		based on HSV histogram
		'''
		img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h,s,v = cv2.split(img_hsv)

		hsvHist = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(6)]

		featurevec = []
		hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [6,2,2], [0, 180, 0, 256, 0, 256])
		for i in range(6):
			for j in range(2):
				for k in range(2):
					featurevec.append(hist[i][j][k])
		feature = featurevec[1:]
		M = max(feature)
		m = min(feature)

		feature = list(map(lambda x: x * 2, feature))
		feature = (feature - M - m)/(M - m)
		mean=np.mean(feature)
		dev=np.std(feature)
		feature = (feature - mean)/dev

		return feature

	def createFeature(self, img):
		'''
		Creates the feature vector of the image using the three features -
		color, texture, and shape features
		'''
		feature = []
		areaFruit, binaryImg, colourImg, areaSkin, fruitContour, pix_to_cm_multiplier = self.getAreaOfFood(img)
		color = self.getColorFeature(colourImg)
		texture = self.getTextureFeature(colourImg)
		shape = self.getShapeFeatures(binaryImg)
		for i in color:
			feature.append(i)
		for i in texture:
			feature.append(i)
		for i in shape:
			feature.append(i)

		M = max(feature)
		m = min(feature)

		feature = list(map(lambda x: x * 2, feature))
		feature = (feature - M - m)/(M - m)
		mean=np.mean(feature)
		dev=np.std(feature)
		feature = (feature - mean)/dev

		return feature, areaFruit, areaSkin, fruitContour, pix_to_cm_multiplier

	def readFeatureImg(self, filename):
		'''
		Reads an input image when the filename is given,
		and creates the feature vector of the image.
		'''
		img = cv2.imread(filename)
		f, farea, skinarea, fcont, pix_to_cm = self.createFeature(img)
		return f, farea, skinarea, fcont, pix_to_cm
