
import numpy as np
import cv2
import csv
from sklearn import svm
import pickle

from src.create_feature import *
from src.calorie_calc import *

model_file = './src/svm_data.dat'

def classify(img_path):
	#svm_model = cv2.ml.SVM_create()
	#svm_model.load('svm_data.dat')
	svm_model = pickle.load(open(model_file, 'rb'))
	feature_mat = []
	response = []
	image_names = []
	pix_cm = []
	fruit_contours = []
	fruit_areas = []
	fruit_volumes = []
	fruit_mass = []
	fruit_calories = []
	skin_areas = []
	fruit_calories_100grams = []

	try:
		fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(img_path)
		pix_cm.append(pix_to_cm)
		fruit_contours.append(fcont)
		fruit_areas.append(farea)
		feature_mat.append(fea)
		skin_areas.append(skinarea)
		response.append([float(0)])
		image_names.append(img_path)
	# contour error
	except IndexError:
		return False

	testData = np.float32(feature_mat).reshape(-1,94)
	responses = np.float32(response)
	result = svm_model.predict(testData)
	mask = result==responses

	#calculate calories
	for i in range(0, len(result)):
		volume = getVolume(result[i], fruit_areas[i], skin_areas[i], pix_cm[i], fruit_contours[i])
		mass, cal, cal_100 = getCalorie(result[i], volume)
		fruit_volumes.append(volume)
		fruit_calories.append(cal)
		fruit_calories_100grams.append(cal_100)
		fruit_mass.append(mass)

	return result
