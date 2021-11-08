import numpy as np
import cv2
import csv
from sklearn import svm
import pickle

from src.create_feature import *
from src.calorie_calc import *

model_file = 'models/svm_data.dat'

def classify(img_path):
	#svm_model = cv2.ml.SVM_create()
	#svm_model.load('svm_data.dat')
	svm_model = pickle.load(open(model_file, 'rb'))
	feature_mat = []
	response = []

	try:
		fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(img_path)
		feature_mat.append(fea)
		response.append([float(0)])
	# contour error
	except IndexError:
		return False

	testData = np.float32(feature_mat).reshape(-1,94)
	responses = np.float32(response)
	result = svm_model.predict(testData)
	mask = result==responses

	#calculate calories
	volume = getVolume(result[0], farea, skinarea, pix_to_cm, fcont)
	mass, cal, cal_100 = getCalorie(result[0], volume)

	return result[0], cal
