# Adapted from https://github.com/meghanamreddy/Calorie-estimation-from-food-images-OpenCV/blob/master/learn.py

import numpy as np
import cv2
from src.create_feature import *
from src.calorie_calc import *
import csv
from sklearn import svm
import pickle

# svm_params = dict(kernel_type = cv2.ml.SVM_LINEAR, svm_type = cv2.ml.SVM_C_SVC, C=2.67, gamma=5.383 )
filename = 'svm_data.dat'


def training():
	feature_mat = []
	response = []
	for j in range(1, 15):
		for i in range(1,21):
			print ("./Dataset/images/All_Images/"+str(j)+"_"+str(i)+".jpg")
			try:
				fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg("./Dataset/images/All_Images/"+str(j)+"_"+str(i)+".jpg")
				feature_mat.append(fea)
				response.append(float(j))
			# sometimes contours not found; need to figure out how to deal if happens w user image
			except IndexError:
				print("Ignoring file:")

	trainData = np.float32(feature_mat).reshape(-1,94)
	responses = np.float32(response)

	### OLD SVM CODE USING CV2 ###
	# svm = cv2.ml.SVM_create()
	# svm.train(trainData,responses, svm_params)
	# our_svm.save('svm_data.dat')

	train_svm = svm.SVC()
	train_svm.fit(trainData, responses)
	pickle.dump(train_svm, open(filename, 'wb'))


def testing():
	#svm_model = cv2.ml.SVM_create()
	#svm_model.load('svm_data.dat')
	svm_model = pickle.load(open(filename, 'rb'))
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
	for j in range(1, 15):
		for i in range(21,26):
			img_path = "./Dataset/images/Test_Images/"+str(j)+"_"+str(i)+".jpg"
			print (img_path)
			try:
				fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(img_path)
				pix_cm.append(pix_to_cm)
				fruit_contours.append(fcont)
				fruit_areas.append(farea)
				feature_mat.append(fea)
				skin_areas.append(skinarea)
				response.append([float(j)])
				image_names.append(img_path)
			except IndexError:
				print("Ignoring file:")

	testData = np.float32(feature_mat).reshape(-1,94)
	responses = np.float32(response)
	#result = svm_model.predict_all(testData)
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

	#write into csv file
	with open('output.csv', 'w') as outfile:
		writer = csv.writer(outfile)
		data = ["Image name", "Desired response", "Output label", "Volume (cm^3)", "Mass (grams)", "Calories for food item", "Calories per 100 grams"]
		writer.writerow(data)

		for i in range(0, len(result)):
			if (fruit_volumes[i] == None):
				data = [str(image_names[i]), str(responses[i][0]), str(result[i]), "--", "--", "--", str(fruit_calories_100grams[i])]
			else:
				data = [str(image_names[i]), str(responses[i][0]), str(result[i]), str(fruit_volumes[i]), str(fruit_mass[i]), str(fruit_calories[i]), str(fruit_calories_100grams[i])]
			writer.writerow(data)
		outfile.close()

	right = 0
	for i in range(0, len(mask)):
		if responses[i][0] == result[i]:
			right+=1
		# if mask[i][0] == False:
			#print ("(Actual Reponse)", responses[i][0], "(Output)", result[i], image_names[i])

	#correct = np.count_nonzero(mask)
	#print (correct*100.0/result.size)
	print (right/len(mask))



if __name__ == '__main__':
	training()
	# testing()
