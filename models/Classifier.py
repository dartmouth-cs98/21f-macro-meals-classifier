import cv2
import numpy as np
import sys
import pickle
from models.Processor import Processor
from sklearn import svm
import os
import csv

class Classifier:

    index2classification = [
        "none",
        "apple",
        "banana",
        "beans",
        "carrot",
        "cheese",
        "cucumber",
        "onion",
        "orange",
        "pasta",
        "pepper",
        "qiwi",
        "sauce",
        "tomato",
        "watermelon"
    ]

    # model_file = 'models/svm_data.dat'
    # model_file = 'models/svm_data2.dat'
    model_file = 'models/svm_data3.dat'

    # classification2floor_vol = {
    #     "apple": 50,
    #     "banana": 50,
    #     "beans":
    # }

    # in cm^3
    classification2ceiling_vol = {
        "apple": 250,
        "banana": 200,
        "beans": 2000,
        "carrot": 200,
        "cheese": 2000,
        "cucumber": 400,
        "onion": 800,
        "orange": 300,
        "pasta": 2000,
        "pepper": 2000,
        "qiwi": 250,
        "sauce": 2000,
        "tomato": 350,
        "watermelon": 2500
    }

    def __init__(self):
        self.processor = Processor()

    def train(self, folder_path):
        feature_mat = []
        response = []
        for food_type in os.listdir(folder_path):
            try:
                j = self.index2classification.index(food_type.lower())
                print("j: " + str(j))
            except ValueError:
                continue
            for file_name in os.listdir(folder_path + "/" + food_type):
                if file_name == "script.sh" or file_name == "script.sh~":
                    continue
                try:
                    img_path = folder_path+"/"+food_type+"/"+file_name
                    print(img_path)
                    fea, farea, skinarea, fcont, pix_to_cm = self.processor.readFeatureImg(img_path)
                    feature_mat.append(fea)
                    response.append(float(j))
                # sometimes contours not found; need to figure out how to deal if happens w user image
                except IndexError:
                    print("Ignoring file^")

        trainData = np.float32(feature_mat).reshape(-1, 94)
        responses = np.float32(response)

        train_svm = svm.SVC()
        train_svm.fit(trainData, responses)
        with open(self.model_file, "wb") as f:
            pickle.dump(train_svm, f)

    def train2(self):
        feature_mat = []
        response = []
        for j in range(460):
            try:
                fea, farea, skinarea, fcont, pix_to_cm = self.processor.readFeatureImg("train_images/Apple/"+"1_"+str(j+1)+".jpg")
                print("train_images/Apple/"+"1_"+str(j+1)+".jpg")
                feature_mat.append(fea)
                response.append(1)
                # sometimes contours not found; need to figure out how to deal if happens w user image
            except IndexError:
                print("Ignoring file^")
        for j in range(144):
            try:
                fea, farea, skinarea, fcont, pix_to_cm = self.processor.readFeatureImg("train_images/Banana/"+"2_"+str(j+1)+".jpg")
                print("train_images/Banana/"+"2_"+str(j+1)+".jpg")
                feature_mat.append(fea)
                response.append(2)
                # sometimes contours not found; need to figure out how to deal if happens w user image
            except IndexError:
                print("Ignoring file^")

        trainData = np.float32(feature_mat).reshape(-1, 94)
        responses = np.float32(response)

        train_svm = svm.SVC()
        train_svm.fit(trainData, responses)
        with open(self.model_file, "wb") as f:
            pickle.dump(train_svm, f)

    def test(self, folder_path):
        #svm_model = cv2.ml.SVM_create()
        # svm_model.load('svm_data.dat')
        svm_model = pickle.load(open(self.model_file, 'rb'))
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
            for i in range(21, 26):
                img_path = folder_path+str(j)+"_"+str(i)+".jpg"
                print(img_path)
                try:
                    fea, farea, skinarea, fcont, pix_to_cm = self.processor.readFeatureImg(
                        img_path)
                    pix_cm.append(pix_to_cm)
                    fruit_contours.append(fcont)
                    fruit_areas.append(farea)
                    feature_mat.append(fea)
                    skin_areas.append(skinarea)
                    response.append([float(j)])
                    image_names.append(img_path)
                except IndexError:
                    print("Ignoring file:")

        testData = np.float32(feature_mat).reshape(-1, 94)
        responses = np.float32(response)
        #result = svm_model.predict_all(testData)
        result = svm_model.predict(testData)
        mask = result == responses

        # calculate calories
        for i in range(0, len(result)):
            volume = self.processor.getVolume(result[i], fruit_areas[i],
                               skin_areas[i], pix_cm[i], fruit_contours[i])
            macros = self.processor.getMacros(result[i], volume)
            if len(macros) == 3:
                mass, cal, cal_100 = macros
            else:
                mass = macros[0]
                cal = macros[1]
                cal_100 = macros[5]
		    # mass, cal, cal_100 = self.processor.getMacros(result[i], volume)[0:3]
            # mass, cal, protein, carb, fat, cal_100, protein_100, carb_100, fat_100 = self.processor.getMacros(result[i], volume)
            fruit_volumes.append(volume)
            fruit_calories.append(cal)
            fruit_calories_100grams.append(cal_100)
            fruit_mass.append(mass)

        # write into csv file
        with open('output.csv', 'w') as outfile:
            writer = csv.writer(outfile)
            data = ["Image name", "Desired response", "Output label",
                    "Volume (cm^3)", "Mass (grams)", "Calories for food item", "Calories per 100 grams"]
            writer.writerow(data)

            for i in range(0, len(result)):
                if (fruit_volumes[i] == None):
                    data = [str(image_names[i]), str(responses[i][0]), str(
                        result[i]), "--", "--", "--", str(fruit_calories_100grams[i])]
                else:
                    data = [str(image_names[i]), str(responses[i][0]), str(result[i]), str(fruit_volumes[i]), str(
                        fruit_mass[i]), str(fruit_calories[i]), str(fruit_calories_100grams[i])]
                writer.writerow(data)
            outfile.close()

        right = 0
        for i in range(0, len(mask)):
            if responses[i][0] == result[i]:
                right += 1
            # else:
            #     print("actual: " + (result[i][0]))
            #     print("output: " + result[i])
            # if mask[i][0] == False:
                #print ("(Actual Reponse)", responses[i][0], "(Output)", result[i], image_names[i])

        #correct = np.count_nonzero(mask)
        #print (correct*100.0/result.size)
        print("accuracy rate:")
        print(right/len(mask))

    # for sanity check
    def constraint_check(self, classification, vol):

        if classification != "none":
            vol_ceiling = self.classification2ceiling_vol[classification]
            if vol > vol_ceiling:
                return vol_ceiling

            vol_floor = 100  # modify later
            if vol < vol_floor:
                return vol_floor

        return vol


    def classify(self, img_path):

        #svm_model = cv2.ml.SVM_create()
        # svm_model.load('svm_data.dat')
        with open(self.model_file, "rb") as f:
            svm_model = pickle.load(f)

        feature_mat = []
        response = []

        try:
            fea, farea, skinarea, fcont, pix_to_cm = self.processor.readFeatureImg(
                img_path)
            feature_mat.append(fea)
            response.append([float(0)])
        # contour error
        except IndexError:
            return False

        testData = np.float32(feature_mat).reshape(-1, 94)
        responses = np.float32(response)
        result = svm_model.predict(testData)
        mask = result == responses

        # calculate calories
        volume = self.processor.getVolume(
            result[0], farea, skinarea, pix_to_cm, fcont)
        # volume = 5000

        classification = self.index2classification[int(result[0])]
        # volume = 5000
        volume = self.constraint_check(classification, volume)
        mass, cal, protein, carb, fat, cal_100, protein_100, carb_100, fat_100 = self.processor.getMacros(result[0], volume)

        return classification, cal, protein, carb, fat
