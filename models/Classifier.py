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
        "watermelon",
        "cabbage",
        "eggplant",
        "pear",
        "zucchini"
    ]

    # model_file = 'models/svm_data.dat'
    # model_file = 'models/svm_data2.dat'
    # model_file = 'models/svm_data3.dat'
    # model_file = 'models/svm_data4.dat'
    # model_file = 'models/svm_data_final.dat'
    # with train_images
    model_file = 'models/svm_data_final2.dat'
    # with train_images_broken
    # model_file = 'models/svm_data_final2.dat'
    # with train_images_2
    # model_file = 'models/svm_data_final3.dat'

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
        "watermelon": 2500,
        "cabbage": 1500,
        "eggplant": 600,
        "pear": 300,
        "zucchini": 1000
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

        train_svm = svm.SVC(probability=True)
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
        # print(responses)

        #result = svm_model.predict_all(testData)
        final_result = []
        result = svm_model.predict_proba(testData)
        print(result)
        for probability_list in result:
            max_prob = max(probability_list)
            max_index = np.where(probability_list==max_prob)[0] + 1
            # print(max_index)
            final_result.append(max_index)

        print(final_result)

        # mask = final_result == responses

        # calculate calories
        # for i in range(0, len(result)):
        #     volume = self.processor.getVolume(result[i], fruit_areas[i],
        #                        skin_areas[i], pix_cm[i], fruit_contours[i])
        #     macros = self.processor.getMacros(result[i], volume)
        #     # if len(macros) == 3:
        #     mass, cal, cal_100 = macros
        #     # else:
        #     #     mass = macros[0]
        #     #     cal = macros[1]
        #     #     cal_100 = macros[5]
		#     # mass, cal, cal_100 = self.processor.getMacros(result[i], volume)[0:3]
        #     # mass, cal, protein, carb, fat, cal_100, protein_100, carb_100, fat_100 = self.processor.getMacros(result[i], volume)
        #     fruit_volumes.append(volume)
        #     fruit_calories.append(cal)
        #     fruit_calories_100grams.append(cal_100)
        #     fruit_mass.append(mass)

        # write into csv file
        # with open('output.csv', 'w') as outfile:
        #     writer = csv.writer(outfile)
        #     data = ["Image name", "Desired response", "Output label",
        #             "Volume (cm^3)", "Mass (grams)", "Calories for food item", "Calories per 100 grams"]
        #     writer.writerow(data)
        #
        #     for i in range(0, len(result)):
        #         if (fruit_volumes[i] == None):
        #             data = [str(image_names[i]), str(responses[i][0]), str(
        #                 result[i]), "--", "--", "--", str(fruit_calories_100grams[i])]
        #         else:
        #             data = [str(image_names[i]), str(responses[i][0]), str(result[i]), str(fruit_volumes[i]), str(
        #                 fruit_mass[i]), str(fruit_calories[i]), str(fruit_calories_100grams[i])]
        #         writer.writerow(data)
        #     outfile.close()

        right = 0
        for i in range(len(responses)):
            if response[i][0] ==  final_result[i]:
                right += 1
        # for i in range(0, len(mask)):
        #     if responses[i][0] == final_result[i]:
        #         right += 1
            # else:
            #     print("actual: " + (result[i][0]))
            #     print("output: " + result[i])
            # if mask[i][0] == False:
                #print ("(Actual Reponse)", responses[i][0], "(Output)", result[i], image_names[i])

        #correct = np.count_nonzero(mask)
        #print (correct*100.0/result.size)
        print("accuracy rate:")
        print(right/len(responses))

    def test2(self, folder_path):

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
        print(result)

        mask = result == responses

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
        # result = svm_model.predict(testData)
        probabilities = svm_model.predict_proba(testData)[0]

        # ind = np.argpartition(probabilities, -3)[-3:]
        indices = probabilities.argsort()[-3:][::-1]
        print(indices)
        dict = {}

        dict = {
            1: indices[0] + 1,
            2: indices[1] + 1,
            3: indices[2] + 1
        }

        print(dict)

        # dict[]
        #
        # food2probability = {
        #     self.index2classification[indices[0]+1]: probabilities[indices[0]],
        #     self.index2classification[indices[1]+1]: probabilities[indices[1]],
        #     self.index2classification[indices[2]+1]: probabilities[indices[2]],
        # # }
        # max_index = indices[0] + 1
        # # print()
        # max_prob = max(probabilities)
        # # print(indices)
        # classification = self.index2classification[max_index]
        # max_index = np.where(result==max_prob)
        #
        # probabilities[max_index] = 0
        # second_max_prob = max(probabilities)
        # second_index = np.where(result == second_max_prob)
        # #
        # probabilities[second_index] = 0
        # second_max_prob = max(probabilities)
        # second_index = np.where(result == second_max_prob)

        # print(result)
        # print(result)
        # print(probabilities)
        # print(svm_model.classes_)


        # print(max_prob)
        # print(result)
        # mask = result == responses

        # calculate calories
        # volume = self.processor.getVolume(
        #     result[0], farea, skinarea, pix_to_cm, fcont)
        volume = None
        for rank in dict:
            food_index = dict[rank]
            food = self.index2classification[food_index]
            confidence = probabilities[food_index - 1]

            print(confidence)
            if volume is None:
                volume = self.processor.getVolume(
                    food_index, farea, skinarea, pix_to_cm, fcont)

            volume = self.constraint_check(food, volume)
            mass, cal, protein, carb, fat = self.processor.getMacros(food_index, volume)

            dict_value = {
                "food": food,
                "mass": round(mass, 2),
                "calorie": round(cal, 2),
                "protein": round(protein, 2),
                "carb": round(carb, 2),
                "fat": round(fat, 2),
                "confidence": round(confidence, 3)
            }

            dict[rank] = dict_value
            print(dict)

        # classification = self.index2classification[int(result[0])]
        # classification = self.index2classification[max_index]

        # volume = 5000


        # print(max_prob)

        print(dict)

        return dict
        # return classification, cal, protein, carb, fat, max_prob, food2probability
