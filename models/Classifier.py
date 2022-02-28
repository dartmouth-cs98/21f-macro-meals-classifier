import cv2
import numpy as np
import sys
import pickle
from models.Processor import Processor
from sklearn import svm
import os
import csv

class Classifier:

    # index2classification = [
    #     "none",
    #     "apple",
    #     "banana",
    #     "beans",
    #     "carrot",
    #     "cheese",
    #     "cucumber",
    #     "onion",
    #     "orange",
    #     "pasta",
    #     "pepper",
    #     "qiwi",
    #     "sauce",
    #     "tomato",
    #     "watermelon"
    # ]

    # for train_images_unit
    index2classification = [
        "none",
        "apple",
        "banana",
        "carrot",
        "cheese",
        "cucumber",
        "onion",
        "orange",
        "pepper",
        "qiwi",
        "tomato",
        "watermelon"
    ]

    # with train_images
    # model_file = 'models/train_images_data.dat'
    # model_file = 'models/train_images_unit_data.dat'
    # with train_images_broken
    # model_file = 'models/train_images_broken_data.dat'
    model_file = 'models/train_images_unit_cleaned_data.dat'


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
                    print(j)
                # sometimes contours not found; need to figure out how to deal if happens w user image
                except IndexError:
                    print("Ignoring file^")

        trainData = np.float32(feature_mat).reshape(-1, 94)
        responses = np.float32(response)

        train_svm = svm.SVC(probability=True)
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
        responses_including_ignore = 0
        for file_name in os.listdir(folder_path):
            if file_name == "script.sh" or file_name == "script.sh~":
                continue
            responses_including_ignore += 1
            img_path = folder_path + file_name
            j = int(file_name.split("_")[0])
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
        for probability_list in result:
            max_prob = max(probability_list)
            max_index = np.where(probability_list==max_prob)[0] + 1
            # print(max_index)
            final_result.append(max_index)

        right = 0
        for i in range(len(responses)):
            if response[i][0] ==  final_result[i]:
                right += 1
            else:
                print(self.index2classification[int(final_result[i])])

        print("accuracy rate:")
        print(right/len(responses))

        print("accuracy rate including error catches: ")
        print(right/responses_including_ignore)

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
        probabilities = svm_model.predict_proba(testData)[0]

        # get indices of foods w/ top 3 probabilities
        top_3_indices = probabilities.argsort()[-3:][::-1]
        rank2foodindex = {}

        # top 3 foods
        rank2foodindex = {
            "one": top_3_indices[0] + 1,
            "two": top_3_indices[1] + 1,
            "three": top_3_indices[2] + 1
        }

        # volume = None
        for rank in rank2foodindex:
            food_index = rank2foodindex[rank]
            food = self.index2classification[food_index]
            confidence = probabilities[food_index - 1]

            # if volume is None:
            volume = self.processor.getVolume(
                food_index, farea, skinarea, pix_to_cm, fcont)

            volume = self.constraint_check(food, volume)
            mass, cal, protein, carb, fat = self.processor.getMacros(food_index, volume)

            foodindex2info = {
                "food": food,
                "mass": round(mass, 2),
                "calorie": round(cal, 2),
                "protein": round(protein, 2),
                "carb": round(carb, 2),
                "fat": round(fat, 2),
                "confidence": round(confidence, 3)
            }

            rank2foodindex[rank] = foodindex2info


        return rank2foodindex
        # return classification, cal, protein, carb, fat, max_prob, food2probability
