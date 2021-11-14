import cv2
import numpy as np
import sys
import pickle
from models.Processor import Processor
from sklearn import svm

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

    model_file = 'models/svm_data.dat'

    def __init__(self):
        self.processor = Processor()

    def train(self, folder_path):
        feature_mat = []
        response = []
        for j in range(1, 15):
            for i in range(1, 21):
                try:
                    img_path = folder_path+str(j)+"_"+str(i)+".jpg"
                    print(img_path)
                    fea, farea, skinarea, fcont, pix_to_cm = self.processor.readFeatureImg(img_path)
                    feature_mat.append(fea)
                    response.append(float(j))
                # sometimes contours not found; need to figure out how to deal if happens w user image
                except IndexError:
                    print("Ignoring file:")

        trainData = np.float32(feature_mat).reshape(-1, 94)
        responses = np.float32(response)

        train_svm = svm.SVC()
        train_svm.fit(trainData, responses)
        with open(self.model_file, "wb") as f:
            pickle.dump(train_svm, f)

    def test(self, folder_path):
        #svm_model = cv2.ml.SVM_create()
        # svm_model.load('svm_data.dat')
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
        for j in range(1, 15):
            for i in range(21, 26):
                img_path = folder_path+str(j)+"_"+str(i)+".jpg"
                print(img_path)
                try:
                    fea, farea, skinarea, fcont, pix_to_cm = readFeatureImg(
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
            volume = getVolume(result[i], fruit_areas[i],
                               skin_areas[i], pix_cm[i], fruit_contours[i])
            mass, cal, protein, carb, fat, cal_100, protein_100, carb_100, fat_100 = getMacros(result[i], volume)
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
            # if mask[i][0] == False:
                #print ("(Actual Reponse)", responses[i][0], "(Output)", result[i], image_names[i])

        #correct = np.count_nonzero(mask)
        #print (correct*100.0/result.size)
        print(right/len(mask))

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
        mass, cal, protein, carb, fat, cal_100, protein_100, carb_100, fat_100 = self.processor.getMacros(result[0], volume)

        return self.index2classification[int(result[0])], cal, protein, carb, fat
