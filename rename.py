import os
import cv2
from models.Classifier import Classifier

FOLDER_PATH = "train_images_unit"

def reindex(folder_path):

    classifier = Classifier()

    for food_type in os.listdir(folder_path):
        print(food_type)
        try:
            food_index = classifier.index2classification.index(food_type.lower())
        except ValueError:
            continue
        i = 1
        print(i)
        for filename in os.listdir(folder_path + "/" + food_type):
            if filename == "script.sh" or filename == "script.sh~":
                continue
            os.rename(folder_path + "/" + food_type + "/" + filename, folder_path + "/" + food_type + "/" + str(food_index) + "_" + str(i)+".jpg")
            i += 1

reindex(FOLDER_PATH)
