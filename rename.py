import os
import cv2
from models.Classifier import Classifier

FOLDER_PATH = "train_images_unit3"
DELETED_INDEX = 4

def reindex(folder_path):

    classifier = Classifier()

    oldname2newname = {}

    j = 0
    for food_type in os.listdir(folder_path):
        print(food_type)
        try:
            food_index = classifier.index2classification.index(food_type.lower())
        except ValueError:
            continue
        for filename in os.listdir(folder_path + "/" + food_type):
            if filename == "script.sh" or filename == "script.sh~":
                continue
            os.rename(folder_path + "/" + food_type + "/" + filename, folder_path + "/" + food_type + "/" + str(j) + ".jpg")
            j += 1

    for food_type in os.listdir(folder_path):
        try:
            food_index = classifier.index2classification.index(food_type.lower())
        except ValueError:
            continue
        i = 1
        print(i)
        for filename in os.listdir(folder_path + "/" + food_type):
            if filename == "script.sh" or filename == "script.sh~":
                continue
            oldname = folder_path + "/" + food_type + "/" + filename
            newname = folder_path + "/" + food_type + "/" + str(food_index) + "_" + str(i)+".jpg"
            oldname2newname[oldname] = newname
            i += 1

    print(oldname2newname)

    for oldname in oldname2newname:
        newname = oldname2newname[oldname]
        os.rename(oldname, newname)


def remove_icloud(folder_path):

    classifier = Classifier()

    files_to_delete = []

    for food_type in os.listdir(folder_path):
        print(food_type)
        try:
            food_index = classifier.index2classification.index(food_type.lower())
        except ValueError:
            continue
        for filename in os.listdir(folder_path + "/" + food_type):
            if filename == "script.sh" or filename == "script.sh~":
                continue
            split_l = filename.split(" ")
            if len(split_l) > 1:
                files_to_delete.append(folder_path + "/" + food_type + "/" + filename)


    for file in files_to_delete:
        os.remove(file)

# reindex testing images folder given single deletion
# buggy right now
def reindex_testfolder(folder_path):

    filenames = os.listdir(folder_path)
    filenames = sorted(filenames, key = lambda x: int(x.split("_")[0]))

    for filename in filenames:
        split_file_name = filename.split("_")
        index, rest_of_name = int(split_file_name[0]), split_file_name[1]
        if index <= DELETED_INDEX:
            continue
        # move everything down 1
        index -= 1
        new_filename = str(index) + "_" + rest_of_name
        # oldname2newname[filename] = new_filename
        os.rename(folder_path + "/" + filename, folder_path + "/" + new_filename)


reindex(FOLDER_PATH)
# remove_icloud(FOLDER_PATH)
# reindex_testfolder(FOLDER_PATH)
