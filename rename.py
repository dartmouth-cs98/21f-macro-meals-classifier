import os
import cv2
from models.Classifier import Classifier

FOLDER_PATH = "test_images2"
DELETED_INDEX = 4

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

    # print(oldname2newname
    #
    # for filename in oldname2newname:
    #     new_file_name = oldname2newname[filename]


# reindex(FOLDER_PATH)
reindex_testfolder(FOLDER_PATH)
