import os

# j = 1
# for filename in os.listdir("train_images/Banana"):
    # if filename == "script.sh" or filename == "script.sh~":
    #     continue
#     else:
#         os.rename("train_images/Banana/" + filename,"train_images/Banana/"+"2_"+str(j)+".jpg")
#         j+=1

j = 81
for filename in os.listdir("train_images_2/zucchini_dark_1"):
    # if filename == "script.sh" or filename == "script.sh~":
    #     continue
    # else:
    os.rename("train_images_2/zucchini_dark_1/" + filename, "train_images_2/zucchini_dark_1/"+"18_"+str(j)+".jpg")
    j += 1
