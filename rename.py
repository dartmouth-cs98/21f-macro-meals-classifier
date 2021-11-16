import os

j = 1
for filename in os.listdir("train_images/Banana"):
    if filename == "script.sh" or filename == "script.sh~":
        continue
    else:
        os.rename("train_images/Banana/" + filename,"train_images/Banana/"+"2_"+str(j)+".jpg")
        j+=1

