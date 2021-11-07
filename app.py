from flask import Flask, render_template, request
from classify import classify
from Dataset.ClassificationMap import ClassificationMap
import json
import numpy as np
import cv2
import os
import requests
import time

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def classify_img():
    if request.method == 'POST':
        s3_url = request.form.get("s3_url")
        img_filepath = download_url(s3_url)
        classification = classify(img_filepath)
        if classification:
            classificationMap = ClassificationMap()
            translated_classification = classificationMap.get_classification(classification[0])
            msg = translated_classification
        else:
            msg = {
                "Classification failed"
            }
        os.remove(img_filepath)
        return json.dumps(msg)
    else:
        s3_url = "https://macro-meals-images.s3.amazonaws.com/2_5.jpg"
        img_filepath = download_url(s3_url)
        classification, calories = classify(img_filepath)
        if classification:
            classificationMap = ClassificationMap()
            translated_classification = classificationMap.get_classification(classification)
            msg = translated_classification
            # msg += str(calories_list[0])
            msg += str(calories)

            os.remove(img_filepath)
        else:
            msg = {
                "Classification failed"
            }
        return json.dumps(msg)

def download_url(img_url):
    img_data = requests.get(img_url).content

    img_filepath = "UserImages/" + str(time.time()) + ".jpg"
    with open(img_filepath, 'wb') as handler:
        handler.write(img_data)
    return img_filepath

if __name__ == "__main__":
    app.run(debug=True)
