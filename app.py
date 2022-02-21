from flask import Flask, render_template, request
import json
import numpy as np
import cv2
import os
import requests
import time
from models.Classifier import Classifier

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def classify_img():
	classifier = Classifier()
	if request.method == 'POST':
		classifier = Classifier()
		s3_url = request.json['url']
		img_filepath = download_url(s3_url)
		try:
			msg = classifier.classify(img_filepath)
			os.remove(img_filepath)
		except TypeError:
			msg = "Classification failed"

		return json.dumps(msg)
	else:
		classifier = Classifier()
		img_filepath = "train_images_broken/Pear/17_1.jpg"
		# /*** # cv2 error for 2_50 ***/
		# try:
		msg = classifier.classify(img_filepath)
		# except TypeError:
			# msg = "Classification failed"
		return json.dumps(msg)

def download_url(img_url):
	img_data = requests.get(img_url).content

	img_filepath = "user_images/" + str(time.time()) + ".jpg"
	with open(img_filepath, 'wb') as handler:
		handler.write(img_data)
	return img_filepath

if __name__ == "__main__":
	app.run(debug=True)
