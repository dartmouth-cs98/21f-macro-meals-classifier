from flask import Flask, render_template, request
from classify import classify
import json



app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        pass
    else:
        test_img = "./Dataset/images/Test_Images/1_21.jpg"
        classification = str(classify_user_img(test_img))
        if classification:
            msg = str(classification)
        else:
            msg = {
                "Classification failed"
            }
        return json.dumps(msg)

def classify_user_img(test_img):
    return classify(test_img)

if __name__ == "__main__":
    app.run(debug=True)
