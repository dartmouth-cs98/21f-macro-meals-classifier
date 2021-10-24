from flask import Flask, render_template, request
import json

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        pass
    else: 
        msg = {
            "Message": "Welcome to Flask Server"
        }
        return json.dumps(msg)



if __name__ == "__main__":
    app.run(debug=True)