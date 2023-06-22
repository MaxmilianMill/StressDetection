from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import sklearn
from ml import ml_predict
import json

app = Flask(__name__)
CORS(app)


@app.route("/", methods=['POST', 'GET'])
def hello_world():
    if request.method == 'POST':
        req = request.json

        prediction = ml_predict(request=req)

        print(prediction[0])

        response = dict(res=str(prediction[0]))

        print(response)

        return jsonify(response)
