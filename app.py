from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "diabetes_model.sav"), "rb")
)

heart_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "heart_disease_model.sav"), "rb")
)

parkinsons_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "parkinsons_model.sav"), "rb")
)

pcos_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "pcos_model.sav"), "rb")
)

kidney_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "kidney_model.sav"), "rb")
)

liver_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "liver_model.sav"), "rb")
)

thyroid_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "thyroid_model.sav"), "rb")
)

anemia_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "anemia_model.sav"), "rb")
)

alzheimers_model = pickle.load(
    open(os.path.join(BASE_DIR, "saved_models", "alzheimers_model.sav"), "rb")
)

DIABETES_MEANS = [3, 120.5, 72.3, 23.1, 80.2, 24.6, 0.47, 33.4]

PCOS_MEANS = [26.4, 24.8, 0.6, 52.1, 9.4]

KIDNEY_MEANS = [48, 1, 80, 1.02, 0, 0, 0, 1, 0, 0,1, 135, 4.5, 15, 45, 5.0, 140, 4.0,15, 5, 1, 0, 0, 0]

LIVER_MEANS = [45, 1, 24, 2, 0, 0, 5, 0, 0, 30]

THYROID_MEANS = [35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

ANEMIA_MEANS = [0, 13.5, 30, 34, 90]

ALZHEIMERS_MEANS = [70, 1, 14, 2, 2, 0, 25, 0, 1]

def fill_with_means(data, means):

    final = []

    for i, val in enumerate(data):

        if val == 0 or val is None:
            final.append(means[i])
        else:
            final.append(val)

    return final

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/parkinsons")
def parkinsons():
    return render_template("parkinsons.html")

@app.route("/PCOS")
def pcos():
    return render_template("PCOS.html")

@app.route("/kidney")
def kidney():
    return render_template("kidney.html")

@app.route("/liver")
def liver():
    return render_template("liver.html")

@app.route("/thyroid")
def thyroid():
    return render_template("thyroid.html")

@app.route("/anemia")
def anemia():
    return render_template("anemia.html")

@app.route("/autism")
def autism():
    return render_template("autism.html")

@app.route("/alzheimers")
def alzheimers():
    return render_template("alzheimers.html")

@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():

    data = request.json["features"]

    data = fill_with_means(data, DIABETES_MEANS)

    pred = diabetes_model.predict([data])[0]

    return jsonify({"result": int(pred)})

@app.route("/predict/heart", methods=["POST"])
def predict_heart():

    data = request.json["features"]

    pred = heart_model.predict([data])[0]

    return jsonify({"result": int(pred)})

@app.route("/predict/parkinsons", methods=["POST"])
def predict_parkinsons():

    data = request.json["features"]

    pred = parkinsons_model.predict([data])[0]

    return jsonify({"result": int(pred)})

@app.route("/predict/pcos", methods=["POST"])
def predict_pcos():

    data = request.json["features"]

    data = fill_with_means(data, PCOS_MEANS)

    pred = pcos_model.predict([data])[0]

    return jsonify({"result": int(pred)})

@app.route("/predict/kidney", methods=["POST"])
def predict_kidney():

    data = request.json["features"]

    data = fill_with_means(data, KIDNEY_MEANS)

    pred = kidney_model.predict([data])[0]

    return jsonify({"result": int(pred)})

@app.route("/predict/liver", methods=["POST"])
def predict_liver():

    data = request.json["features"]

    data = fill_with_means(data, LIVER_MEANS)

    pred = liver_model.predict([data])[0]

    return jsonify({"result": int(pred)})

@app.route("/predict/thyroid", methods=["POST"])
def predict_thyroid():

    data = request.json["features"]

    data = fill_with_means(data, THYROID_MEANS)

    pred = thyroid_model.predict([data])[0]

    return jsonify({"result": int(pred)})

@app.route("/predict/anemia", methods=["POST"])
def predict_anemia():

    data = request.json["features"]

    data = fill_with_means(data, ANEMIA_MEANS)

    pred = anemia_model.predict([data])[0]

    return jsonify({"result": int(pred)})

@app.route("/predict/alzheimers", methods=["POST"])
def predict_alzheimers():

    data = request.json["features"]

    data = fill_with_means(data, ALZHEIMERS_MEANS)

    pred = alzheimers_model.predict([data])[0]

    return jsonify({"result": int(pred)})

if __name__ == "__main__":
    app.run(debug=True)