import pickle
from flask import Flask, request, jsonify


with open("dv.bin", "rb") as f:
    dv = pickle.load(f)

with open("model2.bin", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    X = dv.transform([data])
    y_pred = model.predict_proba(X)
    result = {"probability": float(y_pred[0][1])}

    return jsonify(result)
