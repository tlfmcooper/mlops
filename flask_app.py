from flask import Flask, request, jsonify
from logzero import logger, logfile
from pickle import load, dump
from datetime import datetime
import numpy as np


logfile(filename="app.log", maxBytes=1000000, backupCount=3, disableStderrLogger=True)


model = load(open("model.pkl", "rb"))

app = Flask(__name__)
logger.info("Starting the Flask application")


@app.route("/")
def health_check():
    return "OK"


@app.route("/predict", methods=["GET", "POST"])  # Allow GET requests
def predict():
    try:
        if request.method == "GET":
            # Get parameters from the query string
            input_data = request.args
        else:
            input_data = request.json

        if not input_data:
            return jsonify({"message": "No input data provided"}), 400

        values_list = [list(input_data.values())]
        prediction = model.predict(values_list)
        if prediction[0] == 0:
            prediction_result = "Not likely to purchase"
        elif prediction[0] == 1:
            prediction_result = "Likely to purchase"

        current_datetime = datetime.now().strftime("%Y-%m-%d %H: %M: %S")
        logger.info(f"{current_datetime} - Prediction: {prediction_result}")
        return jsonify({"prediction": prediction_result}), 200

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
