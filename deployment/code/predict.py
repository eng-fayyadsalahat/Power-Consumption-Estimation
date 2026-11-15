import pickle
import pandas as pd

from flask import Flask, request, jsonify


# Load trained model
model_file = "power_model.pkl"

with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)


app = Flask("power-consumption")


# Expected model features
FEATURES = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "year",
    "Apparent_Power",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
    "time_period",
    "season"
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "model": "HistGradientBoostingRegressor"
    })


@app.route("/predict", methods=["POST"])
def predict():

    input_data = request.get_json()

    if not isinstance(input_data, dict):
        return jsonify({
            "error": "Input must be a JSON object."
        }), 400

    missing_features = [
        feature for feature in FEATURES
        if feature not in input_data
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features
        }), 400

    try:
        X = pd.DataFrame(
            [[input_data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        prediction = model.predict(X)

        result = {
            "global_active_power": float(prediction[0])
        }

        return jsonify(result)

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)