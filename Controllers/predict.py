from flask import Blueprint, request, render_template
import pandas as pd
from Model.train_model import train_polynomial_regression
from Model.preprocess import convert_time_to_seconds
from joblib import load
import numpy as np
import os

def input_data(X_input, scaler, poly):
    X_input = scaler.transform(X_input)
    X_input_poly = poly.transform(X_input)
    return X_input, X_input_poly

predict_blueprint = Blueprint('predict', __name__)

@predict_blueprint.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')

    model_path = 'Model'

    # Load the objects from their respective files
    model = load(os.path.join(model_path, 'model.pkl'))
    scaler = load(os.path.join(model_path, 'scaler.joblib'))
    poly = load(os.path.join(model_path, 'poly.joblib'))

    date_input = request.form.get('Date')
    try:
        date_value = pd.to_datetime(date_input)
        date_value_unix = int(date_value.timestamp())
    except ValueError:
        return 'Invalid Date format', 400

    # Process other form data
    data_keys = ['Distance', 'Max HR', 'Max Run Cadence', 'Total Ascent', 'Total Descent', 'Avg Stride Length']
    X_input = []
    for key in data_keys:
        value = request.form.get(key)
        if value is None or value == '':
            return f"Missing value for {key}", 400
        else:
            try:
                X_input.append(float(value))
            except ValueError:
                return f"Invalid value for {key}: {value}", 400

    X_input_df = pd.DataFrame([X_input], columns=data_keys)
    X_input_df.insert(0, 'Date', date_value_unix)

    print(X_input_df)
    X_input, X_input_poly = input_data(X_input_df, scaler, poly)
    print(X_input, 'second')

    prediction = model.predict(X_input_poly)
    predicted_avg_pace = round(prediction[0] / 60, 2)

    return render_template('prediction.html', predicted_avg_pace=predicted_avg_pace)


