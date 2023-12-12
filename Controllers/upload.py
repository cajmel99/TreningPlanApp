from flask import flash
from flask import Blueprint, redirect, request
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from Model.train_model import train_polynomial_regression, polynomial_data
from model.preprocess import preprocess_data, split_data
from sklearn.preprocessing import StandardScaler
import numpy as np

upload_blueprint = Blueprint('upload', __name__)

ALLOWED_EXTENSIONS = {'csv'}
UPLOAD_FOLDER = 'uploaded_files'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_blueprint.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'csvfile' not in request.files:
        return 'No file part', 400
    file = request.files['csvfile']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(file_path)

        df = preprocess_data(file_path)
        X_train, X_test, X_val, y_train, y_test, y_val, scaler = split_data(df)
        X_train_poly, X_test_poly, X_val_poly, poly = polynomial_data(X_train, X_test, X_val)

        model = train_polynomial_regression(X_train_poly, y_train)
        model_path = 'Model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save the objects with their respective paths
        joblib.dump(model, os.path.join(model_path, 'model.pkl'))
        joblib.dump(scaler, os.path.join(model_path, 'scaler.joblib'))
        joblib.dump(poly, os.path.join(model_path, 'poly.joblib'))
        
        # Flash message
        #flash(f"Model trained with MSE: {mse}", 'success')

        # Redirect to the prediction form
        return redirect('/predict')
    
    return 'Incorrect file type. The file must be a CSV file downloaded from the Garmin Connect website.', 400
