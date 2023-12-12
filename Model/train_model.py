from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error

def polynomial_data(X_train, X_test, X_val, degree = 2):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    X_val_poly = poly.transform(X_val)

    return X_train_poly, X_test_poly, X_val_poly, poly

def train_polynomial_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model