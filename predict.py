import joblib
import numpy as np
import pandas as pd


def predict_wine_quality(input_features):
    model = joblib.load("wine_model.pkl")
    scaler = joblib.load("scaler.pkl")

    feature_names = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
        'density', 'pH', 'sulphates', 'alcohol'
    ]

    input_df = pd.DataFrame([input_features], columns = feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    return prediction[0]

if __name__ == "__main__":
    sample = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    quality = predict_wine_quality(sample)
    print("Predict wine quality: ", round(quality, 2))