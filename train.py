from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from preprocess import load_and_preprocess_data

def train_model():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("winequality-red.csv")

    model = RandomForestRegressor(random_state = 42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("MSE : ", mean_squared_error(y_test, preds))
    print("R2 Score: ", r2_score(y_test, preds))

    joblib.dump(model, "wine_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved.")



if __name__ == "__main__":
    train_model()