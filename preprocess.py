import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path, y_train=None):
    df = pd.read_csv(csv_path)

    X = df.drop("quality", axis = 1)
    y = df["quality"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, t_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

    return X_train, X_test, y_train, y_test, scaler