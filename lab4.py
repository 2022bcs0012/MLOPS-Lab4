import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.metrics import accuracy_score

# Metadata
NAME = "Shanik Hubert"
ROLL_NO = "2022BCS0012"

# Paths
DATA_PATH = "data/winequality-red.csv"
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

def train_model(model, experiment_name, threshold=0.5):
    df = pd.read_csv(DATA_PATH, sep=";")
    y = df["quality"]
    X = df.drop(columns=["quality"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy with threshold
    correct = (abs(y_pred - y_test) < threshold).sum()
    accuracy = correct / len(y_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "name": NAME,
        "roll_no": ROLL_NO,
        "experiment_name": experiment_name,
        "MSE": mse,
        "R2": r2,
        "Accuracy": accuracy
    }

    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Experiment: {experiment_name}")
    print(f"MSE: {mse}, R2: {r2}, Accuracy (±{threshold}): {accuracy}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = DecisionTreeRegressor(
        max_depth=15,        # limit depth to prevent overfitting
        min_samples_leaf=5, # minimum samples per leaf
        random_state=42
    )
    
    # model = LinearRegression()
    experiment_name = type(model).__name__

    train_model(model, experiment_name)
