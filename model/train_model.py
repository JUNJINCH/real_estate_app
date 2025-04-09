import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from data_prep.data_loader import load_data, preprocess_data
from utils.logger import setup_logging

logging = setup_logging()

def train_and_evaluate(data_path="data/real_estate.csv"):
    df = load_data(data_path)
    df = preprocess_data(df)

    X = df[["year_sold", "sqft", "beds", "baths", "lot_size"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    logging.info(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    logging.info(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

    return model, X_train


def plot_feature_importance(model, X_train, output_path="feature_importance.png"):
    importance = model.coef_
    features = X_train.columns

    # Combine features and coefficients into DataFrame for better interpretability
    importance_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": importance,
        "Importance (abs)": np.abs(importance)
    }).sort_values(by="Importance (abs)", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["Feature"], importance_df["Importance (abs)"])
    plt.xlabel("Absolute Coefficient Value")
    plt.title("Feature Importance (Linear Regression)")
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Feature importance saved to {output_path}")
