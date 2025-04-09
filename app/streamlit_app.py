import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pickle
from utils.logger import setup_logging

logging = setup_logging()

st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")
st.title("Real Estate Price Predictor")
st.write("This app predicts real estate prices using a trained linear regression model.")

# Load model
try:
    with open("real_estate_model.pickle", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("Failed to load model.")
    logging.error(f"Model loading failed: {e}")
    st.stop()

# Form input
with st.form("prediction_form"):
    st.subheader("Enter Property Information")

    year_sold = st.number_input("Year Sold", min_value=2000, max_value=2025, step=1)
    sqft = st.number_input("Square Footage (sqft)", min_value=100, step=10)
    beds = st.number_input("Number of Bedrooms", min_value=0, step=1)

    bath_options = [i * 0.5 for i in range(0, 11)]
    baths = st.selectbox("Number of Bathrooms", bath_options, index=bath_options.index(1.0))

    lot_size = st.number_input("Lot Size (sqft)", min_value=0, step=10)
    submitted = st.form_submit_button("Predict Price")

if submitted:
    try:
        features = np.array([[year_sold, sqft, beds, baths, lot_size]])
        result = model.predict(features)[0]
        st.success(f" Estimated Price: ${result:,.2f}")
        logging.info("Prediction made successfully.")
    except Exception as e:
        st.error("Prediction failed.")
        logging.error(f"Prediction error: {e}")

# Optional chart
st.write("---")
st.write("Feature Importance:")
try:
    st.image("feature_importance.png")
except:
    st.info("Feature importance image not found.")
    logging.warning("Feature importance image not found. Please check the file path.")
st.write("This chart shows the absolute coefficient values of the features used in the model. Higher values indicate more important features.")