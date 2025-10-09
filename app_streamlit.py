import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load artifacts
model = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

st.set_page_config(page_title="🏠 California Housing Price Predictor", layout="centered")

st.title("🏠 California Housing Price Prediction App")
st.write("Enter the details below to predict the **median house value** (in USD).")

# Input fields for user
longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=100, value=30)
total_rooms = st.number_input("Total Rooms", min_value=1, value=1000)
total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=200)
population = st.number_input("Population", min_value=1, value=500)
households = st.number_input("Households", min_value=1, value=300)
median_income = st.number_input("Median Income", min_value=0.0, value=4.0)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ("<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN")
)

# Predict button
if st.button("Predict House Price 💰"):
    try:
        input_data = pd.DataFrame({
            "longitude": [longitude],
            "latitude": [latitude],
            "housing_median_age": [housing_median_age],
            "total_rooms": [total_rooms],
            "total_bedrooms": [total_bedrooms],
            "population": [population],
            "households": [households],
            "median_income": [median_income],
            "ocean_proximity": [ocean_proximity]
        })

        # Apply preprocessing
        transformed_data = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(transformed_data)
        st.success(f"🏡 **Predicted Median House Value:** ${prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Built with ❤️ by Sohel Kumar — Data Science Project")
