import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

# Loading the saved XGBoost model and the OneHotEncoder
model = joblib.load('xgboost_model_onehot2.pkl')
encoder = joblib.load('onehot_encoder.pkl') 

# Title
st.title("Belarus Car Price Prediction")

# User input features
make = st.selectbox("Make", options=['toyota', 'Honda', 'BMW', 'Audi', 'Mercedes'])
model_name = st.selectbox("Model", options=['fortuner', 'Civic', 'X5', 'A4', 'C-Class'])
year = st.slider("Year", min_value=1990, max_value=2024, value=2015)
condition = st.selectbox("Condition", options=['with mileage', 'New', 'Used', 'Certified Pre-Owned'])
mileage = st.number_input("Mileage (kilometers)", value=9500.0)
fuel_type = st.selectbox("Fuel Type", options=['petrol', 'Diesel', 'Electric', 'Hybrid'])
volume = st.number_input("Engine Volume (cm³)", value=1500.0)
color = st.selectbox("Color", options=['Red', 'Blue', 'Black', 'white', 'Silver'])
transmission = st.selectbox("Transmission", options=['mechanics', 'Automatic', 'Manual'])
drive_unit = st.selectbox("Drive Unit", options=['front-wheel drive', 'Rear-Wheel Drive', 'All-Wheel Drive'])
segment = st.selectbox("Segment", options=['B', 'C', 'D', 'E', 'F'])

# When the "Predict" button is clicked
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'make': [make],
        'model': [model_name],
        'year': [year],
        'condition': [condition],
        'mileage(kilometers)': [mileage],
        'fuel_type': [fuel_type],
        'volume(cm3)': [volume],
        'color': [color],
        'transmission': [transmission],
        'drive_unit': [drive_unit],
        'segment': [segment]
    })

    # Apply One-Hot Encoding to the input data
    input_data_encoded = encoder.transform(input_data)

    
    # Predict the price
    prediction = model.predict(input_data_encoded)
    predicted_price = f"{prediction[0]:,.2f}"

    # Display the predicted price in big, bold, and pink text
    st.markdown(
        f"<h1 style='color: #FF1493; font-weight: bold; font-size: 36px;'>Predicted Price (USD): {predicted_price}</h1>",
        unsafe_allow_html=True
    )