import streamlit as st 
import requests

# 1. Configure the API endpoint URL

API_URL = "http://backend:8000/predict"

st.title("Car Price Predictor")
st.write("Enter the details of the car to predict its price.")

# 2. Input fields

name = st.text_input("Car Name",  placeholder= "Maruti Swift")
company = st.text_input("Company", placeholder= "Maruti")
year = st.number_input("Year of Manufacture", min_value=1886, max_value=2026, step=1)
kms_driven = st.number_input("Kilometers Driven(kms)", min_value=0, step=100)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

# 3. Predict button

if st.button("Predict Price"):
    if not name or not company or not year or not kms_driven:
        st.error("Please fill in all the fields.")
    else:
        payload = {
            "name" : name,
            "company" : company,
            "year" : int(year),
            "kms_driven" : int(kms_driven),
            "fuel_type" : fuel_type
        }
        try:
            response = requests.post(API_URL, json = payload)
            if response.status_code == 200:
                result = response.json()["predicted_price"]
                st.success(f"Predicted Price: Rs. {result:,.0f}")
            else:
                st.error(f"Error : {response.status_code} - {response.text}")
        except Exception as e:
            st.error("Error connecting to API")
