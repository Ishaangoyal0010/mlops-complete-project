import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# 1. Load Model

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "artifacts", "best_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# 2. App

app = FastAPI(title="Car Price Predictor")

# 3. Input Schema

class CarInput(BaseModel):
    name:str
    company:str
    kms_driven:int
    fuel_type:str
    year:int

# 4. Routes

@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running."}


@app.post("/predict")
def predict(car: CarInput):
    current_year = 2026
    car_age = current_year - car.year

    input_df = pd.DataFrame([{
        "name": car.name,
        "company": car.company,
        "kms_driven": car.kms_driven,
        "fuel_type": car.fuel_type,
        "car_age": car_age,
    }])

    predicted_price = model.predict(input_df)[0]
    predicted_price = max(0, predicted_price)   # no negative prices

    return {
        "predicted_price": round(predicted_price, 2),
        "currency": "INR"
    }