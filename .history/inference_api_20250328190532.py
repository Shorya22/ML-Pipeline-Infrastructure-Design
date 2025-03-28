from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
MODEL_PATH = "model_v1.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Train the model first!")

# Define request model
class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

# Feature transformation function
def preprocess_input(data: PassengerData):
    """Preprocesses input data similar to training."""
    df = pd.DataFrame([data.dict()])
    
    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    # Fill missing values
    df.fillna(df.median(), inplace=True)
    
    return df.values  # Return NumPy array

# Root endpoint
@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running!"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: PassengerData):
    try:
        input_data = preprocess_input(data)
        prediction = model.predict(input_data)[0]
        return {"prediction": int(prediction)}  # Convert to int (0 or 1)
    except Exception as e:
        return {"error": str(e)}
