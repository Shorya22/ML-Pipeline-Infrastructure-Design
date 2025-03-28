from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os

app = FastAPI()

# Ensure the model exists
model_path = "/app/models/model.pkl"

if not os.path.exists(model_path):
    raise RuntimeError(f"❌ Model file not found at {model_path}. Make sure you trained it inside the container.")

# Load the trained model
model = joblib.load(model_path)

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running!"}

@app.post("/predict/")
def predict(data: dict):
    # Convert input data to DataFrame
    df = pd.DataFrame([data])

    # Ensure input has the correct columns (modify based on preprocessing)
    expected_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Fill missing columns with 0

    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
