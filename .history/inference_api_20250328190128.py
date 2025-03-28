from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd

# Load the trained model
MODEL_PATH = "model_v1.pkl"

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise Exception("Model file not found. Train the model first!")

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running!"}

@app.post("/predict/")
def predict(data: dict):
    """
    Accepts a JSON input with passenger features and returns a prediction.
    Example input:
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
    """
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Feature transformations (same as training phase)
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})
        df.fillna(0, inplace=True)

        # Predict
        prediction = model.predict(df)
        return {"prediction": int(prediction[0])}  # 1 = Survived, 0 = Not Survived

    except Exception as e:
        return {"error": str(e)}
