from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model_path = "/app/models/model.pkl"
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
