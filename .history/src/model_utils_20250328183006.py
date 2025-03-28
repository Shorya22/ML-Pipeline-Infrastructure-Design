import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from datetime import datetime

# Load Dataset
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df

# Data Preprocessing
def preprocess_data(df):
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

# Model Training
def train_model():
    df = load_data()
    df = preprocess_data(df)
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        print(f"F1 Score: {f1:.4f}")
        
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        model_version = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(model, os.path.join("models", model_version))
        print(f"Model saved as {model_version}")
    
    return model_version, f1

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    mlflow.set_tracking_uri("./mlruns")  # Set local MLflow tracking
    train_model()