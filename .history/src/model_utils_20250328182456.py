import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import mlflow
import pickle
import os

class TitanicModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.version = "v1"

    def preprocess_data(self, df):
        # Handle missing values
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

        # Feature engineering
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Encode categorical variables
        categorical_cols = ['Sex', 'Embarked']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
                
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']
        return df[features]

    def train(self, df, target_col='Survived'):
        X = self.preprocess_data(df)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(self.model, f"model_{self.version}")
        
        return f1

    def predict(self, data):
        processed_data = self.preprocess_data(data)
        return self.model.predict(processed_data)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'encoders': self.label_encoders, 'version': self.version}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoders = data['encoders']
            self.version = data['version']