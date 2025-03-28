# Install required packages
# Install required packages
!pip install pandas scikit-learn mlflow

# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle
import os
import mlflow

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("Dataset loaded:", df.shape)

# Simple preprocessing
def preprocess_data(df, is_training=True):
    # Copy dataframe
    df_processed = df.copy()
    
    # Handle missing values
    df_processed['Age'].fillna(df_processed['Age'].mean(), inplace=True)
    df_processed['Embarked'].fillna('S', inplace=True)
    
    # Convert categorical to numeric
    df_processed['Sex'] = df_processed['Sex'].map({'male': 0, 'female': 1})
    df_processed['Embarked'] = df_processed['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    return df_processed[features]

# Training function
def train_model(df):
    # Preprocess data
    X = preprocess_data(df)
    y = df['Survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    # Log with MLflow
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model_v1")
    
    return model, f1

# Save model
def save_model(model, version="v1"):
    os.makedirs("/content/models", exist_ok=True)
    model_path = f"/content/models/model_{version}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model_path

# Main execution
def main():
    # Set MLflow tracking
    mlflow.set_tracking_uri("file:/content/mlruns")
    mlflow.set_experiment("titanic_simple")
    
    # Train model
    print("Training model...")
    model, f1_score = train_model(df)
    print(f"F1 Score: {f1_score:.4f}")
    
    # Save model
    model_path = save_model(model)
    print(f"Model saved to: {model_path}")
    
    # Test prediction
    sample = preprocess_data(df.iloc[:1])
    prediction = model.predict(sample)
    print(f"Sample prediction: {prediction[0]}")

# Run everything
if __name__ == "__main__":
    main()