import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model(features, labels, test_size=0.2, random_state=42):
    """
    Train a Random Forest model for multiclass classification.

    Args:
        features (ndarray): Feature matrix (num_samples x num_features).
        labels (ndarray): Labels corresponding to the features.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        dict: A dictionary containing the trained model, test predictions, and evaluation metrics.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics
    }

def save_model(model, model_path):
    """
    Save the trained model to a file.

    Args:
        model (sklearn model): Trained machine learning model.
        model_path (str): Path to save the model file.
    """
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """
    Load a trained model from a file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        sklearn model: Loaded machine learning model.
    """
    return joblib.load(model_path)
