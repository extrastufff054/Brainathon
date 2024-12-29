import os
import sys
import numpy as np
import logging
import pandas as pd
from src.data_loader import load_eeg_data, normalize_paradigm_name
from src.preprocessing import preprocess_eeg_data
import src.feature_extraction
import src.model_training
import src.evaluation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATA_DIR = "data/"
MODEL_PATH = "output/models/random_forest_model.pkl"

# Parameters
FS = 250  # Sampling frequency
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_available_paradigms(all_data, subject_name):
    """
    Get available paradigms for a given subject.
    """
    if subject_name in all_data:
        return list(all_data[subject_name].keys())
    else:
        return []

def main():
    """
    Main function to run the EEG attention classification workflow.
    """
    try:
        # Step 1: Load the data
        logging.info("Loading EEG data...")
        all_data = load_eeg_data(DATA_DIR)

        # Specify subject and paradigm
        subject_name = 'Subject_1'  # Adjust as needed
        available_paradigms = load_available_paradigms(all_data, subject_name)

        if not available_paradigms:
            logging.error(f"No paradigms available for {subject_name}. Exiting.")
            return

        paradigm = 'oddball'  # Adjust based on your data
        normalized_paradigm = normalize_paradigm_name(paradigm)

        if normalized_paradigm not in available_paradigms:
            logging.warning(f"Paradigm '{paradigm}' not found for {subject_name}. Available paradigms: {available_paradigms}")
            return

        eeg_data = all_data[subject_name][normalized_paradigm]['eeg']
        labels = all_data[subject_name][normalized_paradigm]['markers']

        if eeg_data.empty or labels.size == 0:
            logging.error(f"No data found for paradigm {paradigm} in {subject_name}. Exiting.")
            return

        logging.info(f"Loaded data for {subject_name} - {paradigm}")
        logging.info(f"EEG data shape: {eeg_data.shape}")
        logging.info(f"Labels shape: {len(labels)}")

        # Step 2: Preprocess the data
        logging.info("Preprocessing EEG data...")
        preprocessed_data, labels = preprocess_eeg_data(eeg_data, labels, FS)
        logging.info(f"Preprocessed data shape: {preprocessed_data.shape}")

        if preprocessed_data.shape[0] == 0 or len(labels) == 0:
            logging.error("Preprocessed data or labels are empty. Exiting...")
            return

        # Step 3: Extract features
        logging.info("Extracting features...")
        features = src.feature_extraction.extract_features(preprocessed_data, FS)
        logging.info(f"Feature matrix shape: {features.shape}")

        if features.shape[0] == 0:
            logging.error("Feature extraction returned empty features. Exiting...")
            return

        # Step 4: Train the model
        logging.info("Training the model...")
        results = src.model_training.train_model(features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        trained_model = results["model"]
        logging.info(f"Model accuracy: {results['metrics']['accuracy']}")

        # Step 5: Save the model
        logging.info("Saving the trained model...")
        src.model_training.save_model(trained_model, MODEL_PATH)

        # Step 6: Evaluate the model
        logging.info("Evaluating the model...")
        y_true = results["y_test"]
        y_pred = results["y_pred"]

        src.evaluation.evaluate_model(y_true, y_pred)

        # Step 7: Visualize confusion matrix
        logging.info("Plotting confusion matrix...")
        class_names = [f"Class {i}" for i in np.unique(labels)]
        src.evaluation.plot_confusion_matrix(y_true, y_pred, class_names=class_names)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
