import os
import sys
import numpy as np
import logging
from src.data_loader import load_eeg_data
from src.preprocessing import preprocess_eeg_data
import src.feature_extraction
import src.model_training
import src.evaluation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATA_DIR = "data/"
MODEL_PATH = "models/random_forest_model.pkl"

# Parameters
FS = 250  # Sampling frequency
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    """
    Main function to run the EEG attention classification workflow.
    """
    try:
        # Step 1: Load the data
        logging.info("Loading EEG data...")
        all_data = load_eeg_data(DATA_DIR)

        # Example: Access data for a specific subject and paradigm
        subject_name = 'Subject_1'  # Adjust as per your actual data
        paradigm = 'oddball'  # Adjust based on paradigm available in your data

        # Ensure that the subject and paradigm exist in the data
        if subject_name in all_data and paradigm in all_data[subject_name]:
            eeg_data = all_data[subject_name][paradigm]['eeg']
            labels = all_data[subject_name][paradigm]['markers']

            logging.info(f"Loaded data for {subject_name} - {paradigm}")
            logging.info(f"EEG data shape: {eeg_data.shape}")
            logging.info(f"Labels shape: {labels.shape}")

            # Step 2: Preprocess the data
            logging.info("Preprocessing EEG data...")
            preprocessed_data, _ = preprocess_eeg_data(eeg_data, labels, FS)
            logging.info(f"Preprocessed data shape: {preprocessed_data.shape}")

            # Ensure that the number of samples in preprocessed data matches the labels
            if preprocessed_data.shape[0] != len(labels):
                logging.error(f"Mismatch between preprocessed data and labels. Data shape: {preprocessed_data.shape}, Labels shape: {len(labels)}")
                return

            # Step 3: Extract features
            logging.info("Extracting features...")
            features = src.feature_extraction.extract_features(preprocessed_data, FS)
            logging.info(f"Feature matrix shape: {features.shape}")

            # Ensure features and labels are aligned
            if features.shape[0] != len(labels):
                logging.error(f"Mismatch between features and labels. Features shape: {features.shape}, Labels shape: {len(labels)}")
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

            # Evaluate metrics and confusion matrix
            src.evaluation.evaluate_model(y_true, y_pred)

            # Step 7: Visualize confusion matrix
            logging.info("Plotting confusion matrix...")
            class_names = [f"Class {i}" for i in np.unique(labels)]
            src.evaluation.plot_confusion_matrix(y_true, y_pred, class_names=class_names)

        else:
            logging.error(f"Subject '{subject_name}' or paradigm '{paradigm}' not found in the data.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
