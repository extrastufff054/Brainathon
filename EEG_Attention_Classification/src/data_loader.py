import os
import pandas as pd
import logging

def load_subject_data(subject_folder):
    """
    Load EEG and marker files for a given subject folder.
    """
    eeg_files = [f for f in os.listdir(subject_folder) if 'eeg' in f]
    marker_files = [f for f in os.listdir(subject_folder) if 'markers' in f]

    data = {}

    for eeg_file, marker_file in zip(eeg_files, marker_files):
        eeg_path = os.path.join(subject_folder, eeg_file)
        marker_path = os.path.join(subject_folder, marker_file)

        try:
            # Load EEG data
            eeg_data = pd.read_csv(eeg_path)

            # Load markers and check for required columns
            markers = pd.read_csv(marker_path)

            # Ensure the marker file has required columns: 'timestamp' and 'marker'
            if 'timestamp' not in markers.columns or 'marker' not in markers.columns:
                logging.error(f"Missing required columns in markers file: {marker_file}")
                continue  # Skip this file if columns are missing

            # Optionally, rename 'marker' to 'event' for consistency with the rest of the pipeline
            markers.rename(columns={'marker': 'event'}, inplace=True)

            markers = markers[['timestamp', 'event']].values  # Keep only relevant columns
            paradigm_name = eeg_file.split('_')[2]  # Assumes format: `Subject_1_paradigm_eeg.csv`

            data[paradigm_name] = {'eeg': eeg_data, 'markers': markers}

        except Exception as e:
            logging.error(f"Error loading files for {marker_file}: {e}")
            continue  # Skip this file if there's any other error

    return data

def load_eeg_data(dataset_dir):
    """
    Load data for all subjects in the dataset directory.

    Args:
        dataset_dir (str): Path to the dataset directory containing subject folders.

    Returns:
        dict: A nested dictionary containing data for all subjects.
              Example structure:
              {
                  'Subject_1': {...},
                  'Subject_2': {...},
                  ...
              }
    """
    subject_folders = [os.path.join(dataset_dir, subj) for subj in os.listdir(dataset_dir)]
    
    all_data = {}
    for subject_folder in subject_folders:
        subject_name = os.path.basename(subject_folder)
        subject_data = load_subject_data(subject_folder)
        
        if subject_data:  # Only add valid data (not empty or None)
            all_data[subject_name] = subject_data
        else:
            logging.error(f"No valid data found for {subject_name}, skipping...")

    return all_data
