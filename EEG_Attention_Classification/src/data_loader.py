import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def normalize_paradigm_name(paradigm_name):
    """
    Normalize the paradigm name to handle variations in naming conventions.
    """
    return paradigm_name.replace('_paradigm', '').replace('_', '').lower()

def load_subject_data(subject_folder):
    """
    Load EEG and marker files for a given subject folder.
    """
    eeg_files = sorted([f for f in os.listdir(subject_folder) if 'eeg' in f and f.endswith('.csv')])
    marker_files = sorted([f for f in os.listdir(subject_folder) if 'markers' in f and f.endswith('.csv')])

    data = {}

    for eeg_file, marker_file in zip(eeg_files, marker_files):
        eeg_path = os.path.join(subject_folder, eeg_file)
        marker_path = os.path.join(subject_folder, marker_file)

        try:
            # Load EEG data
            eeg_data = pd.read_csv(eeg_path)

            # Validate presence of timestamp column in EEG data
            if 'timestamp' not in eeg_data.columns:
                logging.error(f"Missing 'timestamp' column in EEG file: {eeg_file}")
                continue

            # Load marker data and validate required columns
            markers = pd.read_csv(marker_path)
            if 'timestamp' not in markers.columns or 'marker' not in markers.columns:
                logging.error(f"Missing required columns in marker file: {marker_file}")
                continue

            # Align markers to EEG timestamps within a tolerance window
            eeg_timestamps = eeg_data['timestamp'].values
            aligned_markers = [
                [eeg_timestamps[np.argmin(np.abs(eeg_timestamps - marker[0]))], marker[1]]
                for marker in markers[['timestamp', 'marker']].values
                if np.abs(eeg_timestamps[np.argmin(np.abs(eeg_timestamps - marker[0]))] - marker[0]) <= 0.1
            ]

            # Normalize the paradigm name from the EEG file name
            raw_paradigm_name = '_'.join(eeg_file.split('_')[2:-1])
            paradigm_name = normalize_paradigm_name(raw_paradigm_name)

            # Store EEG and aligned markers in the dictionary
            data[paradigm_name] = {'eeg': eeg_data, 'markers': np.array(aligned_markers)}

        except Exception as e:
            logging.error(f"Error loading files for {eeg_file}: {e}")
            continue

    return data

def load_eeg_data(dataset_dir):
    """
    Load data for all subjects in the dataset directory.

    Args:
        dataset_dir (str): Path to the dataset directory containing subject folders.

    Returns:
        dict: A nested dictionary containing data for all subjects.
    """
    all_data = {}
    subject_folders = [os.path.join(dataset_dir, subj) for subj in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, subj))]

    for subject_folder in subject_folders:
        subject_name = os.path.basename(subject_folder)
        logging.info(f"Loading data for {subject_name}...")

        subject_data = load_subject_data(subject_folder)
        if subject_data:
            all_data[subject_name] = subject_data
        else:
            logging.warning(f"No valid data found for {subject_name}, skipping...")

    return all_data
