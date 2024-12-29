import os
import pandas as pd
import numpy as np
import logging

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

            # Load marker data and validate required columns
            markers = pd.read_csv(marker_path)
            if 'timestamp' not in markers.columns or 'marker' not in markers.columns:
                logging.error(f"Missing required columns in marker file: {marker_file}")
                continue

            # Rename 'marker' column to 'event' (optional)
            markers.rename(columns={'marker': 'event'}, inplace=True)

            # Extract relevant columns
            markers = markers[['timestamp', 'event']].values

            # Convert timestamps to match the EEG timestamps (assuming they are in seconds)
            eeg_timestamps = eeg_data['timestamp'].values  # Assuming EEG data has a 'timestamp' column
            aligned_markers = []

            for marker in markers:
                marker_time = marker[0]
                # Find the closest EEG timestamp within a tolerance window (e.g., 0.1 seconds)
                closest_idx = np.argmin(np.abs(eeg_timestamps - marker_time))
                closest_time = eeg_timestamps[closest_idx]

                if np.abs(closest_time - marker_time) <= 0.1:  # Tolerance window of 0.1 seconds
                    aligned_markers.append([closest_time, marker[1]])

            # Store EEG and marker data in the dictionary
            raw_paradigm_name = '_'.join(eeg_file.split('_')[2:-1])
            paradigm_name = normalize_paradigm_name(raw_paradigm_name)

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
