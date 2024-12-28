import os
import pandas as pd

# Define the base directory (WSL compatible path)
base_dir = "/mnt/c/Users/Sarang/Downloads/Brainathon/EEG_Attention_Classification/data"

# Check if the base directory exists
if not os.path.exists(base_dir):
    print(f"Error: The directory {base_dir} does not exist.")
else:
    # Iterate through each subject folder in the base directory
    for subject_folder in os.listdir(base_dir):
        subject_folder_path = os.path.join(base_dir, subject_folder)

        if os.path.isdir(subject_folder_path):  # Ensure it's a folder
            print(f"Processing subject: {subject_folder}")

            # List of the file suffixes
            file_suffixes = [
                "baseline_eyesclosed_eeg.csv",
                "baseline_eyesclosed_markers.csv",
                "baseline_eyesopen_eeg.csv",
                "baseline_eyesopen_markers.csv",
                "dual-task_paradigm_eeg.csv",
                "dual-task_paradigm_markers.csv",
                "oddball_paradigm_eeg.csv",
                "oddball_paradigm_markers.csv",
                "stroop_task_eeg.csv",
                "stroop_task_markers.csv",
                "task-switching_paradigm_eeg.csv",
                "task-switching_paradigm_markers.csv",
            ]

            # Process each file suffix for the current subject
            for suffix in file_suffixes:
                # Dynamically create the correct file path using the current subject's folder name
                file_path = os.path.join(subject_folder_path, f"{subject_folder}_{suffix}")

                if os.path.exists(file_path):
                    print(f"Found file: {file_path}")
                    # Read the EEG data (assuming CSV format)
                    if 'eeg' in suffix:
                        eeg_data = pd.read_csv(file_path)
                        # Process the EEG data (you can modify this as per your needs)
                        print(f"Processed EEG data from: {file_path}")
                    elif 'markers' in suffix:
                        markers_data = pd.read_csv(file_path)
                        # Process the markers data (you can modify this as per your needs)
                        print(f"Processed markers data from: {file_path}")
                else:
                    print(f"File not found: {file_path}")
