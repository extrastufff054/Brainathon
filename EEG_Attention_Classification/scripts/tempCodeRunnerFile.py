import pandas as pd
import numpy as np

# Load the EEG data and markers CSV files
eeg_df = pd.read_csv('C:/Users/Sarang/Downloads/Brainathon/EEG_Attention_Classification/data/Subject_3/Subject_3_baseline_eyesclosed_markers.csv')
markers_df = pd.read_csv('C:/Users/Sarang/Downloads/Brainathon/EEG_Attention_Classification/data/Subject_3/Subject_3_baseline_eyesclosed_markers.csv')

# Ensure the timestamps are in float format
eeg_df['timestamp'] = eeg_df['timestamp'].astype(float)
markers_df['timestamp'] = markers_df['timestamp'].astype(float)

# Create an empty list to store matched markers
matched_markers = []

# Define the tolerance window (e.g., 0.1 seconds)
tolerance_window = 0.1

# Iterate through the markers and find the closest timestamp in EEG data
for _, marker_row in markers_df.iterrows():
    marker_time = marker_row['timestamp']
    
    # Find the absolute difference between marker timestamp and all EEG timestamps
    time_diff = np.abs(eeg_df['timestamp'] - marker_time)
    
    # Find the index of the EEG timestamp with the minimum difference
    closest_idx = time_diff.idxmin()
    
    # Check if the closest timestamp is within the tolerance window
    if time_diff[closest_idx] <= tolerance_window:
        matched_marker = {
            'marker': marker_row['marker'],
            'marker_timestamp': marker_time,
            'eeg_timestamp': eeg_df.loc[closest_idx, 'timestamp'],
            'eeg_idx': closest_idx
        }
        matched_markers.append(matched_marker)

# Convert matched markers to a DataFrame
matched_markers_df = pd.DataFrame(matched_markers)

# Print the matched markers with corresponding EEG timestamps
print(matched_markers_df)

# Optionally, save the matched markers to a CSV file
matched_markers_df.to_csv('matched_markers.csv', index=False)
