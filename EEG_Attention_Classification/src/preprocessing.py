import logging
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to EEG data.

    Args:
        data (ndarray): EEG signal data (channels x samples).
        lowcut (float): Lower bound of the frequency range.
        highcut (float): Upper bound of the frequency range.
        fs (int): Sampling frequency of the EEG data.
        order (int): Order of the filter.

    Returns:
        ndarray: Filtered EEG data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def segment_epochs(eeg_data, markers, fs, epoch_length=1):
    """
    Segment EEG data into epochs based on event markers.

    Args:
        eeg_data (ndarray): EEG data (channels x samples).
        markers (ndarray): Event markers with columns [timestamp, event].
        fs (int): Sampling frequency of the EEG data.
        epoch_length (int): Length of each epoch in seconds.

    Returns:
        tuple: (epochs, labels)
            - epochs: ndarray of shape (num_epochs, channels, samples_per_epoch).
            - labels: List of corresponding event labels.
    """
    epoch_samples = int(epoch_length * fs)
    epochs = []
    labels = []

    # Get the max timestamp based on the length of the EEG data (in samples)
    max_samples = eeg_data.shape[1]  # Number of samples in EEG data

    # Ensure markers is a 2D ndarray with two columns [timestamp, event]
    if markers.ndim != 2 or markers.shape[1] != 2:
        raise ValueError("Markers should be a 2D ndarray with two columns: [timestamp, event]")
    
    print(f"Total markers: {markers.shape[0]}")  # Debugging: print marker count

    for marker in markers:
        timestamp = int(marker[0])  # Timestamp (in seconds)
        event = marker[1]           # Event type

        # Calculate start and end indices for the epoch
        start_idx = int(timestamp * fs)  # Convert timestamp to sample index
        end_idx = start_idx + epoch_samples

        # Ensure the epoch doesn't exceed the data length
        if start_idx < max_samples and end_idx <= max_samples:
            epochs.append(eeg_data[:, start_idx:end_idx])  # Extract the epoch data
            labels.append(event)
        else:
            # Skip invalid markers where the epoch exceeds the data length
            print(f"Skipping marker at timestamp {timestamp} (epoch out of bounds).")
    
    print(f"Total epochs created: {len(epochs)}")  # Debugging: print number of epochs
    return np.array(epochs), np.array(labels)

def preprocess_eeg_data(eeg_data, labels, fs):
    """
    Preprocess the EEG data: Extract epochs based on the provided markers.
    """
    # Assuming you have defined epoch length and overlap or other parameters
    epoch_length = 1.0  # in seconds
    overlap = 0.5  # in seconds

    # Calculate number of samples per epoch
    epoch_samples = int(epoch_length * fs)
    overlap_samples = int(overlap * fs)

    # Initialize lists to store valid epochs
    epochs = []
    valid_labels = []

    logging.info(f"Total markers: {len(labels)}")

    for marker in labels:
        # Extract the timestamp of the marker (in your case it might be in seconds or samples)
        marker_time = marker[0]  # Assuming marker format is [timestamp, label]
        
        # Define the start and end time for the epoch
        epoch_start = int(marker_time * fs) - int(epoch_samples / 2)
        epoch_end = epoch_start + epoch_samples
        
        # Check if the epoch is within the bounds of the EEG data
        if epoch_start >= 0 and epoch_end <= eeg_data.shape[0]:
            # Extract the epoch data
            epoch_data = eeg_data[epoch_start:epoch_end, :]
            epochs.append(epoch_data)
            valid_labels.append(marker[1])  # Assuming marker[1] is the label
        else:
            logging.warning(f"Skipping marker at timestamp {marker_time} (epoch out of bounds).")

    logging.info(f"Total epochs created: {len(epochs)}")
    
    # Convert lists to numpy arrays for further processing
    if len(epochs) == 0:
        logging.error("No valid epochs were created.")
        return np.array([]), np.array([])  # Return empty arrays in case of failure

    epochs = np.array(epochs)
    valid_labels = np.array(valid_labels)

    return epochs, valid_labels
