import logging
import numpy as np
from scipy.signal import butter, filtfilt

logging.basicConfig(level=logging.INFO)

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
    return filtfilt(b, a, data, axis=1)  # Ensure filtering is applied along the sample axis

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
    max_samples = eeg_data.shape[1]
    epochs = []
    labels = []

    for marker_time, event in markers:
        start_idx = int(marker_time * fs)
        end_idx = start_idx + epoch_samples

        if start_idx >= 0 and end_idx <= max_samples:
            epochs.append(eeg_data[:, start_idx:end_idx])
            labels.append(event)
        else:
            logging.warning(f"Skipping marker at {marker_time} (epoch out of bounds).")

    return np.array(epochs), np.array(labels)

def preprocess_eeg_data(eeg_data, labels, fs, epoch_length=1.0, overlap=0.5):
    """
    Preprocess the EEG data: Extract epochs based on the provided markers.
    
    Args:
        eeg_data (np.ndarray): The raw EEG data.
        labels (np.ndarray): The markers corresponding to the epochs.
        fs (int): The sampling frequency.
        epoch_length (float): The length of each epoch in seconds.
        overlap (float): The overlap between epochs in seconds.

    Returns:
        epochs (np.ndarray): Extracted epochs.
        valid_labels (np.ndarray): Corresponding labels for the epochs.
    """
    # Calculate number of samples per epoch
    epoch_samples = int(epoch_length * fs)
    overlap_samples = int(overlap * fs)

    # Initialize lists to store valid epochs and labels
    epochs = []
    valid_labels = []

    logging.info(f"Total markers: {len(labels)}")

    # Ensure markers is a 2D ndarray with two columns [timestamp, label]
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("Markers should be a 2D ndarray with two columns: [timestamp, label]")

    for marker in labels:
        try:
            # Extract the timestamp of the marker (assuming it is in seconds)
            marker_time = float(marker[0])  # Ensure the timestamp is a float
            marker_label = marker[1]       # Event label

            # Convert timestamp to sample index
            marker_time_samples = int(marker_time * fs)

            # Define the start and end time for the epoch
            epoch_start = marker_time_samples - int(epoch_samples / 2)
            epoch_end = epoch_start + epoch_samples

            # Check if the epoch is within the bounds of the EEG data
            if epoch_start >= 0 and epoch_end <= eeg_data.shape[1]:  # Check against number of samples (not channels)
                # Extract the epoch data
                epoch_data = eeg_data[:, epoch_start:epoch_end]
                epochs.append(epoch_data)
                valid_labels.append(marker_label)
                logging.debug(f"Epoch created for marker at {marker_time}s")
            else:
                # Skip invalid markers where the epoch exceeds the data length
                logging.warning(f"Skipping marker at timestamp {marker_time} (epoch out of bounds).")
        except ValueError as e:
            logging.error(f"Invalid marker data encountered: {marker}. Error: {e}")
            continue

    logging.info(f"Total epochs created: {len(epochs)}")

    # Convert lists to numpy arrays for further processing
    if len(epochs) == 0:
        logging.error("No valid epochs were created. Check if markers are within the bounds of the EEG data.")
        return np.array([]), np.array([])  # Return empty arrays in case of failure

    epochs = np.array(epochs)
    valid_labels = np.array(valid_labels)

    # Check if the epochs array is empty
    if epochs.shape[0] == 0:
        logging.error("No epochs created. Please check the parameters or data.")
        return np.array([]), np.array([])

    logging.info(f"Epoch data shape: {epochs.shape}")
    logging.info(f"Labels shape: {valid_labels.shape}")

    return epochs, valid_labels

