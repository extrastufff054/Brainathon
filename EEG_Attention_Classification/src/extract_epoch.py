import pandas as pd

def extract_epoch(data, timestamp, fs, epoch_duration=1):
    """
    Extract an epoch of EEG data around a given timestamp.
    
    Parameters:
    - data: DataFrame containing the EEG data with timestamps.
    - timestamp: The center timestamp for the epoch.
    - fs: Sampling frequency.
    - epoch_duration: Duration of the epoch in seconds (default is 1 second).
    
    Returns:
    - epoch: The extracted epoch as a DataFrame.
    """
    half_epoch_samples = int(epoch_duration * fs / 2)
    start_time = timestamp - half_epoch_samples / fs
    end_time = timestamp + half_epoch_samples / fs
    
    epoch = data[(data.iloc[:, 0] >= start_time) & (data.iloc[:, 0] <= end_time)]
    return epoch