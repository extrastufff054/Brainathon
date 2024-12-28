import numpy as np
import scipy.stats as stats
from scipy.signal import welch

def extract_time_domain_features(epoch):
    """
    Extract time-domain features from an EEG epoch.

    Args:
        epoch (ndarray): EEG epoch (channels x samples).

    Returns:
        ndarray: Extracted features.
    """
    features = []
    for channel in epoch:
        features.append(np.mean(channel))           # Mean
        features.append(np.std(channel))            # Standard Deviation
        features.append(stats.skew(channel))        # Skewness
        features.append(stats.kurtosis(channel))    # Kurtosis
    return np.array(features)

def extract_frequency_domain_features(epoch, fs):
    """
    Extract frequency-domain features using power spectral density (PSD).

    Args:
        epoch (ndarray): EEG epoch (channels x samples).
        fs (int): Sampling frequency.

    Returns:
        ndarray: Extracted features.
    """
    features = []
    for channel in epoch:
        freqs, psd = welch(channel, fs=fs, nperseg=fs)

        # Frequency bands
        delta_band = (0.5, 4)
        theta_band = (4, 8)
        alpha_band = (8, 13)
        beta_band = (13, 30)

        # Band-specific power
        features.append(np.sum(psd[(freqs >= delta_band[0]) & (freqs < delta_band[1])]))  # Delta power
        features.append(np.sum(psd[(freqs >= theta_band[0]) & (freqs < theta_band[1])]))  # Theta power
        features.append(np.sum(psd[(freqs >= alpha_band[0]) & (freqs < alpha_band[1])]))  # Alpha power
        features.append(np.sum(psd[(freqs >= beta_band[0]) & (freqs < beta_band[1])]))    # Beta power

    return np.array(features)

def extract_features(epochs, fs):
    """
    Extract combined time and frequency domain features from all epochs.

    Args:
        epochs (ndarray): EEG epochs (num_epochs x channels x samples).
        fs (int): Sampling frequency.

    Returns:
        ndarray: Feature matrix (num_epochs x num_features).
    """
    feature_matrix = []

    for epoch in epochs:
        time_features = extract_time_domain_features(epoch)
        freq_features = extract_frequency_domain_features(epoch, fs)
        combined_features = np.concatenate((time_features, freq_features))
        feature_matrix.append(combined_features)

    return np.array(feature_matrix)

if __name__ == "__main__":
    # Simple test to see if the function is accessible
    print(extract_features(np.random.randn(10, 64, 250), 250))  # Test with dummy data
