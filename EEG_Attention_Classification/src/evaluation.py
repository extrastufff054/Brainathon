import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay,
)

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model's performance using classification metrics.

    Args:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.

    Returns:
        dict: Dictionary containing accuracy and classification report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

    return {
        "accuracy": accuracy,
        "classification_report": report
    }

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize='true'):
    """
    Plot a confusion matrix to visualize model performance.

    Args:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.
        class_names (list): List of class names. Defaults to numerical labels.
        normalize (str): Normalization mode for confusion matrix ('true', 'pred', 'all', None).

    Returns:
        None: Displays the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()
