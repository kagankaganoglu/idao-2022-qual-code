"""metrics for evaluating datasets"""
import numpy as np


def mae_eski(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simple mean absolute error calculations

    Args:
        y_true: (numpy array) ground truth
        y_pred: (numpy array) predicted values
    Returns:
         (float) mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred)).item()

def mae(prediction, target):
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.
    print('HELLLOOO MAE KODU CAGIRILDI AMA BIZ ICERI SIZDIK')
    e_thresh = 0.02
    error_energy = np.abs(target - prediction)

    success = np.count_nonzero(error_energy < e_thresh)
    total = np.size(target)
    return success / np.int64(total)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simple accuracy calculation

    Args:
        y_true: numpy array of 0 and 1's
        y_pred: numpy array of predict sigmoid
    Returns:
        (float) accuracy
    """
    y_pred = y_pred > 0.5
    return np.sum(y_true == y_pred) / len(y_pred)
