import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        mod = 1e-7
        pre = -(1/len(y_true))
        comp = y_true * np.log(y_pred + mod) + (1-y_true) * np.log(1 - y_pred + mod)
        return np.round(pre * np.sum(comp), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        mod = 1e-7
        pre = -(1/y_true.shape[0])

        y_clip = np.clip(y_pred, mod, 1 - mod)
        ans = pre * np.sum(y_clip) * np.sum(y_true) * np.log(y_clip)
        return np.round(pre * np.sum(y_true * np.log(y_clip)), 4)

        
