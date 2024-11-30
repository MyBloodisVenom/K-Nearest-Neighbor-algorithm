import numpy as np
from collections import Counter

def knn_classifier(X_train, y_train, X_test, k):
    """
    A simple KNN classifier.

    Parameters:
    - X_train: numpy array or list of training data.
    - y_train: numpy array or list of training names.
    - X_test: numpy array or list of test data.
    - k: Number of nearest neighbors to analyze.

    Returns:
    - predictions: A list of predicted labels for the test set.
    """
    # First we initialize an empty array
    predictions = []

    # Now we iterate through each test point
    for test_point in X_test:

        # First we find the distance from the test point to the training points
        distances = np.linalg.norm(X_train - test_point, axis=1)

        # We find the indices of the smallest k distances
        k_indices = distances.argsort()[:k]

        # We find the labels of the k nearest neighbors
        k_neighbors = y_train[k_indices]

        # we use majority voting to get the most common label
        most_common_label = Counter(k_neighbors).most_common(1)[0][0]

        # We append our prediction
        predictions.append(most_common_label)

    return predictions


