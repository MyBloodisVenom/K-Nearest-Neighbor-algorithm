import pandas as pd
import numpy as np
from data.data_splitter import split_and_write
from knn.knn import knn_classifier

def main():
    # Split the data (if not already split)
    print("Splitting the data...")
    split_and_write(test_file_prefix="data/test", train_file_prefix="data/train")

    # Load the split data from CSV files
    print("Loading split data...")
    X_train = pd.read_csv("data/train_X.csv").to_numpy()
    y_train = pd.read_csv("data/train_y.csv").squeeze().to_numpy()  # Squeeze to convert to 1D array
    X_test = pd.read_csv("data/test_X.csv").to_numpy()
    y_test = pd.read_csv("data/test_y.csv").squeeze().to_numpy()

    # Test KNN algorithm for different values of k
    k_values = [1, 3, 5]
    for k in k_values:
        print(f"\nRunning KNN with k={k}...")
        predictions = knn_classifier(X_train, y_train, X_test, k)

        # Calculate accuracy
        accuracy = np.mean(predictions == y_test) * 100
        print(f"Accuracy for k={k}: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

