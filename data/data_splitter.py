import pandas as pd
from ucimlrepo import fetch_ucirepo

def split_and_write(test_file_prefix="test", train_file_prefix="train"):
    """
    Splits the Iris dataset into training and testing sets, and saves the split data to CSV files.

    Parameters:
    - test_file_prefix (str): Prefix for the test data files (e.g., "test").
    - train_file_prefix (str): Prefix for the train data files (e.g., "train").
    """
    # Fetch dataset
    iris = fetch_ucirepo(id=53)

    # Data (as pandas dataframes)
    X = iris.data.features
    y = iris.data.targets

    # Concatenate the data for grouping
    data = pd.concat([X, y], axis=1)
    grouped = data.groupby('class')

    # Initialize empty dataframes for train and test
    test = pd.DataFrame()
    train = pd.DataFrame()

    # Split the data into groups of 75 and 75(specific to our data set)
    for _, group in grouped:
        test = pd.concat([test, group.iloc[:25]])
        train = pd.concat([train, group.iloc[25:]])

    # Separate features (X) and labels (y) for both train and test
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    # Save the split data to CSV files
    X_test.to_csv(f'{test_file_prefix}_X.csv', index=False)
    y_test.to_csv(f'{test_file_prefix}_y.csv', index=False)
    X_train.to_csv(f'{train_file_prefix}_X.csv', index=False)
    # Technically the y_train isn't needed as the data is split by 50, but we'll make it anyways
    y_train.to_csv(f'{train_file_prefix}_y.csv', index=False)

    print("Data splitting complete and saved to files!")
