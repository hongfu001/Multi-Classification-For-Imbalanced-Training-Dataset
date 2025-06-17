import numpy as np
import pandas as pd

from collections import Counter
from common import setup_seed

def read_dataset(train_test_ratio, seed):
    setup_seed(seed)
    # Read the CSV file
    dat = pd.read_csv('NYC0606.csv')
    # Drop rows with any NaN values
    dat = dat.dropna(axis=0, how='any')

    # Extract features and target
    X = np.array(dat.iloc[:, :-1])  # All columns except the last one as features
    y = np.array(dat.iloc[:, -1])   # Last column as target

    # Step 1: Determine the size of the test set
    n_classes = len(np.unique(y))  # Number of classes (should be 4)
    min_class_count = min(Counter(y).values())  # Smallest class size
    test_size_per_class = int(min_class_count * train_test_ratio)  # Example: take half of smallest class size per class

    # Step 2: Create a balanced test set
    X_test = []
    y_test = []
    X_train = []
    y_train = []

    for class_label in np.unique(y):
        # Get indices for the current class
        class_indices = np.where(y == class_label)[0]
        # Randomly sample test_size_per_class indices for test set
        test_indices = np.random.choice(class_indices, size=test_size_per_class, replace=False)
        # Remaining indices go to training set
        train_indices = np.setdiff1d(class_indices, test_indices)
        
        # Append to test and train sets
        X_test.append(X[test_indices])
        y_test.append(y[test_indices])
        X_train.append(X[train_indices])
        y_train.append(y[train_indices])

    # Concatenate the arrays
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Shuffle the test and train sets
    test_perm = np.random.permutation(len(y_test))
    X_test = X_test[test_perm]
    y_test = y_test[test_perm]

    train_perm = np.random.permutation(len(y_train))
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    return X_train, y_train, X_test, y_test