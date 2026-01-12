from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

def get_breast_cancer_dataset(test_size= 0.3, random_state= 42):
    # load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size= test_size, random_state= random_state)
    
    # standardize
    mean = np.mean(X_train, axis= 0)
    std = np.std(X_train, axis= 0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test