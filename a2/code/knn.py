"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        n, d = X_hat.shape

        s_dist_mat = euclidean_dist_squared(X_hat, self.X)
        
        result = np.zeros(n)
        for test_index in range(n):
            distances = s_dist_mat[test_index, :]
            sorted_args = np.argsort(distances, axis=0)
            neighbours_classes = self.y[sorted_args[:self.k]]
            result[test_index] = utils.mode(neighbours_classes)
            
        return result