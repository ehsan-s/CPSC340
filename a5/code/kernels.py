import numpy as np
from utils import euclidean_dist_squared


class Kernel:
    def __call__(self, X1, X2):
        """
        Evaluate Gram's kernel matrix based on the two input matrices.
        Shape of X1 is (n1, d) and shape of X2 is (n2, d).
        That is, both matrices should have the same number of columns.
        Will return a n1-by-n2 matrix, e.g. X1 @ X2.T
        """
        # This is the "magic method" that happens when you call kernel(X1, X2)
        raise NotImplementedError("this is a base class, don't call this")


class LinearKernel(Kernel):
    def __call__(self, X1, X2):
        return X1 @ X2.T


class PolynomialKernel(Kernel):
    def __init__(self, p):
        """
        p is the degree of the polynomial
        """
        self.p = p

    def __call__(self, X1, X2):
        """
        Evaluate the polynomial kernel.
        A naive implementation will use change of basis.
        A "kernel trick" implementation bypasses change of basis.
        """

        """YOUR CODE HERE FOR Q1.1"""
        n1, d = X1.shape
        n2, d = X2.shape
        K = np.zeros((n1, n2))
        for i in range(n1):
            X_i = X1[i]
            for j in range(n2):
                X_j = X2[j]
                K[i, j] = (1 + X_i @ X_j.T) ** self.p

        return K


class GaussianRBFKernel(Kernel):
    def __init__(self, sigma):
        """
        sigma is the curve width hyperparameter.
        """
        self.sigma = sigma

    def __call__(self, X1, X2):
        """
        Evaluate Gaussian RBF basis kernel.
        """

        """YOUR CODE HERE FOR Q1.1"""
        n1, d = X1.shape
        n2, d = X2.shape
        K = np.zeros((n1, n2))
        for i in range(n1):
            X_i = X1[i]
            for j in range(n2):
                X_j = X2[j]
                K[i, j] = np.exp(- (np.linalg.norm(X_i - X_j) ** 2) / (2 * (self.sigma**2) ) )

        return K
