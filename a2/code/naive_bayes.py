import numpy as np


class NaiveBayes:
    """
    Naive Bayes implementation.
    Assumes the feature are binary.
    Also assumes the labels go from 0,1,...k-1
    """

    p_y = None
    p_xy = None
    not_p_xy = None

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n

        """YOUR CODE HERE FOR Q3.3"""
        p_xy = np.zeros((d, k))
        count_xy = np.zeros((d, k))
        for j in range(d):
            for i in range(n):
                if X[i, j]:
                    count_xy[j, y[i]] += 1
            for l in range(k):
                p_xy[j, l] = count_xy[j, l]/counts[l]
        self.p_y = p_y
        self.p_xy = p_xy
        self.not_p_xy = 1 - p_xy

    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        not_p_xy = self.not_p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):

            probs = p_y.copy()  # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= not_p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred


class NaiveBayesLaplace(NaiveBayes):
    def __init__(self, num_classes, beta=0):
        super().__init__(num_classes)
        self.beta = beta

    def fit(self, X, y):
        """YOUR CODE FOR Q3.4"""
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n

        p_xy = np.zeros((d, k))
        count_xy = np.zeros((d, k))
        not_p_xy = np.zeros((d, k))
        not_count_xy = np.zeros((d, k))

        for j in range(d):
            for i in range(n):
                if X[i, j]:
                    count_xy[j, y[i]] += 1
                else:
                    not_count_xy[j, y[i]] += 1
            for l in range(k):
                p_xy[j, l] = (count_xy[j, l] + self.beta) / (counts[l] + self.beta * 2)
                not_p_xy[j, l] = (not_count_xy[j, l] + self.beta) / (counts[l] + self.beta * 2)

        self.p_y = p_y
        self.p_xy = p_xy
        self.not_p_xy = not_p_xy
