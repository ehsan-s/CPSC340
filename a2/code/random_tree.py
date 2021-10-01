from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
from scipy import stats
import numpy as np
import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpInfoGain)

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    num_trees = None
    max_depth = None
    random_trees = None

    """
    YOUR CODE HERE FOR Q4
    Hint: start with the constructor __init__(), which takes the hyperparameters.
    Hint: you can instantiate objects inside fit().
    Make sure predict() is able to handle multiple examples.
    """
    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth

    def fit(self, X, y):
        random_trees = np.empty(self.num_trees, dtype=object)
        for i in range(self.num_trees):
            random_tree = RandomTree(self.max_depth)
            random_tree.fit(X, y)
            random_trees[i] = random_tree

        self.random_trees = random_trees

    def predict(self, X):
        n, d = X.shape
        y_trees = np.zeros((self.num_trees, n))
        y = np.zeros(n)
        for i in range(self.num_trees):
            y_trees[i, :] = self.random_trees[i].predict(X)

        for i in range(n):
            y[i] = stats.mode(y_trees[:, i].flatten())[0][0]

        return y
