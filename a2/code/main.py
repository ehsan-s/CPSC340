#!/usr/bin/env python
import argparse
from functools import partial
import os
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
import utils
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree

import math


def load_dataset(filename):
    with open(Path("..", "data", filename), "rb") as f:
        return pickle.load(f)


# this just some Python scaffolding to conveniently run the functions below;
# don't worry about figuring out how it works if it's not obvious to you
func_registry = {}


def handle(number):
    def register(func):
        func_registry[number] = func
        return func

    return register


def run(question):
    if question not in func_registry:
        raise ValueError(f"unknown question {question}")
    return func_registry[question]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", required=True, choices=func_registry.keys())
    args = parser.parse_args()
    print(args)
    return run(args.question)


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    print('KNN Classifier')
    for k in [1, 3, 10]:
        KNN_model = KNN(k)
        KNN_model.fit(X, y)

        y_predicted = KNN_model.predict(X)
        train_error = np.mean(y_predicted != y)

        y_predicted = KNN_model.predict(X_test)
        test_error = np.mean(y_predicted != y_test)
        
        print('error for k =', k , ': train error', train_error, 'test error', test_error)

        # section 1.3 ___START
        if(k==1):
            utils.plot_classifier(KNN_model, X, y)
            fname = os.path.join("..", "figs", "q1_3_myKNNClassifier.pdf")
            plt.savefig(fname)
            print("Figure saved as '%s'" % fname)

            sklearn_KNN = KNeighborsClassifier(n_neighbors = 1)
            sklearn_KNN.fit(X, y)
            utils.plot_classifier(sklearn_KNN, X, y)
            fname = os.path.join("..", "figs", "q1_3_sklearnKNNClassifier.pdf")
            plt.savefig(fname)
            print("Figure saved as '%s'" % fname)
        # section 1.3 ___END

    print('Decision Tree Classifier')
    decision_tree_model = DecisionTree(3)
    decision_tree_model.fit(X, y)

    y_predicted = decision_tree_model.predict(X)
    train_error = np.mean(y_predicted != y)
    
    y_predicted = decision_tree_model.predict(X_test)
    test_error = np.mean(y_predicted != y_test)

    print('error for decision tree: train error', train_error, 'test error', test_error)


@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    print()

    ks = list(range(1, 30, 4))
    
    cv_accs = []
    n, d = X.shape
    n_fold = 10
    fold_size = math.ceil(n/n_fold)

    for k in ks:
        KNN_model = KNN(k)

        val_errors = []
        for i in range(n_fold):
            mask = np.ones(n, dtype=bool)
            mask [i * fold_size : min(n, (i+1) * fold_size)] = False
            X_train, y_train = X[mask], y[mask]
            X_val, y_val = X[~mask], y[~mask]

            KNN_model.fit(X_train, y_train)
            y_predicted = KNN_model.predict(X_val)
            val_error = np.mean(y_predicted != y_val)
            val_errors += [val_error]

        cv_accs += [np.mean(val_errors)]

    print('ks: ', ks)
    print('cv_accs: ', cv_accs)

    # section 2.2 ___START
    train_errors = []
    test_errors = []
    for k in ks:
        KNN_model = KNN(k)
        KNN_model.fit(X, y)

        y_predicted = KNN_model.predict(X)
        train_error = np.mean(y_predicted != y)
        train_errors += [train_error]

        y_predicted = KNN_model.predict(X_test)
        test_error = np.mean(y_predicted != y_test)
        test_errors += [test_error]

    print('test_errors: ', test_errors)
    print('train_errors: ', train_errors)

    plt.figure() 
    plt.plot(ks, cv_accs, label = 'Cross Validation')
    plt.plot(ks, test_errors, label = 'Test')
    plt.xlabel('k')
    plt.xticks(ks)
    plt.ylabel('Error')
    plt.legend()
    fname = os.path.join("..", "figs", "q2_2_valtestErrorsForDifferentK.pdf")
    plt.savefig(fname)
    print("Figure saved as '%s'" % fname)

    test_accuracies = [1-e for e in test_errors]
    cv_accuraries = [1-e for e in cv_accs]
    train_accuracies = [1-e for e in train_errors]
    print('test_accuracies: ', test_errors)
    print('train_accuracies: ', train_errors)
    plt.figure() 
    plt.plot(ks, cv_accuraries, label = 'Cross Validation')
    plt.plot(ks, test_accuracies, label = 'Test')
    plt.xlabel('k')
    plt.xticks(ks)
    plt.ylabel('Accuracy')
    plt.legend()
    fname = os.path.join("..", "figs", "q2_2_valtestAccuraciesForDifferentK.pdf")
    plt.savefig(fname)
    print("Figure saved as '%s'" % fname)
    # section 2.2 ___END

    # section 2.4 ___START
    plt.figure() 
    plt.plot(ks, train_errors, label = 'Train')
    plt.xlabel('k')
    plt.xticks(ks)
    plt.ylabel('Error')
    plt.legend()
    fname = os.path.join("..", "figs", "q2_4_trainErrorsForDifferentK.pdf")
    plt.savefig(fname)
    print("Figure saved as '%s'" % fname)

    plt.figure() 
    plt.plot(ks, train_accuracies, label = 'Train')
    plt.xlabel('k')
    plt.xticks(ks)
    plt.ylabel('Accuracy')
    plt.legend()
    fname = os.path.join("..", "figs", "q2_4_trainAccuraciesForDifferentK.pdf")
    plt.savefig(fname)
    print("Figure saved as '%s'" % fname)
    # section 2.4 ___END


@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    """YOUR CODE HERE FOR Q3.2"""
    # print(X)
    print(X.shape)

    print(wordlist[72], X[0,72])

    for i in range(X.shape[1]):
        if X[802, i]:
            print(i, wordlist[i])

    print(y[802])
    for i in range(4):
        print(groupnames[i])
    print(groupnames[y[802]])



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayesLaplace(num_classes=4, beta=1)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4"""
    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes Laplace training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes Laplace validation error: {err_valid:.3f}")


@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain:")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    """YOUR CODE HERE FOR Q4"""
    print("Random tree:")
    evaluate_model(RandomTree(max_depth=np.inf))

    print("Random forest:")
    evaluate_model(RandomForest(num_trees=50, max_depth=np.inf))


@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]
    
    best_kmeansModel = None
    best_kmeansModelError = np.inf

    for i in range(50):
        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        error = model.error(X, y, model.means)
        if(error < best_kmeansModelError):
            best_kmeansModelError = error
            best_kmeansModel = model

    y = best_kmeansModel.predict(X)
    print('finalKmeansClassifier error:', best_kmeansModelError)
    utils.plot_4class_classifier(best_kmeansModel, X, y)
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")
    fname = Path("..", "figs",  "q5_1_finalKmeansClassifier.pdf")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    minError_k = []

    for k in range(1,11):
        best_kmeansModelError = np.inf
        for i in range(50):
            model = Kmeans(k=k)
            model.fit(X)
            y = model.predict(X)
            error = model.error(X, y, model.means)
            if(error < best_kmeansModelError):
                best_kmeansModelError = error
        minError_k += [best_kmeansModelError]

    plt.figure() 
    plt.plot(range(1,11), minError_k, label = 'Min Error')
    plt.xlabel('k')
    plt.ylabel('Error')
    fname = os.path.join("..", "figs", "q5_2_trainAccuraciesForDifferentK.pdf")
    plt.savefig(fname)
    print("Figure saved as '%s'" % fname)



if __name__ == "__main__":
    main()
