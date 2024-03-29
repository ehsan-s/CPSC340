#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from encoders import PCAEncoder
from kernels import GaussianRBFKernel, LinearKernel, PolynomialKernel
from linear_models import (
    LinearModel,
    LinearClassifier,
    KernelClassifier,
)
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
from fun_obj import (
    LeastSquaresLoss,
    LeastSquaresLossL2,
    LogisticRegressionLossL2,
    KernelLogisticRegressionLossL2,
)
from learning_rate_getters import (
    ConstantLR,
    InverseLR,
    InverseSqrtLR,
    InverseSquaredLR,
)
import utils
from utils import load_dataset, load_trainval, load_and_split


# this just some Python scaffolding to conveniently run the functions below;
# don't worry about figuring out how it works if it's not obvious to you
_funcs = {}


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--question", required=True, choices=sorted(_funcs.keys()) + ["all"]
    )
    args = parser.parse_args()
    if args.question == "all":
        for q in sorted(_funcs.keys()):
            start = f"== {q} "
            print("\n" + start + "=" * (80 - len(start)))
            run(q)
    else:
        return run(args.question)


@handle("1")
def q1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # Standard (regularized) logistic regression
    loss_fn = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    lr_model = LinearClassifier(loss_fn, optimizer)
    lr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(lr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(lr_model.predict(X_val) != y_val):.1%}")

    fig = utils.plot_classifier(lr_model, X_train, y_train)
    utils.savefig("logRegPlain.png", fig)

    # kernel logistic regression with a linear kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel = LinearKernel()
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("logRegLinear.png", fig)


@handle("1.1")
def q1_1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    """YOUR CODE HERE FOR Q1.1"""
    # kernel logistic regression with a polynomial kernel
    loss_fn = KernelLogisticRegressionLossL2(0.01)
    optimizer = GradientDescentLineSearch()
    poly_kernel = PolynomialKernel(2)
    klr_model = KernelClassifier(loss_fn, optimizer, poly_kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("logRegPolynomialKernel.png", fig)

    # kernel logistic regression with a Gaussian RBF kernel
    loss_fn = KernelLogisticRegressionLossL2(0.01)
    optimizer = GradientDescentLineSearch()
    RBF_kernel = GaussianRBFKernel(0.5)
    klr_model = KernelClassifier(loss_fn, optimizer, RBF_kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("logRegGaussianRBFKernel.png", fig)


@handle("1.2")
def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])

    # train_errs[i, j] should be the train error for sigmas[i], lammys[j]
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val

    """YOUR CODE HERE FOR Q1.2"""
    print('i, j = ', len(sigmas), len(lammys))

    for i in range(len(sigmas)):
        for j in range(len(lammys)):
            print('step', i, j)
            loss_fn = KernelLogisticRegressionLossL2(lammys[j])
            optimizer = GradientDescentLineSearch()
            RBF_kernel = GaussianRBFKernel(sigmas[i])
            klr_model = KernelClassifier(loss_fn, optimizer, RBF_kernel)
            klr_model.fit(X_train, y_train)

            train_errs[i][j] = np.mean(klr_model.predict(X_train) != y_train)
            val_errs[i][j] = np.mean(klr_model.predict(X_val) != y_val)

    #------- min train
    argmin_train_errs = np.unravel_index(np.argmin(train_errs, axis=None), train_errs.shape)
    print('min train_err:\n', train_errs[argmin_train_errs])
    min_train_sigma, min_train_lammy = sigmas[argmin_train_errs[0]], lammys[argmin_train_errs[1]]
    print('min train_error occurs for (sigma, lammy):', (min_train_sigma, min_train_lammy) )
    
    loss_fn = KernelLogisticRegressionLossL2(min_train_lammy)
    optimizer = GradientDescentLineSearch()
    RBF_kernel = GaussianRBFKernel(min_train_sigma)
    klr_model = KernelClassifier(loss_fn, optimizer, RBF_kernel)
    klr_model.fit(X_train, y_train)
    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")
    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("GridSearch_min_train_Kernel.png", fig)

    #------- min val
    argmin_val_errs = np.unravel_index(np.argmin(val_errs, axis=None), val_errs.shape)
    print('min val_err:\n', val_errs[argmin_val_errs])
    min_val_sigma, min_val_lammy = sigmas[argmin_val_errs[0]], lammys[argmin_val_errs[1]]
    print('min val_error occurs for (sigma, lammy):', (min_val_sigma, min_val_lammy) )

    loss_fn = KernelLogisticRegressionLossL2(min_val_lammy)
    optimizer = GradientDescentLineSearch()
    RBF_kernel = GaussianRBFKernel(min_val_sigma)
    klr_model = KernelClassifier(loss_fn, optimizer, RBF_kernel)
    klr_model.fit(X_train, y_train)
    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")
    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("GridSearch_min_val_Kernel.png", fig)

    # Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    utils.savefig("logRegRBF_grids.png", fig)


@handle("3.2")
def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = utils.standardize_cols(X_train)
    n, d = X_train_standardized.shape

    # Matrix plot
    fig, ax = plt.subplots()
    ax.imshow(X_train_standardized)
    utils.savefig("animals_matrix.png", fig)

    # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples

    fig, ax = plt.subplots()
    ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    for i in random_is:
        xy = X_train_standardized[i, [j1, j2]]
        ax.annotate(animal_names[i], xy=xy)
    utils.savefig("animals_random.png", fig)

    """YOUR CODE HERE FOR Q3"""
    pca_encoder = PCAEncoder(2)
    pca_encoder.fit(X_train)
    Z = pca_encoder.encode(X_train)
    fig, ax = plt.subplots()
    ax.scatter(Z[:, 0], Z[:, 1])
    for i in random_is:
        xy = Z[i, :]
        ax.annotate(animal_names[i], xy=xy)
    utils.savefig("animals_pca.png", fig)
    pca_w = pca_encoder.W[0, :]
    print(trait_names.shape)
    print(np.argmax(pca_w))
    print(np.argmin(pca_w))
    print(trait_names[np.argmax(abs(pca_w))])
    print(trait_names[np.argmax(abs(pca_encoder.W[1, :]))])
    X_st = X_train - mu
    ve = np.linalg.norm(Z @ pca_encoder.W - X_st)**2 / np.linalg.norm(X_st)**2
    print(ve)

@handle("4")
def q4():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = utils.standardize_cols(X_train_orig)
    X_val, _, _ = utils.standardize_cols(X_val_orig, mu, sigma)

    # Train ordinary regularized least squares
    loss_fn = LeastSquaresLoss()
    optimizer = GradientDescentLineSearch()
    model = LinearModel(loss_fn, optimizer, check_correctness=False)
    model.fit(X_train, y_train)
    print(model.fs)  # ~700 seems to be the global minimum.

    print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
    print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    utils.savefig("gd_line_search_curve.png", fig)


@handle("4.1")
def q4_1():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = utils.standardize_cols(X_train_orig)
    X_val, _, _ = utils.standardize_cols(X_val_orig, mu, sigma)

    """YOUR CODE HERE FOR Q4.1"""
    loss_fn = LeastSquaresLoss()

    g_optimizer = GradientDescent()
    lr_getter = ConstantLR(0.0003)
    sg_optimizer = StochasticGradient(g_optimizer, lr_getter, batch_size=1, max_evals=10)
    
    model = LinearModel(loss_fn, sg_optimizer, check_correctness=False)
    model.fit(X_train, y_train)
    print(model.fs) 

    print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
    print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Stochastic Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    utils.savefig("sgd_b1_line_search_curve.png", fig)

@handle("4.3")
def q4_3():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = utils.standardize_cols(X_train_orig)
    X_val, _, _ = utils.standardize_cols(X_val_orig, mu, sigma)

    """YOUR CODE HERE FOR Q4.3"""
    loss_fn = LeastSquaresLoss()

    g_optimizer = GradientDescent()
    lr_getter = InverseSqrtLR(0.1)
    sg_optimizer = StochasticGradient(g_optimizer, lr_getter, batch_size=10, max_evals=60)
    
    model = LinearModel(loss_fn, sg_optimizer, check_correctness=False)
    model.fit(X_train, y_train)
    print(model.fs) 

    print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
    print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Stochastic Gradient descent epochs")
    ax.set_ylabel("Objective function f value")
    utils.savefig("InverseSqrtLR_sgd_line_search_curve.png", fig)


if __name__ == "__main__":
    main()
