import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha, plot_bars_early_stopping_mse_comparison

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""

__author__ = 'bellec,subramoney'


def calculate_mse(nn, x, y):
    """
    Calculate the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO
    return mean_squared_error(y, nn.predict(x))


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    # using 8 hidden neurons
    n_h = 40
    # use random state for a fixed split - guarantees always same output (200 seems beautiful)
    nn = MLPRegressor(activation='logistic', solver='lbfgs', alpha=0.0, hidden_layer_sizes=(n_h,), max_iter=200, random_state=200)
    nn.fit(x_train, y_train)

    pred_train_y = nn.predict(x_train)
    pred_test_y = nn.predict(x_test)

    plot_learned_function(n_h, x_train, y_train, pred_train_y, x_test, y_test, pred_test_y)
    pass


def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    n_h = 8;
    mse_array = np.zeros(10)
    for i in range(0, 10):
        nn = MLPRegressor(activation='logistic', solver='lbfgs', alpha=0.0, hidden_layer_sizes=(n_h,), max_iter=200, random_state=i)
        nn.fit(x_train, y_train)
        mse_array[i] = calculate_mse(nn, x_test, y_test)

    print("Different mean squared errors:\n", mse_array)
    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model
    Use max_iter = 10000 and tol=1e-8
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    n_h = [1,2,3,4,6,8,12,20,40]
    train_array = np.zeros((9,10))
    test_array = np.zeros((9,10))

    for n in range(0,9):
        for i in range(0, 10):
            nn = MLPRegressor(tol=1e-8,activation='logistic', solver='lbfgs', alpha=0.0, hidden_layer_sizes=(n_h[n],), max_iter=1000, random_state=i)
            nn.fit(x_train, y_train)
            train_array[n][i] = calculate_mse(nn, x_train, y_train)
            test_array[n][i] = calculate_mse(nn, x_test, y_test)


    plot_mse_vs_neurons(np.array(train_array),np.array(test_array),n_h)

    pass


def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    Use n_iterations = 10000 and tol=1e-8
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    pass


def ex_1_2_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    pass


def ex_1_2_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 b)
    Remember to set alpha and momentum to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    pass


def ex_1_2_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 c)
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    ## TODO
    pass
