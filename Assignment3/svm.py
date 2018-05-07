import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

RBF = 'rbf'

LINEAR = 'linear'

"""
Computational Intelligence TU - Graz
Assignment 3: Support Vector Machine, Kernels & Multiclass classification
Part 1: SVM, Kernels

TODOS are all contained here.
"""

__author__ = 'bellec,subramoney'


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    # TODO:
    # Train an SVM with a linear kernel
    # and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    svc = svm.SVC(kernel=LINEAR)
    svc.fit(x, y)
    plot_svm_decision_boundary(svc, x, y)


def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    # TODO:
    # Add a point (4,0) with label 1 to the data set and then
    # train an SVM with a linear kernel
    # and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########

    x = np.append(x, [[4, 0]], axis=0)
    y = np.append(y, [1])

    svc = svm.SVC(kernel=LINEAR)
    svc.fit(x, y)
    plot_svm_decision_boundary(svc, x, y)


def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    # TODO:
    # Add a point (4,0) with label 1 to the data set and then
    # train an SVM with a linear kernel with different values of C
    # and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    Cs = [1e6, 1, 0.1, 0.001]

    x = np.append(x, [[4, 0]], axis=0)
    y = np.append(y, [1])

    svc = svm.SVC(kernel=LINEAR)

    for value in Cs:
        svc.C = value
        svc.fit(x, y)
        plot_svm_decision_boundary(svc, x, y)


def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    # TODO:
    # Train an SVM with a linear kernel for the given dataset
    # and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########

    svc = svm.SVC(kernel=LINEAR)
    svc.fit(x_train, y_train)

    plot_svm_decision_boundary(svc, x_train, y_train, x_test, y_test)

    score = svc.score(x_test, y_test)
    print(score)


def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    # TODO:
    # Train SVMs with polynomial kernels for different values of the degree
    # (Remember to set the 'coef0' parameter to 1)
    # and plot the variation of the test and training scores with polynomial degree using 'plot_score_vs_degree' func.
    # Plot the decision boundary and support vectors for the best value of degree
    # using 'plot_svm_decision_boundary' function
    ###########
    degrees = range(1, 21)
    train_score = []
    test_score = []

    svc = svm.SVC(kernel='poly', coef0=1)

    for degree in degrees:
        svc.degree = degree
        svc.fit(x_train, y_train)
        test_score.append(svc.score(x_test, y_test))
        train_score.append(svc.score(x_train, y_train))

    plot_score_vs_degree(train_score, test_score, degrees)

    highest_test_score = np.max(test_score)
    print("Highest test score: " + str(highest_test_score))
    svc.degree = highest_test_score
    svc.fit(x_train, y_train)
    plot_svm_decision_boundary(svc, x_train, y_train, x_test, y_test)


def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    # TODO:
    # Train SVMs with RBF kernels for different values of the gamma
    # and plot the variation of the test and training scores with gamma using 'plot_score_vs_gamma' function.
    # Plot the decision boundary and support vectors for the best value of gamma
    # using 'plot_svm_decision_boundary' function
    ###########
    gammas = np.arange(0.01, 2, 0.02)
    test_score = []
    train_score = []

    svc = svm.SVC(kernel=RBF)

    for gamma in gammas:
        svc.gamma = gamma
        svc.fit(x_train, y_train)
        train_score.append(svc.score(x_train, y_train))
        test_score.append(svc.score(x_test, y_test))

    plot_score_vs_gamma(train_score, test_score, gammas)

    highest_test_score = np.max(test_score)
    print("Highest test score: " + str(highest_test_score))
    svc.gamma = highest_test_score
    svc.fit(x_train, y_train)
    plot_svm_decision_boundary(svc, x_train, y_train, x_test, y_test)


def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    # TODO:
    # Train multi-class SVMs with one-versus-rest strategy with
    # - linear kernel
    # - rbf kernel with gamma going from 10**-5 to 10**5
    # - plot the scores with varying gamma using the function plot_score_versus_gamma
    # - Mind that the chance level is not .5 anymore and add the score obtained
    #   with the linear kernel as optional argument of this function
    ###########

    gammas = [pow(10, i) for i in range(-5, 6)]

    svc = svm.SVC(kernel=RBF, decision_function_shape='ovr')
    train_scores = []
    test_scores = []

    for gamma in gammas:
        svc.gamma = gamma
        svc.fit(x_train, y_train)
        train_scores.append(svc.score(x_train, y_train))
        test_scores.append(svc.score(x_test, y_test))

    lin_score_train = []
    plot_score_vs_gamma(train_scores, test_scores, gammas, lin_score_train=lin_score_train)
    highest_error_rate = np.max(test_scores)
    print("Highest error rate: " + highest_error_rate)


def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    # TODO:
    # Train multi-class SVMs with a LINEAR kernel
    # Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    # Find the index for which you get the highest error rate.
    # Plot the confusion matrix with plot_confusion_matrix.
    # Plot the first 10 occurrences of the most misclassified digit using plot_mnist.
    ###########

    labels = range(1, 6)
    y_pred = np.zeros(x_train.shape)

    sel_error = np.array([0])  # Numpy indices to select images that are misclassified.
    i = 0  # should be the label number corresponding the largest classification error

    # Plot with mnist plot
    plot_mnist(x_test[sel_error], y_pred[sel_error], labels=labels[i], k_plots=10, prefix='Predicted class')
