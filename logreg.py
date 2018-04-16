#!/usr/bin/env python
import numpy as np

from logreg_toolbox import sig

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) (Logistic Regression)
TODO Fill the cost function and the gradient
"""


def cost(theta, x, y):
    """
    Cost of the logistic regression function.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: cost
    """
    ##############
    #
    # TODO
    #
    # Write the cost of logistic regression as defined in the lecture
    # Hint:
    #   - use the logistic function sig imported from the file toolbox
    #   - sums of logs of numbers close to zero might lead to numerical errors, try splitting the cost into the sum
    # over positive and negative samples to overcome the problem. If the problem remains note that low errors is not
    # necessarily a problem for gradient descent because only the gradient of the cost is used for the parameter updates

    # m  --> number of training examples
    m, n = x.shape

    dot_product = np.dot(x, theta)
    mean_error = dot_product - y
    mean_squared_error = np.power(mean_error, 2)

    hypo = sig(dot_product)  # logistic regression hypothesis
    positiv_samples = y[y > 0]
    negative_samples = y[y < 0]

    c = (-y) # * np.log(hypo) - (1 - y) * np.log(1 - hypo)  # log-likelihood vector

    # END TODO
    ###########

    return c


def grad(theta, x, y):
    """

    Compute the gradient of the cost of logistic regression

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: gradient
    """

    ##############
    #
    # TODO
    #

    g = np.zeros(theta.shape)

    m, n = x.shape

    dot_product = np.dot(x, theta)
    mean_error = dot_product - y
    hypo = sig(dot_product)

    grad = (1 / m) * mean_error.dot(x)

    # END TODO
    ###########

    return grad
