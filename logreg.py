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

    hypo = sig(dot_product)  # logistic regression hypothesis
    positive_samples = 0.0
    negative_samples = 0.0

    for i in range(0, m):
        if y[i]:
            positive_samples += np.log(hypo[i])
        else:
            negative_samples += np.log(1 - hypo[i])

    c = (-1) * (1 / m) * (positive_samples + negative_samples)

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

    m, n = x.shape

    dot_product = np.dot(x, theta)
    hypo = sig(dot_product)
    mean_error = hypo - y

    gradient = np.zeros(n)
    for j in range(0, n):
        for i in range(0, m):
            gradient[j] += mean_error[i] * x[i][j]

    # END TODO
    ###########

    # sigma = cost(theta, x, y)
    # df = sigma * (1 - sigma)
    # print(df)

    # print((1 / m) * gradient)

    g = (1 / m) * gradient
    return g
