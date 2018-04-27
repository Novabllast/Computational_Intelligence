from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from Assignment2.nn_classification_plot import *
import numpy as np
from sklearn.metrics import confusion_matrix

__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!

the first column codes the person
the second column the pose
the third column the emotion
the last column indicates whether the person is wearing sunglasses
"""

ACTIVATION = 'tanh'


def ex_2_1(input2, target2):
    """
    Solution for exercise 2.1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    ## TODO - done

    hidden_units = 6
    nn = MLPClassifier(activation=ACTIVATION, solver='adam', hidden_layer_sizes=(hidden_units,),
                       max_iter=200)
    pose = target2[:, 1]
    nn.fit(input2, pose)

    # using index 0 because of just one hidden layer
    hidden_layer_weights = nn.coefs_[0]

    y_pred = nn.predict(input2)
    matrix = confusion_matrix(pose, y_pred)

    print("The Confusion Matrix we obtained: \n" + str(matrix))

    plot_hidden_layer_weights(hidden_layer_weights)


def ex_2_2(input1, target1, input2, target2):
    """
    Solution for exercise 2.2
    :param input1: The input from dataset1
    :param target1: The target from dataset1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    ## TODO
    hidden_units = 20

    test_face = target2[:, 0]
    train_face = target1[:, 0]

    test_accuracy = np.zeros(10)
    train_accuracy = np.zeros(10)

    best_network = 0
    max_accuracy = 0
    nn = MLPClassifier(activation=ACTIVATION, solver="adam", hidden_layer_sizes=(hidden_units,),
                       max_iter=1000)

    for i in range(0, 10):
        nn.random_state = i

        nn.fit(input1, train_face)
        train_accuracy[i] = nn.score(input1, train_face)
        test_accuracy[i] = nn.score(input2, test_face)

        if test_accuracy[i] > max_accuracy:
            best_network = nn
            max_accuracy = test_accuracy[i]

    plot_histogram_of_acc(train_accuracy, test_accuracy)

    # Use the best network to calculate the confusion matrix for the test set.
    y_pred = best_network.predict(input2)
    matrix = confusion_matrix(test_face, y_pred)

    print("The Confusion Matrix we obtained: \n" + str(matrix))

    # Plot a few misclassified images.
    annas_favorit_number = 177
    marcos_favorit_numer = 490
    strugers_favorit_number_aka_best_mirp = 13
    manfreds_favorit_number_is_a_emirp_a_lucky_fortunate_sexy_and_happy_prime = 79
    best_numbers_ever = [annas_favorit_number, strugers_favorit_number_aka_best_mirp, marcos_favorit_numer,
                         manfreds_favorit_number_is_a_emirp_a_lucky_fortunate_sexy_and_happy_prime]

    for _ in best_numbers_ever:
        misclassified = np.where(test_face != best_network.predict(input2))
        plot_random_images(input2[misclassified])
