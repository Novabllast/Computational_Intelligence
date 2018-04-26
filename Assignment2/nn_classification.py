from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import *
import numpy as np
from sklearn.metrics import confusion_matrix

ADAM = 'adam'

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
    ## TODO

    iteration = 200
    hidden_units = 6
    mlp_classifier = MLPClassifier(activation=ACTIVATION, solver='adam', hidden_layer_sizes=(hidden_units,),
                                   max_iter=iteration)
    pose = target2[:, 1]
    mlp_classifier.fit(input2, pose)
    coefs_ = mlp_classifier.coefs_
    hidden_layer_weights = coefs_[0]  # using index zero because using index 1 => exception

    y_pred = mlp_classifier.predict(input2)
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
    iteration = 1000
    hidden_units = 20
    mlp_classifier = MLPClassifier(activation=ACTIVATION, solver=ADAM, hidden_layer_sizes=(hidden_units,),
                                   max_iter=iteration)

    train_acc = []
    test_acc = []
    training_person = target1[:, 0]
    test_person = target2[:, 0]

    for i in range(0, 10):
        mlp_classifier.random_state = i
        mlp_classifier.fit(input1, training_person)
        predict_test = mlp_classifier.predict(input2)
        predict_training = mlp_classifier.predict(input1)
        # test_acc.append(mlp_classifier.score(test_person, predict_test))      # TODO throws exception
        # train_acc.append(mlp_classifier.score(training_person, predict_training))# TODO throws exception
        test_acc.append(accuracy_score(test_person, predict_test))
        train_acc.append(accuracy_score(training_person, predict_training))

    best_network = max(test_acc)
    best_network_index = test_acc.index(best_network)

    mlp_classifier.random_state = best_network_index
    mlp_classifier.fit(input1, training_person)

    y_pred = mlp_classifier.predict(input2)
    matrix = confusion_matrix(test_person, y_pred)
    print("The Confusion Matrix we obtained: \n" + str(matrix))

    plot_histogram_of_acc(train_acc, test_acc)

    annas_favorit_number = 177
    strugers_favorit_number = 42
    marcos_favorit_numer = 490357
    manfreds_favorit_number_aka_best_number = 7
    best_numbers_ever = [annas_favorit_number, strugers_favorit_number, marcos_favorit_numer,
                         manfreds_favorit_number_aka_best_number]

    for i in best_numbers_ever:
        misclassified_images = input2[i, :]
        # plot_random_images(misclassified_images) # TODO throws exception
