from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
from collections import defaultdict
import math


number_of_folders = 10


def read_folder(i):
    path = "messages/part" + str(i)
    X = []
    Y = []
    for filename in os.listdir(path):
        with open(path + "/" + filename, 'r') as file:
            subject = list(map(int, file.readline().split()[1:]))
            file.readline()
            text = list(map(int, file.readline().split()))
            subject.extend(text)
            X.append(subject)
            Y.append(0 if "legit" in filename else 1)
    return X, Y


def read_all_folders():
    X = []
    Y = []
    for i in range(1, number_of_folders + 1):
        current_X, current_Y = read_folder(i)
        X.append(current_X)
        Y.append(current_Y)
    return X, Y


def list_to_string(lst):
    return "".join(map(str, lst))


def get_statistics(X, Y):
    frequency = defaultdict(lambda: 0)
    legit, spam = 0, 0
    for i in range(len(X)):
        text = X[i]
        message_type = "legit" if Y[i] == 0 else "spam"
        if message_type == "legit":
            legit += 1
        else:
            spam += 1
        for word in text:
            frequency[list_to_string(word), message_type] += 1
    for word, message_type in frequency:
        frequency[list_to_string(word), message_type] /= (legit if message_type == "legit" else spam)
    legit /= len(X)
    spam /= len(X)
    return frequency, legit, spam


def transform_to_n_gram(lst, n):
    n_gram = []
    for i in range(len(lst) - n + 1):
        n_gram.append(lst[i:(i + n)])
    return n_gram


def predict(statistics, message, alpha, legit_penalty=1, spam_penalty=1):
    frequency, legit, spam = statistics
    number_of_messages = legit + spam
    prior_legit_probability = math.log(legit / number_of_messages)
    prior_spam_probability = math.log(spam / number_of_messages)
    conditional_legit_probability, conditional_spam_probability = 0, 0
    for word in message:
        str_word = list_to_string(word)
        word_frequency = frequency[str_word, "legit"] + frequency[str_word, "spam"]
        conditional_legit_probability += math.log((frequency[str_word, "legit"] + alpha) / (word_frequency + 2 * alpha))
        conditional_spam_probability += math.log((frequency[str_word, "spam"] + alpha) / (word_frequency + 2 * alpha))
    legit_probability = prior_legit_probability + conditional_legit_probability + math.log(legit_penalty)
    spam_probability = prior_spam_probability + conditional_spam_probability + math.log(spam_penalty)
    return 0 if legit_probability > spam_probability else 1


def get_parameters(X, Y, alpha, legit_penalty=1, spam_penalty=1):
    kf = KFold(n_splits=number_of_folders)
    accuracy, sensitivity, specificity = 0, 0, 0
    total_FN = 0
    for train_indices, test_indices in kf.split(X):
        X_train, Y_train = list(chain.from_iterable(X[train_indices])), list(chain.from_iterable(Y[train_indices]))
        X_test, Y_test = list(chain.from_iterable(X[test_indices])), list(chain.from_iterable(Y[test_indices]))
        statistics = get_statistics(X_train, Y_train)
        predictions = []
        for i in range(len(X_test)):
            prediction = predict(statistics, X_test[i], alpha, legit_penalty, spam_penalty)
            predictions.append(prediction)
        TP, FP, FN, TN = count_metrics(predictions, Y_test)
        accuracy += (TP + TN) / (TP + TN + FP + FN)
        sensitivity += TP / (TP + FN)
        specificity += TN / (TN + FP)
        total_FN += FN
    return accuracy / number_of_folders, sensitivity / number_of_folders, specificity / number_of_folders, total_FN


def count_metrics(predictions, real):
    TP = FP = FN = TN = 0
    for i in range(len(predictions)):
        if predictions[i] == real[i] == 0:
            TP += 1
        elif predictions[i] == 0 and real[i] == 1:
            FP += 1
        elif predictions[i] == 1 and real[i] == 0:
            FN += 1
        elif predictions[i] == real[i] == 1:
            TN += 1
    return TP, FP, FN, TN


def process_dataset(n):
    X, Y = read_all_folders()
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = transform_to_n_gram(X[i][j], n)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def draw_graph(x_values, y_values, x_name, y_name, color, x_scale_log=False, is_roc_curve=False):
    plt.plot(x_values, y_values, color=color)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if x_scale_log:
        plt.xscale('log')
    if is_roc_curve:
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
    plt.show()


def build_graph_accuracy_from_alpha(X, Y, color='b'):
    alpha_values = [10 ** i for i in range(-9, 1)]
    accuracy_values = []
    for alpha in alpha_values:
        accuracy_values.append(get_parameters(X, Y, alpha)[0])
    draw_graph(alpha_values, accuracy_values, 'alpha', 'accuracy', color, True)


def build_graph_FN_from_legit_penalty(X, Y, alpha, color='r'):
    legit_penalty_values = [10 ** (10 * i) for i in range(0, 28, 3)]
    total_FN_values = []
    for legit_penalty in legit_penalty_values:
        total_FN_values.append(get_parameters(X, Y, alpha, legit_penalty)[3])
    draw_graph(legit_penalty_values, total_FN_values, 'legit penalty', 'false negative', color, False)


def build_graph_accuracy_from_legit_penalty(X, Y, alpha, color='g'):
    legit_penalty_values = [10 ** (10 * i) for i in range(0, 28, 3)]
    accuracy_values = []
    for legit_penalty in legit_penalty_values:
        accuracy_values.append(get_parameters(X, Y, alpha, legit_penalty)[0])
    draw_graph(legit_penalty_values, accuracy_values, 'legit penalty', 'accuracy', color, False)


def build_ROC_curve(X, Y, alpha):
    sensitivity_values = [0]
    anti_specificity_values = [0]
    legit_penalty_values = [10 ** (10 * i) for i in range(-27, 28, 3)]
    for legit_penalty in legit_penalty_values:
        accuracy, sensitivity, specificity, total_FN = get_parameters(X, Y, alpha, legit_penalty)
        sensitivity_values.append(sensitivity)
        anti_specificity_values.append(1 - specificity)
    sensitivity_values.append(1)
    anti_specificity_values.append(1)
    draw_graph(anti_specificity_values, sensitivity_values, '1 - specificity', 'sensitivity', '#164dcb', False, True)


X, Y = process_dataset(1)
build_graph_accuracy_from_alpha(X, Y)
build_graph_FN_from_legit_penalty(X, Y, 1e-9)
build_graph_accuracy_from_legit_penalty(X, Y, 1e-9)
build_ROC_curve(X, Y, 1e-9)
