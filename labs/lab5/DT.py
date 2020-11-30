from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def print_best_hyperparameters(hyperparameters):
    print("Criterion:", hyperparameters[0])
    print("Splitter:", hyperparameters[1])
    print("Depth:", hyperparameters[2])


def read_dataset(dataset_index):
    train = pd.read_csv('data/{:02d}_train.csv'.format(dataset_index))
    X_train = train.iloc[:, :-1].to_numpy()
    Y_train = train.iloc[:, -1].to_numpy()
    test = pd.read_csv('data/{:02d}_test.csv'.format(dataset_index))
    X_test = test.iloc[:, :-1].to_numpy()
    Y_test = test.iloc[:, -1].to_numpy()
    return X_train, Y_train, X_test, Y_test


# dataset 5: criterion = gini, splitter = best, depth = 1, accuracy = 0.9956
# dataset 21: criterion = entropy, splitter = best, depth = 13, accuracy = 0.8053
def find_datasets_with_min_and_max_heights():
    for dataset_index in range(1, 22):
        X_train, Y_train, X_test, Y_test = read_dataset(dataset_index)
        best_hyperparameters = ["", "", 0]
        best_accuracy = 0
        for criterion in ["gini", "entropy"]:
            for splitter in ["best", "random"]:
                for depth in [i for i in range(1, 21)]:
                    clf = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
                    clf.fit(X_train, Y_train)
                    prediction = clf.predict(X_test)
                    accuracy = np.sum((prediction == Y_test)) / len(Y_test)
                    if best_accuracy < accuracy:
                        best_hyperparameters = [criterion, splitter, depth]
                        best_accuracy = accuracy
        print("--- Dataset", dataset_index, "---")
        print_best_hyperparameters(best_hyperparameters)
        print("Accuracy:", best_accuracy)


def draw_graph(dataset_index, x_values, y_values, x_name, y_name, color):
    plt.title("Dataset " + str(dataset_index))
    plt.plot(x_values, y_values, color=color)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def build_graph_accuracy_from_depth(dataset_index, criterion, splitter):
    X_train, Y_train, X_test, Y_test = read_dataset(dataset_index)
    accuracy_values = []
    depth_values = [i for i in range(1, 21)]
    for depth in depth_values:
        clf = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_test)
        accuracy = np.sum((prediction == Y_test)) / len(Y_test)
        accuracy_values.append(accuracy)
    draw_graph(dataset_index, depth_values, accuracy_values, "depth", "accuracy", "#9C34E0")


build_graph_accuracy_from_depth(5, criterion="gini", splitter="best")
build_graph_accuracy_from_depth(21, criterion="entropy", splitter="best")
