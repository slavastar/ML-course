# from sklearn import tree
# import pandas as pd
#
# model = tree.DecisionTreeClassifier
#
#
# def read_data(i):
#     train = pd.read_csv('data/{:02d}_train.csv'.format(i))
#     X_train = train.iloc[:, :-1].to_numpy()
#     Y_train = train.iloc[:, -1].to_numpy()
#     test = pd.read_csv('data/{:02d}_test.csv'.format(i))
#     X_test = test.iloc[:, :-1].to_numpy()
#     Y_test = test.iloc[:, -1].to_numpy()
#     return X_train, Y_train, X_test, Y_test

from sklearn.datasets import load_iris
from sklearn import tree
X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
tree.plot_tree(clf)
