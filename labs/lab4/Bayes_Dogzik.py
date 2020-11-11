import numpy as np
import sklearn as sk
from os import listdir
from os.path import isfile, join
import sklearn.metrics as skmetrics
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import math
import matplotlib.pyplot as plt


def read_file(filename):
    with open(filename, "r") as f:
        subject = f.readline().split()[1:]
        f.readline()
        body = f.readline().split()
        distinct_words = subject + body
        is_spam = "spmsg" in filename
        return np.array(distinct_words), int(is_spam)


def read_part(num):
    dirname = "messages/part" + str(num)
    files = [dirname + "/" + f for f in listdir(dirname) if isfile(join(dirname, f))]
    data = map(read_file, files)
    unziped_data = list(zip(*data))
    return list(unziped_data[0]), list(unziped_data[1])


def read_all_parts():
    X = []
    Y = []
    for i in range(1, 11):
        cur_X, cur_Y = read_part(i)
        X.extend(cur_X)
        Y.extend(cur_Y)
    return np.array(X), np.array(Y)


X, Y = read_all_parts()


# all probalities are repsresented as pairs of numinator(x[0]) and denuminator(x[1])
class Bayes(BaseEstimator):
    def __init__(self, alpha=1, lambdas=[1, 1]):
        cnt = len(lambdas)
        self.alpha = alpha
        self.lambdas = lambdas
        self.classes_cnt = cnt
        self.classes_prob = None
        self.words_prob = None
        self.distinct_words = None

    def fit(self, X, Y):
        self.classes_prob = np.zeros((self.classes_cnt, 2), dtype=int)
        self.distinct_words = np.zeros(self.classes_cnt, dtype=int)
        self.words_prob = [{} for _ in range(self.classes_cnt)]

        default_word_prob = np.array([self.alpha, 0], dtype=int)
        for i in range(X.shape[0]):
            cur_words = set(X[i])
            cur_class = Y[i]
            for word in cur_words:
                self.words_prob[cur_class].setdefault(word, default_word_prob)
                self.words_prob[cur_class][word][0] += 1
            self.classes_prob[cur_class][0] += 1

        for cur_class in range(self.classes_cnt):
            distinct = len(self.words_prob[cur_class])
            self.distinct_words[cur_class] = distinct
            self.classes_prob[cur_class][1] = self.classes_cnt
            for word in self.words_prob[cur_class].keys():
                self.words_prob[cur_class][word][1] = self.classes_prob[cur_class][0] + distinct * self.alpha

    def get_prob(self, cur_class, word):
        zero_numinator = self.alpha
        zero_denuminator = self.classes_prob[cur_class][0] + self.alpha * self.distinct_words[cur_class]
        zero_prob = np.array([zero_numinator, zero_denuminator])
        return self.words_prob[cur_class].get(word, zero_prob)

    def predict_one(self, cur_msg):
        metrics = np.log(self.classes_prob)
        cur_words = set(cur_msg)
        for cur_class in range(self.classes_cnt):
            metrics[cur_class][0] += math.log(self.lambdas[cur_class])
            for word in cur_words:
                metrics[cur_class] += np.log(self.get_prob(cur_class, word))
        return np.argmax(metrics[0] - metrics[1])

    def predict(self, X):
        return np.vectorize(self.predict_one)(X)


def get_score(y_true, y_pred):
    tmp = y_true - y_pred
    if -1 in tmp:
        return -np.inf
    else:
        return skmetrics.accuracy_score(y_true, y_pred)
my_scorer = skmetrics.make_scorer(get_score)


bayes = Bayes()
a = 9
