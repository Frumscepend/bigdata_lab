from __future__ import division
import pandas as pd
import numpy as np
from math import sqrt
import math
import operator
from collections import Counter


url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
df.columns = [u'Чашелистик длина, см',
              u'Чашелистик ширина, см',
              u'Лепесток длина, см',
              u'Лепесток ширина, см',
             'Class']


def test_and_train(df, proportion):
    mask = np.random.rand(len(df)) < proportion
    return df[mask], df[~mask]
train, test = test_and_train(df, 0.67)


def euclidean_distance(instance1, instance2):
    squares = [(i-j)**2 for i, j in zip(instance1, instance2)]
    return sqrt(sum(squares))


def get_neighbours(instance, train, k, h):
    distances = []
    index = 0
    for i in train.ix[:, :5].values:
        y = 0
        if i[4] == "Iris-setosa":
            y = 1
        if i[4] == "Iris-versicolor":
            y = 2
        if i[4] == "Iris-virginica":
            y = 3
        i = i[:-1]
        weight = (1 - k) * k * y
        r = 2
        h = 1
        while r > 1:
            r = euclidean_distance(i, instance) * weight / h
            h += 1
        distances.append((1 - abs(r)) * r)
        index += 1
    distances = tuple(zip(distances, train[u'Class'].values))
    return sorted(distances, key=operator.itemgetter(0))[:k]


def get_response(neighbours):
    return Counter(neighbours).most_common()[0][0][1]


def get_predictions(train, test, k, h):
    predictions = []
    for i in test.ix[:, :-1].values:
        neighbours = get_neighbours(i, train, k, h)
        response = get_response(neighbours)
        predictions.append(response)
    return predictions


def get_tp(test, predictions, type):
    return sum([i == j and j == type for i, j in zip(test[u'Class'].values, predictions)])


def get_fp(test, predictions, type):
    return sum([i != j and j == type for i, j in zip(test[u'Class'].values, predictions)])


def get_tn(test, predictions, type):
    return sum([i == j and j != type for i, j in zip(test[u'Class'].values, predictions)])


def get_fn(test, predictions, type):
    return sum([i != j and j != type for i, j in zip(test[u'Class'].values, predictions)])


def get_precision(test, predictions, type):
    return get_tp(test, predictions, type) / (get_tp(test, predictions, type) + get_fp(test, predictions, type))


def get_recall(test, predictions, type):
    return get_tp(test, predictions, type) / (get_tp(test, predictions, type) + get_fn(test, predictions, type))


def get_accuracy(test, predictions, type):
    return (get_tp(test, predictions, type) + get_tn(test, predictions, type)) / len(predictions)
    # return mean([i == j for i, j in zip(test[u'Class'].values, predictions)])

predictions = get_predictions(train, test, 5, 5)
for type in set(predictions):
    print(type, " accuracy : ", get_accuracy(test, predictions, type))
    print(type, " precision : ", get_precision(test, predictions, type))
    print(type, " recall : ", get_recall(test, predictions, type))
print("")

y_actu = pd.Series(test[u'Class'].values, name='Actual')
y_pred = pd.Series(predictions, name='Predicted')
print(pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True))
