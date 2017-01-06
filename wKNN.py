from __future__ import division
import pandas as pd
import numpy as np
from math import sqrt
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


def get_neighbours(instance, train):
    distances = []
    for i in train.ix[:, :-1].values:
        distances.append(euclidean_distance(instance, i))
    weight = 0
    for d in distances:
        if (d > 0):
            weight = weight + 1.0/d
    weight = 4 * weight / len(distances) + 1
    distances = tuple(zip(distances, train[u'Class'].values))
    return sorted(distances, key=operator.itemgetter(0))[:int(weight)]


def get_response(neighbours):
    return Counter(neighbours).most_common()[0][0][1]


def get_predictions(train, test):
    predictions = []
    for i in test.ix[:, :-1].values:
        neighbours = get_neighbours(i, train)
        response = get_response(neighbours)
        predictions.append(response)
    return predictions


def mean(instance):
    return sum(instance)/len(instance)


def get_accuracy(test, predictions):
    return mean([i == j for i, j in zip(test[u'Class'].values, predictions)])


print(get_accuracy(test, get_predictions(train, test)))
