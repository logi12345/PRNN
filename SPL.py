import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from copy import deepcopy
from DataProcessor import DataProcessor as dp


def plot_graph(w, co):
    intercept = -w[0] / w[2]
    gradient = -w[1] / w[2]
    x = np.arange(start=0, stop=4)
    y = gradient * x + intercept
    plt.scatter(*zip(*co))
    plt.plot(x, y, label='g(x) = w^tx + w^0')
    plt.grid(alpha=.4, linestyle='--')
    plt.show()


def spl(data, a, lr, co):
    count = 0
    npa = np.array(a)
    co_copy = deepcopy(co)
    plot_graph(npa, co_copy)
    for i in range(len(data)):
        data['xt'][i].insert(0, 1)
    while count < 4:
        for row in data.itertuples():
            gx = npa.dot(np.array([row[1]]).T)
            if row[2] > 0 and gx[0] < 0 or row[2] < 0 and gx[0] > 0:
                npa = npa + lr * row[2] * np.array(row[1])
                plot_graph(npa, co_copy)
        count += 1


def step_function(wx):
    if wx < 0:
        return 0
    else:
        return 1


def delta_learning_rule(data, a, lr, co):
    count = 0
    npa = np.array(a)
    co_copy = deepcopy(co)
    plot_graph(npa, co_copy)
    for i in range(len(data)):
        data['xt'][i].insert(0, 1)
    while count < 4:
        for row in data.itertuples():
            y = step_function(npa.dot(np.array([row[1]]).T))
            npa = npa + lr * (row[2] - y) * np.array(row[1])
            plot_graph(npa, co_copy)
        count += 1


coord = [[0, 0], [1, 0], [2, 1], [0, 1], [1, 2]]
d = dp.create_data(coord, [1, 1, 1, 0, 0])
at = [-1.5, 5, -1]

# spl(d, at, 1, coord)
delta_learning_rule(d, at, 1, coord)
