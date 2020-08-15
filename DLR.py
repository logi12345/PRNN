import numpy as np
from copy import deepcopy
from DataProcessor import DataProcessor as dp


def step_function(wx):
    if wx < 0:
        return 0
    else:
        return 1


def delta_learning_rule(data, a, lr, co):
    count = 0
    npa = np.array(a)
    co_copy = deepcopy(co)
    dp.plot_graph(npa, co_copy)
    for i in range(len(data)):
        data['input'][i].insert(0, 1)
    while count < 4:
        for row in data.itertuples():
            y = step_function(npa.dot(np.array([row[1]]).T))
            npa = npa + lr * (row[2] - y) * np.array(row[1])
            dp.plot_graph(npa, co_copy)
        count += 1


coord = [[0, 0], [1, 0], [2, 1], [0, 1], [1, 2]]
d = dp.create_data(coord, [1, 1, 1, 0, 0])
at = [-1.5, 5, -1]

delta_learning_rule(d, at, 1, coord)
