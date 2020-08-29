import numpy as np
from copy import deepcopy
from DataProcessor import DataProcessor as dp


def spl(data, a, lr, co):
    count = 0
    npa = np.array(a)
    co_copy = deepcopy(co)
    dp.plot_graph(npa, co_copy)
    for i in range(len(data)):
        data['input'][i].insert(0, 1)
    while count < 4:
        for row in data.itertuples():
            gx = npa.dot(np.array([row[1]]).T)
            if row[2] > 0 and gx[0] < 0 or row[2] < 0 and gx[0] > 0:
                npa = npa + lr * row[2] * np.array(row[1])
                dp.plot_graph(npa, co_copy)
        count += 1


coord = [[0, 0], [1, 0], [2, 1], [0, 1], [1, 2]]
d = dp.create_dataframe(coord, [1, 1, 1, 0, 0])
at = [-1.5, 5, -1]

spl(d, at, 1, coord)

