import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self):
        return

    @staticmethod
    def create_data(x, classes=None):
        de = {'input': x, 'classes': classes} if classes else {'input': x}
        return DataFrame(data=de)

    @staticmethod
    def transform_into_transposed_vector(x):
        if type(x) is np.ndarray:
            try:
                x = x.reshape(1, len(x))
            except:
                raise Exception('incorrect dimensions should have a shape of (n,) or (n,1)')
        elif type(x) is list:
            x = np.array(x)
            try:
                x = x.reshape(1, len(x))
            except:
                raise Exception('incorrect dimensions should have a shape of [] not [[]]')
        else:
            print('invalid input')
        return x

    @staticmethod
    def plot_graph(w, co):
        intercept = -w[0] / w[2]
        gradient = -w[1] / w[2]
        x = np.arange(start=0, stop=4)
        y = gradient * x + intercept
        plt.scatter(*zip(*co))
        plt.plot(x, y, label='g(x) = w^tx + w^0')
        plt.grid(alpha=.4, linestyle='--')
        plt.show()

    def create_column_vector(self, x):
        x = self.transform_into_transposed_vector(x)
        return np.transpose(x)
