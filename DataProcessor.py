import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing


class DataProcessor:
    def __init__(self):
        return

    @staticmethod
    def create_dataframe(x, classes=None):
        de = {'input': x, 'classes': classes} if classes else {'input': x}
        return DataFrame(data=de)

    @staticmethod
    def custom_encoder(labels):
        new_labels = np.transpose(np.array([labels]))
        c_labels = []
        for label in new_labels:
            if label[0] == 1:
                c = np.append(label, 0)
            else:
                c = np.append(label, 1)
            c_labels.append(c)
        return np.array(c_labels)

    def create_2d_moon_shape_data(self):
        np.random.seed(0)
        feature_set, labels = datasets.make_moons(300, noise=0.20)
        labels = self.custom_encoder(labels)

        return feature_set, labels

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
