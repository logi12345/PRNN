import numpy as np
from pandas import DataFrame


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

    def create_column_vector(self, x):
        x = self.transform_into_transposed_vector(x)
        return np.transpose(x)
