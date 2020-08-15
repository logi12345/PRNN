import numpy as np
from DataProcessor import DataProcessor as dataprep
from numpy import linalg as LA

x = [[-1.3536, 0.4098], [0.0, 0.0], [2.2353, 0.0621], [-0.8817, -0.4719]]
dp = dataprep()
data = dataprep.create_data(x)


def pca(dataset):
    mean = 0
    covariance = 0

    # Transpose data into column vector in dataframe
    for i in range(len(dataset)):
        dataset['input'][i] = dp.create_column_vector(dataset['input'][i])
        mean = mean + dataset['input'][i]

    # Find mean of data vectors
    mean = mean / len(dataset)

    # Create a new zero mean columns in dataset
    dataset['zero_mean'] = dataset.apply(lambda row: row.input - mean, axis=1)

    # Calculate the covariance of the zero mean
    for i in range(len(dataset)):
        covariance = covariance + dataset['zero_mean'][i] * np.transpose(dataset['zero_mean'][i])
    covariance = covariance / len(dataset)

    # Compute the eigenvalues and eigenvectors respectively
    w, v = LA.eigh(covariance)
    w = np.around(w, decimals=4)
    v = np.around(v, decimals=4)

    # Order data from largest to smallest using the eigenvalues and remove the smallest column
    column_to_remove = np.argmin(w)
    w = np.delete(w, column_to_remove)

    ordered_index_list = w.argsort()[::-1]
    idx = np.empty_like(ordered_index_list)
    idx[ordered_index_list] = np.arange(len(ordered_index_list))

    v = np.delete(v, column_to_remove, 1)
    v = v[:, idx]

    # Project data onto space spanning first two principle components
    dataset['new_value'] = dataset.apply(lambda row: np.around(np.transpose(v).dot(row.zero_mean), decimals=4), axis=1)
    print(dataset['new_value'])


pca(data)
