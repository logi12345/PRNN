# PRNN
Implementations of different algorithms used for pattern recognition and in neural networks.

## Sequential Perceptron Learning
Algorithm adjusts weights for misclassified samples.

## Delta Learning Rule
Algorithm adjusts weights for every sample in the dataset until no updates can happen.

## Karhunen Loeve Transform
Calculate the mean of the data samples. 

Find the zero mean of each sample. 

Find the covariance of all the samples. 

Compute eigenvalues and eigenvectors. 

Order the eigenvectors from largest to smallest based on the eigenvalues and remove the smallest. 

Project the data on the principal components by multiplying the transposed eigenvectors with the zero mean of each sample.