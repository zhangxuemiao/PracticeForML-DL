# coding: utf-8
from numpy import *
import time
import matplotlib.pyplot as plt


#calculate kernel value
def calcKernelValue(matrix_x, sample_x, kernelOption):
    kernelType = kernelOption[0]
    numSamples = matrix_x.shape[0]
    kernelValue = mat(zeros((numSamples, 1)))

    if kernelType == 'linear':
        kernelValue = matrix_x * sample_x.T
    elif kernelType == 'rbf':
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1.0
        for i in xrange(numSamples):
            diff = matrix_x[i, :] - sample_x
            kernelValue[i] = exp(diff*diff.T/(-2.0*sigma**2))
    else:
        raise NameError('Not support kernel type! You can use linear or rbf!')
    return kernelValue


# calculate kernel matrix given train set and kernel type
def calcKernelMatrix(train_x, kernelOption):
    numSamples = train_x.shape[0]
    kernelMatrix = mat(zeros((numSamples, numSamples)))
    for i in xrange(numSamples):
        kernelMatrix[:, i] = calcKernelMatrix(train_x, train_x[i, :], kernelOption)