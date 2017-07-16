import pandas as pd
import numpy as np


class Data:
    def __init__(self, csv):
        self.raw_mat = pd.read_csv(csv, names=['a', 'b', 'c', 'd', 'e'])
        self.matrix = self.raw_mat.fillna(self.raw_mat.mean())


def covar(vec_1, vec_2):
    # covriance of two random vectors
    n = vec_1.shape[0]
    return np.sum(np.multiply((vec_1 - np.tile(np.mean(vec_1), n)), (vec_2 - np.tile(np.mean(vec_2), n)))) / (n - 1)


def corre(vec_1, vec_2):
    # correlation of two random vectors
    return covar(vec_1, vec_2) / (np.std(vec_1) * np.std(vec_2))


def covar_mat(mat):
    # covar. matrix of n obervations (rows) of k random variables (columns)
    mat = mat.T
    i = mat.shape[0]
    n = mat.shape[1]
    c_mat = np.zeros((i, i))
    for j in range(0, i):
        for k in range(0, i):
            c_mat[j][k] = np.sum(
                np.multiply((mat.ix[j] - np.tile(mat.ix[j].mean(), n)), (mat.ix[k] - np.tile(mat.ix[k].mean(), n)))) / (n - 1)
    return c_mat


def corre_mat(mat):
    # corre. matrix of n obervations (rows) of k random variables (columns)
    mat = mat.T
    i = mat.shape[0]
    cov = np.cov(mat)
    c_mat = np.zeros((i, i))
    for j in range(0, i):
        for k in range(0, i):
            c_mat[j][k] = cov[j][k] / np.sqrt(cov[j][j] * cov[k][k])
    return c_mat


data = Data('lab03Exercise.csv')
print data.matrix
print "covarinace matrix:"
print covar_mat(data.matrix)
print "\n", "correlation matrix:"
print corre_mat(data.matrix)


# print data.matrix
# print 'mat'
# print data.matrix.cov()
# print data.matrix.corr()
# print 'mat'
# print covar_mat(data.matrix)
# print corre_mat(data.matrix)
# print 'mat'
# print np.cov(data.matrix.T)
# print np.corrcoef(data.matrix.T)
