import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# python -m pip install matplotlib

df = pd.read_csv('sampleDataSet.csv')
# print df.shape

dg = pd.read_csv('sampleDataSet.csv', names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
# print dg.shape          # size (100, 9)
# print dg[['a', 'b']]    # printing columns a nd b
# print dg.ix[0]          # print raw 0
# print dg.a
# print dg.a.shape[0]
# print dg.a
# print dg.isnull().a
# print dg.isnull().sum()       # sum of null element
# print dg.dropna(axis=0)       # drop raws with null values
# print dg.dropna(axis=1)       # drop cols with null values
# print dg.drop(2, axis=0)      # drop raw 2
# print dg.fillna(0)            # fill NaNs with 0
# print dg.fillna(0).replace(0, 777)
# print dg['a'].nunique()
# print dg.mean(axis=1)
# print dg.head()
fg = lambda dm: dm.max() - dm.min()

# print dg.ix[:, 3:5] #.apply(f)

# plt.plot(dg[['a', 'b', 'c', 'd', 'e', 'f', 'g']].fillna(0))
# plt.show()

# grouped = dg[['a', 'b']].groupby(dg['i'], dg['c'])
# print grouped.mean()

dn = dg[np.random.rand(dg.shape[0]) > 0.5] = 33
# print dn


dh = pd.DataFrame(dg, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], index=range(0, 200))
# print dh

# try it out
data = pd.read_csv('lab03Exercise.csv', names=['channel1', 'channel2', 'channel3', 'channel4', 'channel5'])
print data
print data['channel1'].mean()
datana = data.fillna(data.mean())
print datana
# ndata = datana/2
# print ndata

f = lambda c1, c2, c3, c4, c5: 1 if ((c1 + c5) / 2 < (c2 + c3 + c4) / 3) else 0

datana['class'] = datana.apply(
    lambda row: f(row['channel1'], row['channel2'], row['channel3'], row['channel4'], row['channel5']), axis=1)
# datana['class'] = pd.Series(cls, index=datana.index)
# print datana

# scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
# plt.show()

"""
n = vec_1.shape[0]
m1 = np.tile(np.mean(vec_1), n)
m2 = np.tile(np.mean(vec_2), n)
d1 = vec_1 - m1
d2 = vec_2 - m2
mul = np.multiply(d1, d2)
s = np.sum(mul)

c_mat[j, k] = np.sum(np.multiply((mat.ix[j]-np.tile(mat.ix[j].mean(), n)), (mat.ix[k]-np.tile(mat.ix[k].mean(), n)))) / (n - 1)

"""
