# coding: utf-8
__author__ = 'qingyi'
import numpy as np
import matplotlib.pyplot as plt

sampleDataSet =np.array(
  [[1, 1, 2, 4, 2],
  [1, 3, 3, 4, 4]])
mean_data = np.mean(sampleDataSet,1)

print('mean_data =\n', mean_data)

move_mean_sample = (sampleDataSet.transpose() - mean_data).transpose()

print('move_mean_sample =\n', move_mean_sample)

p1 = plt.subplot(121)
p1.plot(sampleDataSet[0,:],sampleDataSet[1,:],'*')
p1.axis([0,5,0,5])
p2 = plt.subplot(122)
p2.plot(move_mean_sample[0,:],move_mean_sample[1,:],'*')
p2.axis([-5,5,-5,5])

np_cov = np.cov(move_mean_sample,rowvar=True)

print('sampleDataSet =\n', sampleDataSet)
print('move_mean_sample =\n', move_mean_sample)
print('np_cov =\n', np_cov)

mat_cov = np.mat(np_cov)

print('mat_cov =\n', mat_cov)

(eigV, eigVector) = np.linalg.eigh(mat_cov)

print('(eigV, eigVector)', (eigV, eigVector))

pca_mat = eigVector[:,-1]

print('pca_mat =\n', pca_mat)

pca_data = pca_mat.T * np.mat(move_mean_sample) #Y = P^T * X

print('pca_data =\n', pca_data)

recon_data = ((pca_mat * pca_data).transpose() + mean_data).transpose() #X = P*Y

print('recon_data =\n', recon_data)

p1.plot(recon_data[0,:],recon_data[1,:],'o')
p1.axis([0,5,0,5])

k = pca_mat[1,0]/pca_mat[0,0]
b = recon_data[1,0] - k*recon_data[0,0]

print('k=\n', k)
print('b=\n', b)

xx = [0,5]
k = int(k)
yy = k * xx + b
p1.plot(xx,yy)

print('eigV=\n',eigV)
print('eigVector=\n',eigVector)
print('pca_data=\n',pca_data)
plt.show()

class PCA(object):
    def __init__(self):
        print('use PCA(Principal Components Analysis) algorithm to reduce the dimension....')


    def pca_dimension_reduction(self, sampleDataSet):
        '''

        :param sampleDataSet: n*m矩阵，n为每条样本的维度，m为数据集中样本的个数；
                            其是原来m*n矩阵()的转置
        :return:
        '''
        mean_data = np.mean(sampleDataSet, 1)
        move_mean_sample = (sampleDataSet.transpose() - mean_data).transpose()
        np_cov = np.cov(move_mean_sample, rowvar=True)
        mat_cov = np.mat(np_cov)
        (eigV, eigVector) = np.linalg.eigh(mat_cov)
        pca_mat = eigVector[:, -1]
        pca_data = pca_mat.T * np.mat(move_mean_sample)
        print(pca_data)