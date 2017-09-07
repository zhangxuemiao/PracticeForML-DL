# coding: utf-8
import numpy as np
class PcaReducer(object):
    def __init__(self):
        print('use PCA(Principal Components Analysis) algorithm to reduce the dimension....')


    def zeroMean(self, dataMat):
        '''
        零均值化

        假如原始数据集为矩阵dataMat，dataMat中每一行代表一个样本，每一列代表同一个特征。
        零均值化就是求每一列的平均值，然后该列上的所有数都减去这个均值。
        也就是说，这里零均值化是对每一个特征而言的，零均值化都，每个特征的均值变成0
        :param dataMat:
        :return: 该函数返回两个变量，newData是零均值化后的数据，meanVal是每个特征的均值，是给后面重构数据用的。
        '''
        meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
        newData=dataMat-meanVal
        return newData,meanVal


    def pca(self, dataMat, n):
        newData, meanVal = self.zeroMean(dataMat)

        '''
        求协方差矩阵

        covMat即所求的协方差矩阵; numpy中的cov函数用于求协方差矩阵，
        参数rowvar很重要！若rowvar=0，说明传入的数据一行代表一个样本，
        若非0，说明传入的数据一列代表一个样本。

        因为newData每一行代表一个样本，所以将rowvar设置为0。
        '''
        covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

        '''
        求特征值、特征矩阵

        eigVals存放特征值，行向量。
        eigVects存放特征向量，每一列带别一个特征向量。
        特征值和特征向量是一一对应的
        '''
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量

        '''
        保留主要的成分[即保留值比较大的前n个特征]

        第三步得到了特征值向量eigVals，假设里面有m个特征值，我们可以对其排序，
        排在前面的n个特征值所对应的特征向量就是我们要保留的，它们组成了新的特征空间的一组基n_eigVect。
        将零均值化后的数据乘以n_eigVect就可以得到降维后的数据。
        '''
        eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序

        '''
        首先argsort对特征值是从小到大排序的，那么最大的n个特征值就排在后面，
        所以eigValIndice[-1:-(n+1):-1]就取出这个n个特征值对应的下标。
        [python里面，list[a:b:c]代表从下标a开始到b，步长为c。]
        '''
        n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
        n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
        lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
        reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # reconMat是重构的数据，乘以n_eigVect的转置矩阵，再加上均值meanVal。

        '''
        这几步下来就可以从高维的数据dataMat得到低维的数据lowDDataMat，另外，程序也返回了重构数据reconMat，有些时候reconMat课便于数据分析。
        '''
        return lowDDataMat, reconMat


    def percentage2n(self, eigVals,percentage):
        sortArray=np.sort(eigVals)   #升序
        sortArray=sortArray[-1::-1]  #逆转，即降序
        arraySum=sum(sortArray)
        tmpSum=0
        num=0
        for i in sortArray:
            tmpSum+=i
            num+=1
            if tmpSum>=arraySum*percentage:
                return num


    def pcaByPercentage(self, dataMat,percentage=0.99):
        '''
        应用PCA的时候，对于一个1000维的数据，我们怎么知道要降到几维的数据才是合理的？
        即n要取多少，才能保留最多信息同时去除最多的噪声？一般，我们是通过方差百分比来确定n的，
        这一点在Ufldl教程中说得很清楚，并且有一条简单的公式

        参见： http://blog.csdn.net/u012162613/article/details/42177327
        :param dataMat:
        :param percentage:
        :return:
        '''
        newData,meanVal=self.zeroMean(dataMat)
        covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
        eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        n=self.percentage2n(eigVals,percentage)                 #要达到percent的方差百分比，需要前n个特征向量
        eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
        n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
        n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
        lowDDataMat=newData*n_eigVect               #低维特征空间的数据
        reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #reconMat是重构的数据，乘以n_eigVect的转置矩阵，再加上均值meanVal。
        return lowDDataMat,reconMat


if __name__ == '__main__':
    sampleDataSet = np.array(
        [[1, 1, 2, 4, 2],
         [1, 3, 3, 4, 4]])
    dataMat = sampleDataSet.T
    pcaReducer = PcaReducer()
    print(pcaReducer.pca(dataMat, 1))
