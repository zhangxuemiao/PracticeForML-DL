# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import csv
from PCA import PcaReducer
import random

class DataSet():
    def __init__(self, fileName, splitRatio):
        self.fileName = fileName # './dataForVisualization/pima-indians-diabetes.data.csv'
        self.records = None # 所有的记录，一条记录包括，一个样本及样本分类标签
        self.samples = None # 所有的样本，不含分类标签
        self.sampleLabels = None # 所有样本的分类标签，与样本一一对应
        self.separatedSamplesByClass = None

        self.splitRatio = splitRatio # 0.67
        self.dataSet = None
        self.trainSet = None
        self.testSet = None


    def loadDataSetTxt(self):
        with open(self.fileName) as f:
            recordsList = []
            for line in f.readlines():
                splits = line.split()
                if len(splits) != 3:
                    continue
                splits = [float(splits[0]),float(splits[1]), int(splits[2])]
                recordsList.append(splits)
            self.records = np.array(recordsList) # 先使用Python原生列表来存数据，然后再将列表转换成numpy中的ndarray对象
            self.samples = self.records[:, :2]
            labelsTemp = self.records[:, 2:]
            labelsTempList = []
            for i in range(labelsTemp.shape[0]):
                labelsTempList.append(labelsTemp[i, 0])
            self.sampleLabels = np.array(labelsTempList)
            print(self.records.shape)
            print(self.samples.shape)
            print(self.sampleLabels.shape)


    def loadDataSetCsv(self):
        recordsList = []
        csv_reader = csv.reader(open(self.fileName))
        for row in csv_reader:
            if len(row) != 9:
                continue
            row = [float(x) for x in row]
            recordsList.append(row)
        self.records = np.array(recordsList)
        self.samples = self.records[:, :8]
        labelsTemp = self.records[:, 8:]
        labelsTempList = []
        for i in range(labelsTemp.shape[0]):
            labelsTempList.append(labelsTemp[i, 0])
        self.sampleLabels = np.array(labelsTempList)
        print('records.shape', self.records.shape)
        print('samples.shape', self.samples.shape)
        print('sampleLabels.shape', self.sampleLabels.shape)


    def splitDataSet(self):
        trainSize = int(len(self.dataSet) * self.splitRatio)
        self.trainSet = []
        self.testSet = list(self.dataSet)
        while len(self.trainSet) < trainSize:
            index = random.randrange(len(self.testSet))
            self.trainSet.append(self.testSet.pop(index))


    # 按照分类将样本归类
    def separatedByClass(self):
        shape = self.records.shape
        positiveSamples = []
        negativeSampels = []
        for i in range(shape[0]):
            if self.records[i,8] > 0.5:
                positiveSamples.append(self.records[i,:8].tolist())
            if self.records[i,8] < 0.5:
                negativeSampels.append(self.records[i,:8].tolist())
        self.separatedSamplesByClass = {'pos': positiveSamples, 'neg': negativeSampels}
        print(self.separatedSamplesByClass)


    # 绘制2维样本散点图
    def drawScatterDiagram(self):
        '''
        此方法针对原样本就是二维的，不需要降维操作即可直接画散点图
        :return:
        '''
        positiveXX = []
        positiveYY = []
        negativeXX = []
        negativeYY = []
        for posSample in self.separatedSamplesByClass['pos']:
            positiveXX.append(posSample[0])
            positiveYY.append(posSample[1])
        for negSample in self.separatedSamplesByClass['neg']:
            negativeXX.append(negSample[0])
            negativeYY.append(negSample[1])

        plt.figure()
        subplt = plt.subplot(1,1,1)
        subplt.set_title('scatter diagram')
        plt.xlabel('dimension-1')
        plt.ylabel('dimension-2')
        subplt.scatter(positiveXX, positiveYY, c='green')
        subplt.scatter(negativeXX, negativeYY, c='red')
        subplt.legend(['pos', 'neg'])
        plt.show()

    # 绘制n(n>2)维样本散点图
    def drawScatterDiagramByPCA(self):
        '''
        当样本的维度n大于2时，通过PCA算法将样本的维度降到2，然后再画散点图
        :return:
        '''
        pcaReducer = PcaReducer()
        samplesAfterPCA = {'pos': None, 'neg': None}
        samplesAfterPCA['pos'] = pcaReducer.pca(self.separatedSamplesByClass['pos'], 2)[0]
        samplesAfterPCA['neg'] = pcaReducer.pca(self.separatedSamplesByClass['neg'], 2)[0]

        positiveXX = []
        positiveYY = []
        negativeXX = []
        negativeYY = []

        for i in range(samplesAfterPCA['pos'].shape[0]):
            positiveXX.append(samplesAfterPCA['pos'][i, 0])
            positiveYY.append(samplesAfterPCA['pos'][i, 1])
        for j in range(samplesAfterPCA['neg'].shape[0]):
            negativeXX.append(samplesAfterPCA['neg'][j, 0])
            negativeYY.append(samplesAfterPCA['neg'][j, 1])

        print(len(negativeXX), len(negativeXX))

        # for posSample in samplesAfterPCA['pos'].tolist():
        #     print('posSample=\n', posSample)
        #     positiveXX.append(posSample[0])
        #     positiveYY.append(posSample[1])
        #     break
        # for negSample in samplesAfterPCA['neg'].tolist():
        #     negativeXX.append(negSample[0])
        #     negativeYY.append(negSample[1])

        plt.figure()
        subplt = plt.subplot(1,1,1)
        subplt.set_title('scatter diagram')
        plt.xlabel('dimension-1')
        plt.ylabel('dimension-2')
        subplt.scatter(positiveXX, positiveYY, c='green')
        subplt.scatter(negativeXX, negativeYY, c='red')
        subplt.legend(['pos', 'neg'])
        plt.show()


if __name__ == '__main__':
    dataSet = DataSet('./dataForVisualization/pima-indians-diabetes.data.csv', 0.67)
    dataSet.loadDataSetCsv()
    dataSet.separatedByClass()
    dataSet.drawScatterDiagramByPCA()
    pass