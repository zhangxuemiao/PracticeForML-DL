# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np

class DataSet():
    def __init__(self):
        self.fileName = '/tmp/temp/SVM/testSet.txt'
        self.records = None # 所有的记录，一条记录包括，一个样本及样本分类标签
        self.samples = None # 所有的样本，不含分类标签
        self.sampleLabels = None # 所有样本的分类标签，与样本一一对应
        self.separatedSamplesByClass = None


    def loadDataSet(self):
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


    # 按照分类将样本归类
    def separatedByClass(self):
        shape = self.records.shape
        positiveSamples = []
        negativeSampels = []
        for i in range(shape[0]):
            if self.records[i,2] > 0:
                positiveSamples.append(self.records[i,:2].tolist())
            if self.records[i,2] < 0:
                negativeSampels.append(self.records[i,:2].tolist())
        self.separatedSamplesByClass = {'pos': positiveSamples, 'neg': negativeSampels}
        print(self.separatedSamplesByClass)


    # 绘制样本散点图
    def drawScatterDiagram(self):
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


if __name__ == '__main__':
    dataSet = DataSet()
    dataSet.loadDataSet()
    dataSet.separatedByClass()
    dataSet.drawScatterDiagram()