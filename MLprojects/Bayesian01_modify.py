# coding: utf-8
# filename = '/tmp/temp/Bayesian/pima-indians-diabetes.data.csv'
# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math


class DataSet(object):
    def __init__(self):
        self.fileName = '/tmp/temp/Bayesian/pima-indians-diabetes.data.csv'
        self.splitRatio = 0.67
        self.dataSet = None
        self.trainSet = None
        self.testSet = None


    def loadCsv(self):
        lines = csv.reader(open(self.fileName, 'r'))
        self.dataSet = list(lines)
        for i in range(len(self.dataSet)):
            self.dataSet[i] = [float(x) for x in self.dataSet[i]]


    def splitDataSet(self):
        trainSize = int(len(self.dataSet) * self.splitRatio)
        self.trainSet = []
        self.testSet = list(self.dataSet)
        while len(self.trainSet) < trainSize:
            index = random.randrange(len(self.testSet))
            self.trainSet.append(self.testSet.pop(index))


    def main(self):
        self.loadCsv()
        self.splitDataSet()
        print('Split {0} rows into train={1} and test={2} rows'.format(len(self.dataSet), len(self.trainSet), len(self.testSet)))


class NBayesian(object):
    def __init__(self):
        dataSet =  DataSet()
        dataSet.main()
        self.dataSet = dataSet.dataSet
        self.trainSet = dataSet.trainSet
        self.testSet = dataSet.testSet

        self.model = None


    def separateByClass(self, dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        return separated


    def mean(self, numbers):
        return sum(numbers) / float(len(numbers))


    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)


    def summarize(self, dataSet):
        summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*dataSet)]
        del summaries[-1]
        return summaries

    def summarizeByClass(self, dataset):
        separated = self.separateByClass(dataset)
        summaries = {}
        for classValue, instances in separated.items():
            summaries[classValue] = self.summarize(instances)
        return summaries

    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculateClassProbabilities(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
        return probabilities

    def predict(self, summaries, inputVector):
        probabilities = self.calculateClassProbabilities(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel


    def getPredictions(self, summaries, testSet):
        predictions = []
        for i in range(len(testSet)):
            result = self.predict(summaries, testSet[i])
            predictions.append(result)
        return predictions


    def getAccuracy(self, predictions):
        correct = 0
        for i in range(len(self.testSet)):
            if self.testSet[i][-1] == predictions[i]:
                correct += 1
        return (correct / float(len(self.testSet))) * 100.0


    def train(self):
        self.model = self.summarizeByClass(self.trainSet)


    def test(self):
        predictions = self.getPredictions(self.model, self.testSet)
        accuracy = self.getAccuracy(predictions)
        print('Accuracy: {0}%'.format(accuracy))


def main():
    nb = NBayesian()
    nb.train()
    nb.test()


if __name__ == '__main__':
    main()