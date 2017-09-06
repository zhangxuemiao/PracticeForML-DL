# coding: utf-8
from Nbayes_lib import *

dataSet, listClasses = loadDataSet() # 导入外部数据集
nb = NBayes()
nb.train_set(dataSet, listClasses) # 训练数据集
nb.map2vocab(dataSet[1]) #随机选择一个测试句，这里2表示文本中的第三句话，不是脏话，应输出0
    # ['fuck', 'bitch', 'shit', 'fuck', 'bitch', 'I', 'hate', 'him','my'],
# vacabulary = ['fuck', 'bitch', 'shit', 'fuck', 'bitch', 'I', 'hate', 'him','my']
# nb.map2vocab(vacabulary)

print(nb.predict(nb.testset)) # 输出分类结果

print('vocabulary->------------------------------------', nb.vocablen)
print(nb.vocabulary)
print('tdm->--------------------------------------------', nb.tdm.shape)
print(nb.tdm)
print('tf->----------------------------------------------', nb.tf.shape)
print(nb.tf)
print('idf->--------------------------------------------', nb.idf.shape)
print(nb.idf)

print('Pcates->--------------------------------------------',)
print(nb.Pcates)

print('labels->--------------------------------------------', len(nb.labels))
print(nb.labels)

print('doclength->--------------------------------------------')
print(nb.doclength)



# self.Pcates = {}  # P(yi)--是个类别字典
# self.labels = []  # 对应每个文本的分类，是个外部导入的列表
# self.doclength = 0  # 训练集文本数
# self.vocablen = 0  # 词典词长
# self.testset = 0  # 测试集
# #   加载训练集并生成词典，以及tf, idf值