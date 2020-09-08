# https://blog.csdn.net/weixin_45459911/article/details/105397228
# 引入所需包
import numpy as np
import random


#读取数据函数,输入为数据文件名和训练、测试切分比率，返回为list类型的训练数据集和测试数据集


def loadData(fileName, ratio):
    trainingData = []
    testData = []
    with open(fileName) as txtData:
        lines=txtData.readlines()
        for line in lines:
            lineData=line.strip().split(',')    #去除空白和逗号“,”
            if random.random() < ratio:             #数据集分割比例
                trainingData.append(lineData)   #训练数据集列表
            else:
                testData.append(lineData)       #测试数据集列表
            np.savetxt('./diabetes_train.txt', trainingData, delimiter=',',fmt = '%s')
            np.savetxt('./diabetes_test.txt', testData, delimiter=',',fmt = '%s')

    return trainingData, testData


iris_file = './diabetes.csv'
ratio = 0.7
trainingData, testData = loadData(iris_file, ratio)   # 加载文件，按一定比率切分为训练样本和测试样本
