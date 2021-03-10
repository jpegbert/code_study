import pandas as pd
import numpy as np


"""
将评分看作是一个连续的值而不是离散的值，那么就可以借助线性回归思想来预测目标用户对某物品的评分。其中一种实现策略被称为Baseline（基准预测）

#### Baseline：基准预测
Baseline设计思想基于以下的假设：
- 有些用户的评分普遍高于其他用户，有些用户的评分普遍低于其他用户。比如有些用户天生愿意给别人好评，心慈手软，比较好说话，而有的人就比较苛刻，总是评分不超过3分（5分满分）
- 一些物品的评分普遍高于其他物品，一些物品的评分普遍低于其他物品。比如一些物品一被生产便决定了它的地位，有的比较受人们欢迎，有的则被人嫌弃。
这个用户或物品普遍高于或低于平均值的差值，我们称为偏置(bias)

**Baseline目标：**
- 找出每个用户普遍高于或低于他人的偏置值$b_u$
- 找出每件物品普遍高于或低于其他物品的偏置值$b_i$
- 我们的目标也就转化为寻找最优的$b_u$和$b_i$

使用Baseline的算法思想预测评分的步骤如下：
- 计算所有电影的平均评分$\mu$（即全局平均评分）
- 计算每个用户评分与平均评分$\mu$的偏置值$b_u$
- 计算每部电影所接受的评分与平均评分$\mu$的偏置值$b_i$
- 预测用户对电影的评分：
pred_b_{ui} = mu + b_{u} + b_{i}
其中：mu是全局的均值，b_{u}是每个用户评分与与平均评分mu的偏置，b_{i}是每部电影所接受的评分与平均评分mu的偏置

这里可以
"""


class BaselineCFBySGD(object):
    def __init__(self, number_epochs, alpha, reg, columns=["uid", "iid", "rating"]):
        # 梯度下降最高迭代次数
        self.number_epochs = number_epochs
        # 学习率
        self.alpha = alpha
        # 正则参数
        self.reg = reg
        # 数据集中user-item-rating字段的名称
        self.columns = columns

    def fit(self, dataset):
        '''
        :param dataset: uid, iid, rating
        :return:
        '''
        self.dataset = dataset
        # 用户评分数据
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        # 物品评分数据
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        # 计算全局平均分
        self.global_mean = self.dataset[self.columns[2]].mean()
        # 调用sgd方法训练模型参数
        self.bu, self.bi = self.sgd()

    def sgd(self):
        '''
        利用随机梯度下降，优化bu，bi的值
        :return: bu, bi
        '''
        # 初始化bu、bi的值，全部设为0
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))

        for i in range(self.number_epochs):
            print("iter%d" % i)
            for uid, iid, real_rating in self.dataset.itertuples(index=False):
                error = real_rating - (self.global_mean + bu[uid] + bi[iid])

                bu[uid] += self.alpha * (error - self.reg * bu[uid])
                bi[iid] += self.alpha * (error - self.reg * bi[iid])

        return bu, bi

    def predict(self, uid, iid):
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating


if __name__ == '__main__':
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
    dataset = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols=range(3), dtype=dict(dtype))

    bcf = BaselineCFBySGD(20, 0.1, 0.1, ["userId", "movieId", "rating"])
    bcf.fit(dataset)

    while True:
        uid = int(input("uid: "))
        iid = int(input("iid: "))
        print(bcf.predict(uid, iid))
