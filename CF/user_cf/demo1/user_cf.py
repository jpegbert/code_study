import random
import math
import json
import os

"""
实例：编写一个基于UserCF算法的电影推荐系统
数据集：MovieLens。下载地址：https://grouplens.org/datasets/movielens/1m/
"""


class UserCFRec:
    def __init__(self, datafile):
        self.datafile = datafile
        self.data = self.loadData()

        self.trainData, self.testData = self.splitData(3, 47)  # 训练集与数据集
        self.users_sim = self.UserSimilarityBest()

    # 加载评分数据到data
    def loadData(self):
        print("加载数据...")
        data = []
        for line in open(self.datafile):
            userid, itemid, record, _ = line.split("::")
            data.append((userid, itemid, int(record)))
        return data

    """
    拆分数据集为训练集和测试集
        k: 参数
        seed: 生成随机数的种子
        M: 随机数上限
    """
    def splitData(self, k, seed, M=8):
        print("训练数据集与测试数据集切分...")
        train, test = {}, {}
        random.seed(seed)
        for user, item, record in self.data:
            if random.randint(0, M) == k:
                test.setdefault(user, {})
                test[user][item] = record
            else:
                train.setdefault(user, {})
                train[user][item] = record
        return train, test

    # 计算用户之间的相似度，采用惩罚热门商品和优化算法复杂度的算法
    def UserSimilarityBest(self):
        print("开始计算用户之间的相似度 ...")
        if os.path.exists("../../data/user_sim.json"):
            print("用户相似度从文件加载 ...")
            userSim = json.load(open("../../data/user_sim.json", "r"))
        else:
            # 得到每个item被哪些user评价过
            item_users = dict()
            for u, items in self.trainData.items():
                for i in items.keys():
                    item_users.setdefault(i, set())
                    if self.trainData[u][i] > 0:
                        item_users[i].add(u)
            # 构建倒排表
            count = dict()
            user_item_count = dict()
            for i, users in item_users.items():
                for u in users:
                    user_item_count.setdefault(u, 0)
                    user_item_count[u] += 1
                    count.setdefault(u, {})
                    for v in users:
                        count[u].setdefault(v, 0)
                        if u == v:
                            continue
                        count[u][v] += 1 / math.log(1+len(users))
            # 构建相似度矩阵
            userSim = dict()
            for u, related_users in count.items():
                userSim.setdefault(u, {})
                for v, cuv in related_users.items():
                    if u == v:
                        continue
                    userSim[u].setdefault(v, 0.0)
                    userSim[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v])
            json.dump(userSim, open('../../data/user_sim.json', 'w'))
        return userSim

    """
    为用户user进行物品推荐
        user: 为用户user进行推荐
        k: 选取k个近邻用户
        nitems: 取nitems个物品
    """
    def recommend(self, user, k=8, nitems=40):
        result = dict()
        have_score_items = self.trainData.get(user, {})
        for v, wuv in sorted(self.users_sim[user].items(), key=lambda x: x[1], reverse=True)[0:k]:
            for i, rvi in self.trainData[v].items():
                if i in have_score_items:
                    continue
                result.setdefault(i, 0)
                result[i] += wuv * rvi
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:nitems])

    """
    计算准确率
        k: 近邻用户数
        nitems: 推荐的item个数
    """
    def precision(self, k=8, nitems=10):
        print("开始计算准确率 ...")
        hit = 0
        precision = 0
        for user in self.trainData.keys():
            tu = self.testData.get(user, {})
            rank = self.recommend(user, k=k, nitems=nitems)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision += nitems
        return hit / (precision * 1.0)


if __name__ == '__main__':
    cf = UserCFRec("../../data/ratings.dat")
    result = cf.recommend("1")
    print("user '1' recommend result is {} ".format(result))

    precision = cf.precision()
    print("precision is {}".format(precision))

