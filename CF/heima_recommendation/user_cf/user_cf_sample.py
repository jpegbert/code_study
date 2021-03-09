import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import pairwise_distances
from pprint import pprint
import numpy as np


"""
最简单的user cf，把都买和不购买看做1和0，直接使用杰卡德计算相抵度
"""


users = ["User1", "User2", "User3", "User4", "User5"]
items = ["Item A", "Item B", "Item C", "Item D", "Item E"]
# 用户购买记录数据集，1表示购买，0表示没购买
datasets = [
    [1, 0, 1, 1, 0],
    [1, 0, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
]

df = pd.DataFrame(datasets, columns=items, index=users)
print(df)


# 直接计算某两项的杰卡德相似系数
# 计算Item A 和Item B的相似度
print(jaccard_score(df["Item A"], df["Item B"]))


# 计算所有的数据两两的杰卡德相似系数
# 计算用户间相似度
user_similar = 1 - pairwise_distances(df.values, metric="jaccard")
user_similar = pd.DataFrame(user_similar, columns=users, index=users)
print("用户之间的两两相似度：")
print(user_similar)

topN_users = {}
# 遍历每一行数据
for i in user_similar.index:
    # 取出每一列数据，并删除自身，然后排序数据
    _df = user_similar.loc[i].drop([i]) # 这样drop表示按照索引删除
    # 按照相似度降序排序
    _df_sorted = _df.sort_values(ascending=False)
    top2 = list(_df_sorted.index[:2])
    topN_users[i] = top2

print("Top2相似用户：")
pprint(topN_users)

rs_results = {}
# 构建推荐结果
for user, sim_users in topN_users.items():
    rs_result = set()    # 存储推荐结果
    for sim_user in sim_users:
        # 构建初始的推荐结果
        rs_result = rs_result.union(set(df.loc[sim_user].replace(0, np.nan).dropna().index))
    # 过滤掉已经购买过的物品
    rs_result -= set(df.loc[user].replace(0, np.nan).dropna().index)
    rs_results[user] = rs_result
print("最终推荐结果：")
pprint(rs_results)
