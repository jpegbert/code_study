import pandas as pd

users = ["User1", "User2", "User3", "User4", "User5"]
items = ["Item A", "Item B", "Item C", "Item D", "Item E"]
# 用户购买记录数据集
datasets = [
    [5, 3, 4, 4, None],
    [3, 1, 2, 3, 3],
    [4, 3, 4, 3, 5],
    [3, 3, 1, 5, 4],
    [1, 5, 5, 2, 1],
]

df = pd.DataFrame(datasets, columns=items, index=users)

# 直接计算皮尔逊相关系数
# 默认是按列进行计算，因此如果计算用户间的相似度，当前需要进行转置
print("物品之间的两两相似度：")
item_similar = df.corr()
print(item_similar.round(4))

# 可以看到与物品A最相似的物品分别是物品E和物品D。
# 注意：我们在预测评分时，往往是通过与其有正相关的用户或物品进行预测，如果不存在正相关的情况，那么将无法做出预测。
# 这一点尤其是在稀疏评分矩阵中尤为常见，因为稀疏评分矩阵中很难得出正相关系数。

# 后面是评分预测，评分预测需要基于用户相似度进行加权平均得到
# 用户u对物品i的评分预测公式是 r_{u, i} = (sum_{j \in related i}(sim(i, j) * r_{uj}) / sum_{j \in related i}(sim(i, j)))
