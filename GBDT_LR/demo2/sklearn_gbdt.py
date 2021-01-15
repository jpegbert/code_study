# 除了 pandas 中的 get_dummies()，sklearn 也提供了一种对 Dataframe 做 One-hot 的方法。
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


gbm1 = GradientBoostingClassifier(n_estimators=50, random_state=10, subsample=0.6, max_depth=7, min_samples_split=900)
gbm1.fit(X_train, Y_train)
# model.apply(X_train)返回训练数据 X_train 在训练好的模型里每棵树中所处的叶子节点的位置（索引）
train_new_feature = gbm1.apply(X_train)
train_new_feature = train_new_feature.reshape(-1, 50)

enc = OneHotEncoder()
# OneHotEncoder() 首先 fit() 过待转换的数据后，再次 transform() 待转换的数据，就可实现对这些数据的所有特征进行 One-hot 操作。
enc.fit(train_new_feature)

# # 每一个属性的最大取值数目
# print('每一个特征的最大取值数目:', enc.n_values_)
# print('所有特征的取值数目总和:', enc.n_values_.sum())

# 由于 transform() 后的数据格式不能直接使用，所以最后需要使用.toarray() 将其转换为我们能够使用的数组结构。
train_new_feature2 = np.array(enc.transform(train_new_feature).toarray())
