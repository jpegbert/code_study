import lightgbm as lgb
import numpy as np


params = {
'task': 'train',
'boosting_type': 'gbdt',
'objective': 'binary', # 二分类使用binary
'metric': {'binary_logloss'},
'num_leaves': 64,
'num_trees': 100,
'learning_rate': 0.01,
'feature_fraction': 0.9,
'bagging_fraction': 0.8,
'bagging_freq': 5,
'verbose': 0
}

# number of leaves,will be used in feature transformation
num_leaf = 64

print('Start training...')

# train
gbm = lgb.train(params=params,
train_set=lgb_train,
valid_sets=lgb_train, )

print('Start predicting...')

# y_pred 分别落在 100 棵树上的哪个节点上
# 返回训练数据在训练好的模型里预测结果所在的每棵树中叶子节点的位置（索引），形式为7999*100的二维数组。
y_pred = gbm.predict(x_train, pred_leaf=True)
y_pred_prob = gbm.predict(x_train)

result = []
threshold = 0.5
for pred in y_pred_prob:
    result.append(1 if pred > threshold else 0)
print('result:', result)

print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[1]) * num_leaf], dtype=np.int64) # N * num_tress * num_leafs
for i in range(0, len(y_pred)):
    # temp 表示在每棵树上预测的值所在节点的序号（0,64,128,...,6436 为 100 棵树的序号，中间的值为对应树的节点序号）
    temp = np.range(len(y_pred[0])) * num_leaf + np.array(y_pred[i]) # 构造 one-hot 训练数据集
    transformed_training_matrix[i][temp] += 1

y_pred = gbm.predict(x_test, pred_leaf=True)
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[1]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    # 构造 one-hot 测试数据集
    transformed_testing_matrix[i][temp] += 1

