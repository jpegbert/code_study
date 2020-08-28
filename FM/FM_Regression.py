import pandas as pd
import numpy as np
from scipy.sparse import csr
from itertools import count
from collections import defaultdict
import tensorflow as tf


"""
采用FM完成回归任务
https://mp.weixin.qq.com/s/hKRGD02LumttFsJt_WyILQ
"""


def vectorize_dic(dic, label2index=None, hold_num=None):
    if label2index == None:
        d = count(0)
        label2index = defaultdict(lambda: next(d))  # 数值映射表

    sample_num = len(list(dic.values())[0])  # 样本数
    feat_num = len(list(dic.keys()))  # 特征数
    total_value_num = sample_num * feat_num

    col_ix = np.empty(total_value_num, dtype=int)

    i = 0
    for k, lis in dic.items():
        col_ix[i::feat_num] = [label2index[str(k) + str(el)] for el in lis]
        i += 1

    row_ix = np.repeat(np.arange(sample_num), feat_num)
    data = np.ones(total_value_num)

    if hold_num is None:
        hold_num = len(label2index)

    left_data_index = np.where(col_ix < hold_num)  # 为了剔除不在train set中出现的test set数据

    return csr.csr_matrix(
        (data[left_data_index], (row_ix[left_data_index], col_ix[left_data_index])),
        shape=(sample_num, hold_num)), label2index


def batcher(X_, y_, batch_size=-1):
    assert X_.shape[0] == len(y_)

    n_samples = X_.shape[0]
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = y_[i:upper_bound]
        yield (ret_x, ret_y)


def load_dataset():
    cols = ['user', 'item', 'rating', 'timestamp']
    train = pd.read_csv('../data/ml-100k/ua.base', delimiter='\t', names=cols)
    test = pd.read_csv('../data/ml-100k/ua.test', delimiter='\t', names=cols)

    x_train, label2index = vectorize_dic({'users': train.user.values, 'items': train.item.values})
    x_test, label2index = vectorize_dic({'users': test.user.values, 'items': test.item.values}, label2index,
                                        x_train.shape[1])

    y_train = train.rating.values
    y_test = test.rating.values

    x_train = x_train.todense()
    x_test = x_test.todense()

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_dataset()

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

vec_dim = 10
batch_size = 1000
epochs = 10
learning_rate = 0.001
sample_num, feat_num = x_train.shape

x = tf.placeholder(tf.float32, shape=[None, feat_num], name="input_x")
y = tf.placeholder(tf.float32, shape=[None, 1], name="ground_truth")

w0 = tf.get_variable(name="bias", shape=(1), dtype=tf.float32)
W = tf.get_variable(name="linear_w", shape=(feat_num), dtype=tf.float32)
V = tf.get_variable(name="interaction_w", shape=(feat_num, vec_dim), dtype=tf.float32)

linear_part = w0 + tf.reduce_sum(tf.multiply(x, W), axis=1, keep_dims=True)
interaction_part = 0.5 * tf.reduce_sum(tf.square(tf.matmul(x, V)) - tf.matmul(tf.square(x), tf.square(V)), axis=1,
                                       keep_dims=True)
y_hat = linear_part + interaction_part
loss = tf.reduce_mean(tf.square(y - y_hat))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        step = 0
        print("epoch:{}".format(e))
        for batch_x, batch_y in batcher(x_train, y_train, batch_size):
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y.reshape(-1, 1)})
            step += 1
            if step % 10 == 0:
                for val_x, val_y in batcher(x_test, y_test):
                    train_loss = sess.run(loss, feed_dict={x: batch_x, y: batch_y.reshape(-1, 1)})
                    val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y.reshape(-1, 1)})
                    print("batch train_mse={}, val_mse={}".format(train_loss, val_loss))

    for val_x, val_y in batcher(x_test, y_test):
        val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y.reshape(-1, 1)})
        print("test set rmse = {}".format(np.sqrt(val_loss)))

