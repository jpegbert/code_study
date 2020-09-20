"""
参考：https://mp.weixin.qq.com/s/i1zCTosIEREKZzx_3-tl1A
下面用加利福利房价数据集简单地实现多输入的Wide & Deep模型
tensorflow版本：tensorflow 2.0.0-alpha0
matplotlib 2.0.2
numpy 1.16.2
pandas 0.20.3
sklearn 0.20.0
tensorflow 2.0.0-alpha0
tensorflow.python.keras.api._v2.keras 2.2.4-tf
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

housing = fetch_california_housing()
print(housing.DESCR) # 打印数据集描述
print(housing.data.shape) # (20640, 8)
print(housing.target.shape) # (20640,)

x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)
print(x_train.shape, y_train.shape) # (11610, 8) (11610,)
print(x_valid.shape, y_valid.shape) # (3870, 8) (3870,)
print(x_test.shape, y_test.shape) # (5160, 8) (5160,)

# 数据集标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

# 构建多输入wide&deep模型
input_wide = keras.layers.Input(shape=[5])  # wide模型输入5列特征
input_deep = keras.layers.Input(shape=[6])  # deep模型输入6列特征
# 函数式写法
hidden1 = keras.layers.Dense(30, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2]) # 连接两个模型
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_wide, input_deep], outputs=[output])
model.compile(loss="mean_squared_error", optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
model.summary()

"""
Wide模型输入取数据集前5个特征，Deep模型输入取数据集后6个特征。
"""
x_train_scaled_wide = x_train_scaled[:, :5]
x_train_scaled_deep = x_train_scaled[:, 2:]
x_valid_scaled_wide = x_valid_scaled[:, :5]
x_valid_scaled_deep = x_valid_scaled[:, 2:]
x_test_scaled_wide = x_test_scaled[:, :5]
x_test_scaled_deep = x_test_scaled[:, 2:]

history = model.fit([x_train_scaled_wide, x_train_scaled_deep],
                    y_train,
                    validation_data=([x_valid_scaled_wide, x_valid_scaled_deep], y_valid),
                    epochs=100,
                    callbacks=callbacks)

plot_learning_curves(history)
res = model.evaluate([x_test_scaled_wide, x_test_scaled_deep], y_test)
print(res)
