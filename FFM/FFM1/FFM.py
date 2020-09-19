import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
# from build_data import transfer_data, get_batch
from FFM.FFM1 import transfer_data, get_batch
from sklearn.model_selection import train_test_split

k = 6  #隐向量个数
f = 3 #field的个数
p = 16 #特征数
learning_rate = 0.1
batch_size = 64
l2_reg_rate = 0.001
feature2field = None
checkpoint_dir = './saver/'
training = True
epoch = 1


#定义权重以及偏置变量，并进行初始化
def createTwoDimensionWeight(input_x_size, field_size, vector_dimension):
    # shape=[p,f,k]
    v = tf.get_variable('v', shape=[input_x_size, field_size, vector_dimension], dtype='float32',
                        initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
    return v


def createOneDimensionWeight(input_x_size):
    w1 = tf.get_variable('w1', shape=[input_x_size, 1], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
    return w1


def createZeroDimensionWeight():
    b = tf.get_variable('bias', shape=[1], initializer=tf.zeros_initializer())  # b形状为1维，初始化为0
    return b


#定义模型，计算模型输出
def inference(X, feature2field, b, w1, v):
    with tf.variable_scope('linear_layer'):
        # shape of [None, 1]
        linear_terms = tf.add(tf.matmul(X, w1), b)  # 线性部分w1*x+b
        print('self.linear_terms:')
        print(linear_terms)
    # 定义交叉项参数v
    with tf.variable_scope('interaction_layer'):
        # v:pxfxk
        field_cross_interaction = tf.constant(0, dtype='float32')
        # 每个特征
        # feature2field={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2}
        for i in range(p):
            for j in range(i + 1, p):
                # print('i:%s,j:%s' % (i, j))
                vifj = v[i, feature2field[j]]  # 找到xj对应的field
                vjfi = v[j, feature2field[i]]  # 找到xi对应的field
                vivj = tf.reduce_sum(tf.multiply(vifj, vjfi))  # 两个矩阵中各自元素相乘
                xixj = tf.multiply(X[:, i], X[:, j])
                field_cross_interaction += tf.multiply(vivj, xixj)
        field_cross_interaction = tf.reshape(field_cross_interaction, (batch_size, 1))  # 转化为64行1列的数组
        print('self.field_cross_interaction:')
        print(field_cross_interaction)
    y_out = tf.add(linear_terms, field_cross_interaction)  # 输出
    print('y_out_prob:')
    print(y_out)

    return y_out


def lodadata(data_path):
    """
    加载数据，划分了训练集以及测试集
    """
    dataset = pd.read_csv(data_path)
    # 将训练数据中的点击与否进行映射，0变为-1
    dataset['click'] = dataset['click'].map(lambda x: -1 if x == 0 else x)
    traindata, testdata = train_test_split(dataset, test_size=0.4, random_state=0)

    # 加载feature2field ，用来对应不同特征的field
    with open('./feature2field.pkl', 'rb') as f:
        feature2field = pickle.load(f)
    fields = ['C1', 'C18', 'C16', 'click']
    # 加载各个field的特征值,
    # fields_dict={'C18': {0: 6, 1: 7, 2: 8, 3: 9},
    # 'click': {1: 16, -1: 17},
    # 'C16': {480: 10, 50: 11, 36: 13, 20: 14, 250: 12, 90: 15},
    # 'C1': {1008: 0, 1010: 1, 1001: 2, 1002: 3, 1005: 4, 1007: 5}}
    fields_dict = {}
    for field in fields:
        with open('./' + field + '.pkl', 'rb') as f:
            fields_dict[field] = pickle.load(f)
    return traindata, testdata, feature2field, fields_dict


def dataset(data, fields_dict):
    """
    构造数据集，对各个特征进行one-hot转换
    """
    all_len = max(fields_dict['click'].values()) + 1  # 18
    cnt = data.shape[0] // batch_size
    for i in range(epoch):
        # 在一个epoch内遍历所有的数据
        for j in range(cnt):
            # 数据转换，将数据转换为one-hot类型
            dataset = get_batch(data, batch_size, j)  # 一次传入traindata的64行，依次向下
            actual_batch_size = len(dataset)
            batch_X = []
            batch_y = []
            for k in range(actual_batch_size):
                sample = dataset.iloc[k, :]  # click      -1
                                             # C1       1005
                                             # C18         0
                                             # C16        50
                array = transfer_data(sample, fields_dict, all_len)  # 每一行的特征都转换为one-hot
                batch_X.append(array[:-2])  # 前面16个是特征
                # 最后一位即为label，[-1]:label=0；[1]:label=1
                batch_y.append(array[-1])  # 最后一个是标签

            batch_X = np.array(batch_X)
            batch_y = np.array(batch_y)

            y_int = batch_y.reshape(len(batch_y), 1)  # 将label转化为64行1列

    return batch_X, y_int, cnt


def save(sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)


def restore(sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)


if __name__ == '__main__':
    # 获取数据
    data_path = './train_sample.csv'
    traindata, testdata, feature2field, fields_dict = lodadata(data_path)
    train_x, train_y, cnt = dataset(traindata, fields_dict)
    test_x, test_y, test_cnt = dataset(testdata, fields_dict)
    # 定义占位符
    X = tf.placeholder(tf.float32, [batch_size, p])
    y = tf.placeholder(tf.float32, [None, 1])
    # 调用函数获得变量
    b = createZeroDimensionWeight()
    w1 = createOneDimensionWeight(p)
    v = createTwoDimensionWeight(p, f, k)
    # 定义损失函数以及优化器
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')
    y_out = inference(X, feature2field, b, w1, v)
    loss = tf.reduce_mean(tf.log(1 + tf.exp(-y * y_out)))  # 损失函数
    l2_norm = tf.reduce_sum(          #正则项
        tf.add(
            tf.multiply(lambda_w, tf.pow(w1, 2)),
            tf.reduce_sum(tf.multiply(lambda_v, tf.pow(v, 2)), axis=[1, 2])
        )
    )
    loss = loss + l2_norm

    global_step = tf.Variable(0, trainable=False)
    # 计算梯度
    opt = tf.train.GradientDescentOptimizer(learning_rate)  # 随机梯度下降算法
    trainable_params = tf.trainable_variables()
    print(trainable_params)  # 查看b，w1,v
    gradients = tf.gradients(loss, trainable_params)  # 求梯度,即文中的g_{i,fj};g_{j,fi}
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    train_op = opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=global_step)
    # 设置GPU
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True  # 当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存

    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #训练
        if training:
            # batch_size data
            # epoch是1
            # loss, train_op = inference(batch_size,p,f,k)
            for i in range(epoch):
                #在一个epoch内遍历所有的数据
                for j in range(cnt):
                    cost, _, step = sess.run([loss, train_op, global_step], feed_dict={
                        X: train_x,
                        y: train_y
                    })
                    if j % 100 == 0:
                        print('After  {%d} training   steps  , and the loss is %s' % (j, cost))
                        save(sess, checkpoint_dir)
        else:
            restore(sess, checkpoint_dir)
            result, cost = sess.run([y_out, loss], feed_dict={X: test_x, y: test_y})
            print(result)
            print(cost)

