import tensorflow as tf
import numpy as np


"""
采用tensorflow实现xDeepFM
数据集：movieLens100k
"""


class XDeepFM(object):
    def __init__(self, vec_dim=None, field_lens=None, cin_layer_num=None, dnn_layers=None, lr=None, dropout_rate=None):
        self.vec_dim = vec_dim # embedding vector维度
        self.field_lens = field_lens # list结构，其中每个元素代表对应Field有多少取值。例如gender有两个取值，那么其对应的元素为2
        self.field_num = len(field_lens)
        self.feat_num = np.sum(field_lens)
        self.cin_layer_num = cin_layer_num # cin network 层数
        self.dnn_layers = dnn_layers # list结构，其中每个元素对应DNN部分节点数目
        self.lr = lr # 学习率
        self.dropout_rate = dropout_rate

        self._build_graph()

    def _build_graph(self):
        self.add_input()
        self.inference()

    def add_input(self):
        self.index = tf.placeholder(tf.int32, shape=[None, self.field_num], name='feat_index') # (batch, m)
        self.x = tf.placeholder(tf.float32, shape=[None, self.field_num], name='feat_value') # (batch, m)
        self.y = tf.placeholder(tf.float32, shape=[None], name='input_y')
        self.is_train = tf.placeholder(tf.bool)

    def cin_layer(self, x0_tensor, xk_tensor, xk_field_num, feature_map_num, name):
        with tf.variable_scope(name):
            x0 = tf.split(value=x0_tensor, num_or_size_splits=self.vec_dim*[1], axis=2)
            xk = tf.split(value=xk_tensor, num_or_size_splits=self.vec_dim*[1], axis=2)
            z_tensor = tf.matmul(x0, xk, transpose_b=True) # (D, batch, m, H_k)
            z_tensor = tf.reshape(z_tensor, shape=[self.vec_dim, -1, self.field_num*xk_field_num]) # (D, batch, m*H_k)
            z_tensor = tf.transpose(z_tensor, perm=[1, 0, 2]) # (batch, D, m*H_k)

            filters = tf.get_variable(name='filters', shape=[1, self.field_num*xk_field_num, feature_map_num], dtype=tf.float32)
            xk_1 = tf.nn.conv1d(z_tensor, filters=filters, stride=1, padding='VALID') # (batch, D, feature_map_num)
            xk_1 = tf.transpose(xk_1, perm=[0, 2, 1]) # (batch, feature_map_num, D)
            return xk_1

    def inference(self):
        with tf.variable_scope('first_order_part'):
            first_ord_w = tf.get_variable(name='first_ord_w', shape=[self.feat_num, 1], dtype=tf.float32)
            first_order = tf.nn.embedding_lookup(first_ord_w, self.index) # (batch, m, 1)
            first_order = tf.reduce_sum(tf.multiply(first_order, tf.expand_dims(self.x, axis=2)), axis=2) # (batch, m)

        with tf.variable_scope('emb_part'):
            embed_matrix = tf.get_variable(name='second_ord_v', shape=[self.feat_num, self.vec_dim], dtype=tf.float32)
            embed_v = tf.nn.embedding_lookup(embed_matrix, self.index) # (batch, m, D)

            embed_x = tf.multiply(tf.expand_dims(self.x, axis=2), embed_v)  # (batch, m, D)
            embed_x = tf.layers.dropout(embed_x, rate=self.dropout_rate, training=self.is_train)  # (batch, m, D)
            node_num = self.field_num * self.vec_dim
            embed_x = tf.reshape(embed_x, shape=[-1, node_num]) # (batch, node_num)

        with tf.variable_scope('cin_part'):
            cross_tensors = []
            x0_tensor = tf.reshape(embed_x, shape=[-1, self.field_num, self.vec_dim]) # (batch, m, D)
            cross_tensors.append(x0_tensor)
            field_nums = []
            field_nums.append(int(self.field_num))
            for i, layer_num in enumerate(self.cin_layer_num):
                xk_tensor = self.cin_layer(x0_tensor, cross_tensors[-1], field_nums[-1], layer_num, 'cin_layer_%d'%i)
                cross_tensors.append(xk_tensor)
                field_nums.append(layer_num)
            p_vec = [tf.reduce_sum(x, axis=2) for x in cross_tensors]
            cin = tf.concat(p_vec, axis=1)
            cin_lens = np.sum(field_nums)

        with tf.variable_scope('dnn_part'):
            dnn = embed_x
            in_num = node_num
            for i in range(len(self.dnn_layers)):
                out_num = self.dnn_layers[i]
                w = tf.get_variable(name='w_%d'%i, shape=[in_num, out_num], dtype=tf.float32)
                b = tf.get_variable(name='b_%d'%i, shape=[out_num], dtype=tf.float32)
                dnn = tf.matmul(dnn, w) + b
                dnn = tf.layers.dropout(tf.nn.relu(dnn), rate=self.dropout_rate, training=self.is_train)
                in_num = out_num

        with tf.variable_scope('output_part'):
            output = tf.concat([first_order, cin, dnn], axis=1)
            global_w = tf.get_variable(name='global_w', shape=[self.field_num+cin_lens+in_num, 1], dtype=tf.float32)
            global_b = tf.get_variable(name='global_b', shape=[1], dtype=tf.float32)
            self.y_logits = tf.matmul(output, global_w) + global_b

        self.y_hat = tf.nn.sigmoid(self.y_logits)
        self.pred_label = tf.cast(self.y_hat > 0.5, tf.int32)
        self.loss = -tf.reduce_mean(self.y*tf.log(self.y_hat+1e-8) + (1-self.y)*tf.log(1-self.y_hat+1e-8))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
