import tensorflow as tf


"""
https://mp.weixin.qq.com/s/eCDYelZKC26MB9rFwJ1opQ
"""


class FNN(object):
    def __init__(self, vec_dim=None, field_lens=None, lr=None, dnn_layers=None, dropout_rate=None, lamda=None):
        self.vec_dim = vec_dim
        self.field_lens = field_lens
        self.field_num = len(field_lens)
        self.lr = lr
        self.dnn_layers = dnn_layers
        self.dropout_rate = dropout_rate
        self.lamda = float(lamda)
        self.l2_reg = tf.contrib.layers.l2_regularizer(self.lamda)

        assert dnn_layers[-1] == 1

        self._build_graph()

    def _build_graph(self):
        self.add_input()
        self.inference()

    def add_input(self):
        self.x = [tf.placeholder(tf.float32, name='input_x_%d'%i) for i in range(self.field_num)]
        self.y = tf.placeholder(tf.float32, shape=[None], name='input_y')
        self.is_train = tf.placeholder(tf.bool)

    def inference(self):
        with tf.variable_scope('fm_part'):
            emb = [tf.get_variable(name='emb_%d'%i, shape=[self.field_lens[i], self.vec_dim], dtype=tf.float32, regularizer=self.l2_reg) for i in range(self.field_num)]
            emb_layer = tf.concat([tf.matmul(self.x[i], emb[i]) for i in range(self.field_num)], axis=1)
        x = emb_layer
        in_node = self.field_num * self.vec_dim
        with tf.variable_scope('dnn_part'):
            for i in range(len(self.dnn_layers)):
                out_node = self.dnn_layers[i]
                w = tf.get_variable(name='w_%d'%i, shape=[in_node, out_node], dtype=tf.float32, regularizer=self.l2_reg)
                b = tf.get_variable(name='b_%d'%i, shape=[out_node], dtype=tf.float32)
                x = tf.matmul(x, w) + b
                if out_node == 1:
                    self.y_logits = x
                else:
                    x = tf.layers.dropout(tf.nn.relu(x), rate=self.dropout_rate, training=self.is_train)
                in_node = out_node


        self.y_hat = tf.nn.sigmoid(self.y_logits)
        self.pred_label = tf.cast(self.y_hat > 0.5, tf.int32)
        self.loss = -tf.reduce_mean(self.y*tf.log(self.y_hat+1e-8) + (1-self.y)*tf.log(1-self.y_hat+1e-8))
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg_variables) > 0:
            self.loss += tf.add_n(reg_variables)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

