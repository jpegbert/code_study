import tensorflow as tf


"""
vec_dim ：代表embedding vector维度。
field_lens ：list结构，其中每个元素代表对应Field有多少取值。例如gender有两个取值，那么其对应的元素为2。
attention_factor：与论文中含义一致。
lr ：学习率。
lamda ：L2正则化强度。
"""


class AFM(object):
    def __init__(self, vec_dim=None, field_lens=None, attention_factor=None, lr=None, dropout_rate=None, lamda=None):
        self.vec_dim = vec_dim
        self.field_lens = field_lens
        self.field_num = len(field_lens)
        self.attention_factor = attention_factor
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.lamda = float(lamda)

        self.l2_reg = tf.contrib.layers.l2_regularizer(self.lamda)

        self._build_graph()

    def _build_graph(self):
        self.add_input()
        self.inference()

    def add_input(self):
        self.x = [tf.placeholder(tf.float32, name='input_x_%d'%i) for i in range(self.field_num)]
        self.y = tf.placeholder(tf.float32, shape=[None], name='input_y')
        self.is_train = tf.placeholder(tf.bool)

    def inference(self):
        with tf.variable_scope('linear_part'):
            w0 = tf.get_variable(name='bias', shape=[1], dtype=tf.float32)
            linear_w = [tf.get_variable(name='linear_w_%d'%i, shape=[self.field_lens[i]], dtype=tf.float32) for i in range(self.field_num)]
            linear_part = w0 + tf.reduce_sum(
                tf.concat([tf.reduce_sum(tf.multiply(self.x[i], linear_w[i]), axis=1, keep_dims=True) for i in range(self.field_num)], axis=1),
                axis=1, keep_dims=True) # (batch, 1)
        with tf.variable_scope('emb_part'):
            emb = [tf.get_variable(name='emb_%d'%i, shape=[self.field_lens[i], self.vec_dim], dtype=tf.float32) for i in range(self.field_num)]
            emb_layer = tf.stack([tf.matmul(self.x[i], emb[i]) for i in range(self.field_num)], axis=1) # (batch, F, K)

        with tf.variable_scope('pair_wise_interaction_part'):
            pi_embedding = []
            for i in range(self.field_num):
                for j in range(i+1, self.field_num):
                    pi_embedding.append(tf.multiply(emb_layer[:,i,:], emb_layer[:,j,:])) # [(batch, K), ....]
            pi_embedding = tf.stack(pi_embedding, axis=1) # (batch, F*(F-1)/2, K)
            cross_num = self.field_num * (self.field_num - 1) / 2

        with tf.variable_scope('attention_network'):
            # (K, t)
            att_w = tf.get_variable(name='attention_w', shape=[self.vec_dim, self.attention_factor], dtype=tf.float32, regularizer=self.l2_reg) # reg weight
            att_b = tf.get_variable(name='attention_b', shape=[self.attention_factor], dtype=tf.float32)
            att_h = tf.get_variable(name='attention_h', shape=[self.attention_factor, 1], dtype=tf.float32) # (t, 1)
            # wx+b
            attention = tf.matmul(tf.reshape(pi_embedding, shape=(-1, self.vec_dim)), att_w) + att_b # (batch*F*(F-1)/2, t)
            # relu(wx+b)
            attention = tf.nn.relu(attention)
            # h^T(relu(wx+b))
            attention = tf.reshape(tf.matmul(attention, att_h), shape=(-1, cross_num)) # (batch, F*(F-1)/2)
            # softmax
            attention_score = tf.nn.softmax(attention) # (batch, F*(F-1)/2)
            attention_score = tf.reshape(attention_score, shape=(-1, cross_num, 1)) # (batch, F*(F-1)/2, 1)

        with tf.variable_scope('prediction_score'):
            weight_sum = tf.multiply(pi_embedding, attention_score) # (batch, F*(F-1)/2, K)
            weight_sum = tf.reduce_sum(weight_sum, axis=1) # (batch, K)
            weight_sum = tf.layers.dropout(weight_sum, rate=self.dropout_rate, training=self.is_train)
            p = tf.get_variable(name='p', shape=[self.vec_dim, 1], dtype=tf.float32)
            pred_score = tf.matmul(weight_sum, p) # (batch, 1)

        self.y_logits = linear_part + pred_score
        self.y_hat = tf.nn.sigmoid(self.y_logits)
        self.pred_label = tf.cast(self.y_hat > 0.5, tf.int32)
        self.loss = -tf.reduce_mean(self.y*tf.log(self.y_hat+1e-8) + (1-self.y)*tf.log(1-self.y_hat+1e-8))
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg_variables) > 0:
            self.loss += tf.add_n(reg_variables)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

