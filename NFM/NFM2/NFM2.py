import tensorflow as tf


class NFM(object):
    def __init__(self, vec_dim=None, field_lens=None, dnn_layers=None, lr=None, dropout_rate=None):
        self.vec_dim = vec_dim
        self.field_lens = field_lens
        self.field_num = len(field_lens)
        self.dnn_layers = dnn_layers
        self.lr = lr
        self.dropout_rate = dropout_rate
        assert isinstance(dnn_layers, list) and dnn_layers[-1] == 1
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
            emb_layer = tf.concat([tf.matmul(self.x[i], emb[i]) for i in range(self.field_num)], axis=1) # (batch, F*K)
            emb_layer = tf.reshape(emb_layer, shape=(-1, self.field_num, self.vec_dim)) # (batch, F, K)
        with tf.variable_scope('bi_interaction_part'):
            sum_square_part = tf.square(tf.reduce_sum(emb_layer, axis=1)) # (batch, K)
            square_sum_part = tf.reduce_sum(tf.square(emb_layer), axis=1) # (batch, K)
            nfm = 0.5 * (sum_square_part - square_sum_part)
            nfm = tf.layers.batch_normalization(nfm, training=self.is_train, name='bi_interaction_bn')
            nfm = tf.layers.dropout(nfm, rate=self.dropout_rate, training=self.is_train)
        with tf.variable_scope('dnn_part'):
            in_node = self.vec_dim
            for i in range(len(self.dnn_layers)-1):
                out_node = self.dnn_layers[i]
                w = tf.get_variable(name='w_%d'%i, shape=[in_node, out_node], dtype=tf.float32)
                b = tf.get_variable(name='b_%d'%i, shape=[out_node], dtype=tf.float32)
                in_node = out_node
                nfm = tf.matmul(nfm, w) + b
                nfm = tf.layers.batch_normalization(nfm, training=self.is_train, name='bn_%d'%i)
                nfm = tf.nn.relu(nfm)
                nfm = tf.layers.dropout(nfm, rate=self.dropout_rate, training=self.is_train)
            h = tf.get_variable(name='h', shape=[in_node, 1], dtype=tf.float32)
            nfm = tf.matmul(nfm, h) # (batch, 1)

        self.y_logits = linear_part + nfm
        self.y_hat = tf.nn.sigmoid(self.y_logits)
        self.pred_label = tf.cast(self.y_hat > 0.5, tf.int32)
        self.loss = -tf.reduce_mean(self.y*tf.log(self.y_hat+1e-8) + (1-self.y)*tf.log(1-self.y_hat+1e-8))
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg_variables) > 0:
            self.loss += tf.add_n(reg_variables)
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
