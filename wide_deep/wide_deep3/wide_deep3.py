import tensorflow as tf


class WideDeep(object):
    def __init__(self, vec_dim=None, field_lens=None, dnn_layers=None, wide_lr=None, l1_reg=None, deep_lr=None):
        self.vec_dim = vec_dim
        self.field_lens = field_lens
        self.field_num = len(field_lens)
        self.dnn_layers = dnn_layers
        self.wide_lr = wide_lr
        self.l1_reg = l1_reg
        self.deep_lr = deep_lr

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
        with tf.variable_scope('wide_part'):
            w0 = tf.get_variable(name='bias', shape=[1], dtype=tf.float32)
            linear_w = [tf.get_variable(name='linear_w_%d'%i, shape=[self.field_lens[i]], dtype=tf.float32) for i in range(self.field_num)]
            wide_part = w0 + tf.reduce_sum(
                tf.concat([tf.reduce_sum(tf.multiply(self.x[i], linear_w[i]), axis=1, keep_dims=True) for i in range(self.field_num)], axis=1),
                axis=1, keep_dims=True) # (batch, 1)
        with tf.variable_scope('dnn_part'):
            emb = [tf.get_variable(name='emb_%d'%i, shape=[self.field_lens[i], self.vec_dim], dtype=tf.float32) for i in range(self.field_num)]
            emb_layer = tf.concat([tf.matmul(self.x[i], emb[i]) for i in range(self.field_num)], axis=1) # (batch, F*K)
            x = emb_layer
            in_node = self.field_num * self.vec_dim
            for i in range(len(self.dnn_layers)):
                out_node = self.dnn_layers[i]
                w = tf.get_variable(name='w_%d' % i, shape=[in_node, out_node], dtype=tf.float32)
                b = tf.get_variable(name='b_%d' % i, shape=[out_node], dtype=tf.float32)
                in_node = out_node
                if out_node != 1:
                    x = tf.nn.relu(tf.matmul(x, w) + b)
                else:
                    self.y_logits = wide_part + tf.matmul(x, w) + b

        self.y_hat = tf.nn.sigmoid(self.y_logits)
        self.pred_label = tf.cast(self.y_hat > 0.5, tf.int32)
        self.loss = -tf.reduce_mean(self.y*tf.log(self.y_hat+1e-8) + (1-self.y)*tf.log(1-self.y_hat+1e-8))

        # set optimizer
        self.global_step = tf.train.get_or_create_global_step()

        wide_part_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide_part')
        dnn_part_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dnn_part')

        wide_part_optimizer = tf.train.FtrlOptimizer(learning_rate=self.wide_lr, l1_regularization_strength=self.l1_reg)
        wide_part_op = wide_part_optimizer.minimize(loss=self.loss, global_step=self.global_step, var_list=wide_part_vars)

        dnn_part_optimizer = tf.train.AdamOptimizer(learning_rate=self.deep_lr)
        # set global_step to None so only wide part solver gets passed in the global step;
        # otherwise, all the solvers will increase the global step
        dnn_part_op = dnn_part_optimizer.minimize(loss=self.loss, global_step=None, var_list=dnn_part_vars)

        self.train_op = tf.group(wide_part_op, dnn_part_op)

