import tensorflow as tf
import numpy as np
from LR_FM_CCPM_PNN.Model import Model
from LR_FM_CCPM_PNN.util import DTYPE
from LR_FM_CCPM_PNN.util import init_var_map, get_optimizer, activate, max_pool_4d

dtype = DTYPE


class CCPM(Model):
    def __init__(self, field_sizes=None, embed_size=10, filter_sizes=None, layer_acts=None, drop_out=None,
                 init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        init_vars.append(('f1', [embed_size, filter_sizes[0], 1, 2], 'xavier', dtype))
        init_vars.append(('f2', [embed_size, filter_sizes[1], 2, 2], 'xavier', dtype))
        init_vars.append(('w1', [2 * 3 * embed_size, 1], 'xavier', dtype))
        init_vars.append(('b1', [1], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            l = xw

            l = tf.transpose(tf.reshape(l, [-1, num_inputs, embed_size, 1]), [0, 2, 1, 3])
            f1 = self.vars['f1']
            l = tf.nn.conv2d(l, f1, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]),
                    int(num_inputs / 2)),
                [0, 1, 3, 2])
            f2 = self.vars['f2']
            l = tf.nn.conv2d(l, f2, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]), 3),
                [0, 1, 3, 2])
            l = tf.nn.dropout(
                activate(
                    tf.reshape(l, [-1, embed_size * 3 * 2]),
                    layer_acts[0]),
                self.layer_keeps[0])
            w1 = self.vars['w1']
            b1 = self.vars['b1']
            l = tf.matmul(l, w1) + b1

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)
