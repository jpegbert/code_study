import tensorflow as tf
from LR_FM_CCPM_PNN.Model import Model
from LR_FM_CCPM_PNN.util import init_var_map, get_optimizer
from LR_FM_CCPM_PNN.util import DTYPE


dtype = DTYPE


class LR(Model):
    def __init__(self, input_dim=None, output_dim=1, init_path=None, opt_algo='gd', learning_rate=1e-2, l2_weight=0,
                 random_seed=None):
        Model.__init__(self)
        # 声明参数
        init_vars = [('w', [input_dim, output_dim], 'xavier', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            # 用稀疏的placeholder
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            # init参数
            self.vars = init_var_map(init_vars, init_path)

            w = self.vars['w']
            b = self.vars['b']
            # sigmoid(wx+b)
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = tf.reshape(xw + b, [-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)) + \
                        l2_weight * tf.nn.l2_loss(xw)
            self.optimizer = get_optimizer(opt_algo, learning_rate, self.loss)
            # GPU设定
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            # 初始化图里的参数
            tf.global_variables_initializer().run(session=self.sess)
