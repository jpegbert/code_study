from LR_FM_CCPM_PNN.util import DTYPE


# 定义基类模型
dtype = DTYPE
class Model:
    def __init__(self):
        self.sess = None
        self.X = None
        self.y = None
        self.layer_keeps = None
        self.vars = None
        self.keep_prob_train = None
        self.keep_prob_test = None

    # run model
    def run(self, fetches, X=None, y=None, mode='train'):
            # 通过feed_dict传入数据
            feed_dict = {}
            if type(self.X) is list:
                for i in range(len(X)):
                    feed_dict[self.X[i]] = X[i]
            else:
                feed_dict[self.X] = X
            if y is not None:
                feed_dict[self.y] = y
            if self.layer_keeps is not None:
                if mode == 'train':
                    feed_dict[self.layer_keeps] = self.keep_prob_train
                elif mode == 'test':
                    feed_dict[self.layer_keeps] = self.keep_prob_test
            #通过session.run去执行op
            return self.sess.run(fetches, feed_dict)

    # 模型参数持久化
    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print('model dumped at', model_path)
