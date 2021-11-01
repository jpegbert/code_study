import numpy as np
from sklearn.metrics import roc_auc_score
import progressbar
from LR_FM_CCPM_PNN.util import INPUT_DIM, FIELD_SIZES, FIELD_OFFSETS
from LR_FM_CCPM_PNN.LR import LR
import _pickle as pkl

train_file = '../DeepFM_NFM_GoogleNet/criteo_data/train.txt'
test_file = '../DeepFM_NFM_GoogleNet/criteo_data/test.txt'

input_dim = INPUT_DIM

# 读取数据
#train_data = read_data(train_file)
#test_data = read_data(test_file)
train_data = pkl.load(open('./data/train.pkl', 'rb'))
#train_data = shuffle(train_data)
test_data = pkl.load(open('./data/test.pkl', 'rb'))
# pkl.dump(train_data, open('./data/train.pkl', 'wb'))
# pkl.dump(test_data, open('./data/test.pkl', 'wb'))

# 输出数据信息维度
if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')
print('train data size:', train_data[0].shape)
print('test data size:', test_data[0].shape)

# 训练集与测试集
train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(FIELD_SIZES)

# 超参数设定
min_round = 1
num_round = 200
early_stop_round = 5
# train + val
batch_size = 1024

field_sizes = FIELD_SIZES
field_offsets = FIELD_OFFSETS

# 逻辑回归参数设定
lr_params = {
    'input_dim': input_dim,
    'opt_algo': 'gd',
    'learning_rate': 0.1,
    'l2_weight': 0,
    'random_seed': 0
}
print(lr_params)
model = LR(**lr_params)
print("training LR...")
def train(model):
    history_score = []
    # 执行num_round轮
    for i in range(num_round):
        # 主要的2个op是优化器和损失
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            # 进度条工具
            bar = progressbar.ProgressBar()
            print('[%d]\ttraining...' % i)
            for j in bar(range(int(train_size / batch_size + 1))):
                X_i, y_i = slice(train_data, j * batch_size, batch_size)
                # 训练，run op
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        train_preds = []
        print('[%d]\tevaluating...' % i)
        bar = progressbar.ProgressBar()
        for j in bar(range(int(train_size / 10000 + 1))):
            X_i, _ = slice(train_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            train_preds.extend(preds)
        test_preds = []
        bar = progressbar.ProgressBar()
        for j in bar(range(int(test_size / 10000 + 1))):
            X_i, _ = slice(test_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            test_preds.extend(preds)
        # 把预估的结果和真实结果拿出来计算auc
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        # 输出auc信息
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        # early stopping
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                        -1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
                break

train(model)

