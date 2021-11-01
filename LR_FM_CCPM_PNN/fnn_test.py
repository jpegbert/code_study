import numpy as np
from sklearn.metrics import roc_auc_score
import progressbar
import _pickle as pkl
from LR_FM_CCPM_PNN.util import INPUT_DIM, FIELD_SIZES, FIELD_OFFSETS, shuffle, split_data
from LR_FM_CCPM_PNN.FNN import FNN

train_file = '../DeepFM_NFM_GoogleNet/criteo_data/train.txt'
test_file = '../DeepFM_NFM_GoogleNet/criteo_data/test.txt'

input_dim = INPUT_DIM
train_data = pkl.load(open('./data/train.pkl', 'rb'))
train_data = shuffle(train_data)
test_data = pkl.load(open('./data/test.pkl', 'rb'))

if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')
print('train data size:', train_data[0].shape)
print('test data size:', test_data[0].shape)

train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(FIELD_SIZES)

min_round = 1
num_round = 200
early_stop_round = 5
batch_size = 1024

field_sizes = FIELD_SIZES
field_offsets = FIELD_OFFSETS

train_data = split_data(train_data)
test_data = split_data(test_data)
tmp = []
for x in field_sizes:
    if x > 0:
        tmp.append(x)
field_sizes = tmp
print('remove empty fields', field_sizes)

fnn_params = {
    'field_sizes': field_sizes,
    'embed_size': 10,
    'layer_sizes': [500, 1],
    'layer_acts': ['relu', None],
    'drop_out': [0, 0],
    'opt_algo': 'gd',
    'learning_rate': 0.1,
    'embed_l2': 0,
    'layer_l2': [0, 0],
    'random_seed': 0
}
print(fnn_params)
model = FNN(**fnn_params)


def train(model):
    history_score = []
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            bar = progressbar.ProgressBar()
            print('[%d]\ttraining...' % i)
            for j in bar(range(int(train_size / batch_size + 1))):
                X_i, y_i = slice(train_data, j * batch_size, batch_size)
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
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                -1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
                break


train(model)
