from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.learn.python.learn import evaluable
import time

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s\n" % (tf.__version__))

# 我们这里使用的是criteo数据集，X的部分包括13个连续值列和26个类别型值的列
CONTINUOUS_COLUMNS = ["I"+str(i) for i in range(1, 14)] # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C"+str(i) for i in range(1, 27)] # 1-26 inclusive
# 标签是clicked
LABEL_COLUMN = ["clicked"]

# 训练集由 label列 + 连续值列 + 离散值列 构成
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
#TEST_DATA_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

# 特征列就是 连续值列+离散值列
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

# 输出一些信息
print('Feature columns are: ', FEATURE_COLUMNS, '\n')

# 数据示例
sample = [0, 127, 1, 3, 1683, 19, 26, 17, 475, 0, 9, 0, 3, "05db9164", "8947f767", "11c9d79e", "52a787c8", "4cf72387", "fbad5c96", "18671b18", "0b153874", "a73ee510", "ceb10289", "77212bd7", "79507c6b", "7203f04e", "07d13a8f", "2c14c412", "49013ffe", "8efede7f", "bd17c3da", "f6a3e43b", "a458ea53", "35cd95c9", "ad3062eb", "c7dc6720", "3fdb382b", "010f6491", "49d68486"]
print('Columns and data as a dict: ', dict(zip(FEATURE_COLUMNS, sample)), '\n')

BATCH_SIZE = 2000


def generate_input_fn(filename, batch_size=BATCH_SIZE):
    def _input_fn():
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TextLineReader()
        # 只读batch_size行
        key, value = reader.read_up_to(filename_queue, num_records=batch_size)

        # 1个int型的label, 13个连续值, 26个字符串类型
        cont_defaults = [[0] for i in range(1, 14)]
        cate_defaults = [[" "] for i in range(1, 27)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS

        # 第一列数据是label
        record_defaults = label_defaults + cont_defaults + cate_defaults

        # 解析读出的csv数据
        # 我们要手动把数据和header去zip在一起
        columns = tf.decode_csv(value, record_defaults=record_defaults)
        print(columns)

        # 最终是列名到数据张量的映射字典
        all_columns = dict(zip(column_headers, columns))

        # 弹出和保存label标签
        labels = all_columns.pop(LABEL_COLUMN[0])

        # 其余列就是特征
        features = all_columns

        # 类别型的列我们要做一个类似one-hot的扩展操作
        for feature_name in CATEGORICAL_COLUMNS:
            features[feature_name] = tf.expand_dims(features[feature_name], -1)

        return features, labels

    return _input_fn


def create_model_dir(model_type):
    # 返回类似这样的结果 models/model_WIDE_AND_DEEP_1493043407
    return './models/model_' + model_type + '_' + str(int(time.time()))


def get_model(model_type, model_dir):
    """
    指定模型文件夹
    """
    print("Model directory = %s" % model_dir)
    # 对checkpoint去做设定
    runconfig = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=100,)

    m = None

    # 宽模型
    if model_type == 'WIDE':
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns)

    # 深度模型
    if model_type == 'DEEP':
        m = tf.contrib.learn.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[100, 50, 25])

    # 宽度深度模型
    if model_type == 'WIDE_AND_DEEP':
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 70, 50, 25],
            config=runconfig)

    print('estimator built')
    return m


def pred_fn():
    sample = [0, 127, 1, 3, 1683, 19, 26, 17, 475, 0, 9, 0, 3, "05db9164", "8947f767", "11c9d79e", "52a787c8",
              "4cf72387", "fbad5c96", "18671b18", "0b153874", "a73ee510", "ceb10289", "77212bd7", "79507c6b",
              "7203f04e", "07d13a8f", "2c14c412", "49013ffe", "8efede7f", "bd17c3da", "f6a3e43b", "a458ea53",
              "35cd95c9", "ad3062eb", "c7dc6720", "3fdb382b", "010f6491", "49d68486"]
    sample_dict = dict(zip(FEATURE_COLUMNS, sample))
    for feature_name in CATEGORICAL_COLUMNS:
        sample_dict[feature_name] = tf.expand_dims(sample_dict[feature_name], -1)
    for feature_name in CONTINUOUS_COLUMNS:
        sample_dict[feature_name] = tf.constant(sample_dict[feature_name], dtype=tf.int32)
    print(sample_dict)
    return sample_dict


# Sparse base columns.
# C1 = tf.contrib.layers.sparse_column_with_hash_bucket('C1', hash_bucket_size=1000)
# C2 = tf.contrib.layers.sparse_column_with_hash_bucket('C2', hash_bucket_size=1000)
# C3 = tf.contrib.layers.sparse_column_with_hash_bucket('C3', hash_bucket_size=1000)
# ...
# Cn = tf.contrib.layers.sparse_column_with_hash_bucket('Cn', hash_bucket_size=1000)
# wide_columns = [C1, C2, C3, ... , Cn]
wide_columns = []
for name in CATEGORICAL_COLUMNS:
    wide_columns.append(tf.contrib.layers.sparse_column_with_hash_bucket(name, hash_bucket_size=1000))
print('Wide/Sparse columns configured')

# Continuous base columns.
# I1 = tf.contrib.layers.real_valued_column("I1")
# I2 = tf.contrib.layers.real_valued_column("I2")
# I3 = tf.contrib.layers.real_valued_column("I3")
# ...
# In = tf.contrib.layers.real_valued_column("In")
# deep_columns = [I1, I2, I3, ... , In]
deep_columns = []
for name in CONTINUOUS_COLUMNS:
    deep_columns.append(tf.contrib.layers.real_valued_column(name))
print('deep/continuous columns configured')

# No known Transformations. Can add some if desired.
# Examples from other datasets are shown below.
# age_buckets = tf.contrib.layers.bucketized_column(age,
#             boundaries=[ 18, 25, 30, 35, 40, 45, 50, 55, 60, 65 ])
# education_occupation = tf.contrib.layers.crossed_column([education, occupation],
#                                                         hash_bucket_size=int(1e4))
# age_race_occupation = tf.contrib.layers.crossed_column([age_buckets, race, occupation],
#                                                        hash_bucket_size=int(1e6))
# country_occupation = tf.contrib.layers.crossed_column([native_country, occupation],
#                                                       hash_bucket_size=int(1e4))
print('Transformations complete')

# Wide columns and deep columns.
# wide_columns = [gender, race, native_country,
#       education, occupation, workclass,
#       marital_status, relationship,
#       age_buckets, education_occupation,
#       age_race_occupation, country_occupation]
# deep_columns = [
#   tf.contrib.layers.embedding_column(workclass, dimension=8),
#   tf.contrib.layers.embedding_column(education, dimension=8),
#   tf.contrib.layers.embedding_column(marital_status, dimension=8),
#   tf.contrib.layers.embedding_column(gender, dimension=8),
#   tf.contrib.layers.embedding_column(relationship, dimension=8),
#   tf.contrib.layers.embedding_column(race, dimension=8),
#   tf.contrib.layers.embedding_column(native_country, dimension=8),
#   tf.contrib.layers.embedding_column(occupation, dimension=8),
#   age,
#   education_num,
#   capital_gain,
#   capital_loss,
#   hours_per_week,
# ]
# Embeddings for wide columns into deep columns
for col in wide_columns:
    deep_columns.append(tf.contrib.layers.embedding_column(col, dimension=8))
print('wide and deep columns configured')


MODEL_TYPE = 'WIDE_AND_DEEP'
model_dir = create_model_dir(model_type=MODEL_TYPE)
m = get_model(model_type=MODEL_TYPE, model_dir=model_dir)

# 评估
isinstance(m, evaluable.Evaluable)
# 训练文件与测试文件
train_file = '../../DeepFM_NFM_GoogleNet/criteo_data/criteo_train.txt'
eval_file = '../../DeepFM_NFM_GoogleNet/criteo_data/criteo_test.txt'

# This can be found with
# wc -l train.csv
train_sample_size = 2000000
train_steps = train_sample_size / BATCH_SIZE * 20
m.fit(input_fn=generate_input_fn(train_file, BATCH_SIZE), steps=train_steps)
print('fit done')

eval_sample_size = 500000 # this can be found with a 'wc -l eval.csv'
eval_steps = eval_sample_size / BATCH_SIZE
results = m.evaluate(input_fn=generate_input_fn(eval_file), steps=eval_steps)
print('evaluate done')
print('Accuracy: %s' % results['accuracy'])
print(results)
m.predict(input_fn=pred_fn)


