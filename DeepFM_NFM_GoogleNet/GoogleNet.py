from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.layers import Embedding, Merge
from keras.callbacks import ModelCheckpoint
import keras
from keras.preprocessing import image
import numpy as np
import sys, os, re
from keras.applications.inception_v3 import InceptionV3, preprocess_input


# 定义VGG卷积神经网络
def GoogleInceptionV3():
    model = InceptionV3(weights='imagenet', include_top=False)
    model.trainable = False

    return model


# 加载field和feature信息
def load_field_feature_meta(field_info_file):
    field_feature_dic = {}
    for line in open(field_info_file):
        contents = line.strip().split("\t")
        field_id = int(contents[1])
        feature_count = int(contents[4])
        field_feature_dic[field_id] = feature_count
    return field_feature_dic


# CTR特征做embedding
def CTR_embedding(field_feature_dic):
    emd = []
    for field_id in range(len(field_feature_dic)):
        # 先把离散特征embedding到稠密的层
        tmp_model = Sequential()
        # 留一个位置给rare
        input_dims = field_feature_dic[field_id] + 1
        if input_dims > 16:
            dense_dim = 16
        else:
            dense_dim = input_dims
        tmp_model.add(Dense(dense_dim, input_dim=input_dims))
        emd.append(tmp_model)
    return emd


# 总的网络结构
def full_network(field_feature_dic):
    print
    "GoogleNet model loading"
    googleNet_model = GoogleInceptionV3()
    image_model = Flatten()(googleNet_model.outputs)
    image_model = Dense(256)(image_model)

    print("GoogleNet model loaded")
    print("initialize embedding model")
    print("loading fields info...")
    emd = CTR_embedding(field_feature_dic)
    print("embedding model done!")
    print("initialize full model...")
    full_model = Sequential()
    full_input = [image_model] + emd
    full_model.add(Merge(full_input, mode='concat'))
    # 批规范化
    full_model.add(keras.layers.normalization.BatchNormalization())
    # 全连接层
    full_model.add(Dense(128))
    full_model.add(Dropout(0.4))
    full_model.add(Activation('relu'))
    # 全连接层
    full_model.add(Dense(128))
    full_model.add(Dropout(0.4))
    # 最后的分类
    full_model.add(Dense(1))
    full_model.add(Activation('sigmoid'))
    # 编译整个模型
    full_model.compile(loss='binary_crossentropy',
                       optimizer='adadelta',
                       metrics=['binary_accuracy', 'fmeasure'])
    # 输出模型每一层的信息
    full_model.summary()
    return full_model


# 图像预处理
def vgg_image_preoprocessing(image):
    img = image.load_img(image, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# CTR特征预处理
def ctr_feature_preprocessing(field_feature_string):
    contents = field_feature_string.strip().split(" ")
    feature_dic = {}
    for content in contents:
        field_id, feature_id, num = content.split(":")
        feature_dic[int(field_id)] = int(feature_id)
    return feature_dic


# 产出用于训练的一个batch数据
def generate_batch_from_file(in_f, field_feature_dic, batch_num, skip_lines=0):
    # 初始化x和y
    img_x = []
    x = []
    for field_id in range(len(field_feature_dic)):
        x.append(np.zeros((batch_num, int(field_feature_dic[field_id]) + 1)))
    y = [0.0] * batch_num
    round_num = 1

    while True:
        line_count = 0
        skips = 0
        f = open(in_f)
        for line in f:
            if (skip_lines > 0 and round_num == 1):
                if skips < skip_lines:
                    skips += 1
                    continue
            if (line_count + 1) % batch_num == 0:
                contents = line.strip().split("\t")
                img_name = "images/" + re.sub(r'.jpg.*', '.jpg', contents[1].split("/")[-1])
                if not os.path.isfile(img_name):
                    continue
                # 初始化最后一个样本
                try:
                    img_input = vgg_image_preoprocessing(img_name)
                except:
                    continue
                # 图片特征填充
                img_x.append(img_input)
                # ctr特征填充
                ctr_feature_dic = ctr_feature_preprocessing(contents[2])
                for field_id in ctr_feature_dic:
                    x[field_id][line_count][ctr_feature_dic[field_id]] = 1.0
                # 填充y值
                y[line_count] = int(contents[0])
                # print "shape is", np.array(img_x).shape
                yield ([np.array(img_x)] + x, y)

                img_x = []
                x = []
                for field_id in range(len(field_feature_dic)):
                    x.append(np.zeros((batch_num, int(field_feature_dic[field_id]) + 1)))
                y = [0.0] * batch_num
                line_count = 0
            else:
                contents = line.strip().split("\t")
                img_name = "images/" + re.sub(r'.jpg.*', '.jpg', contents[1].split("/")[-1])
                if not os.path.isfile(img_name):
                    continue
                try:
                    img_input = vgg_image_preoprocessing(img_name)
                except:
                    continue
                # 图片特征填充
                img_x.append(img_input)
                # ctr特征填充
                ctr_feature_dic = ctr_feature_preprocessing(contents[2])
                for field_id in ctr_feature_dic:
                    x[field_id][line_count][ctr_feature_dic[field_id]] = 1.0
                # 填充y值
                y[line_count] = int(contents[0])
                line_count += 1
        f.close()
        round_num += 1


def train_network(skip_lines, batch_num, field_info_file, data_file, weight_file):
    print
    "starting train whole network...\n"
    field_feature_dic = load_field_feature_meta(field_info_file)
    full_model = full_network(field_feature_dic)
    if os.path.isfile(weight_file):
        full_model.load_weights(weight_file)
    checkpointer = ModelCheckpoint(filepath=weight_file, save_best_only=False, verbose=1, period=3)
    full_model.fit_generator(generate_batch_from_file(data_file, field_feature_dic, batch_num, skip_lines),
                             samples_per_epoch=1280, nb_epoch=100000, callbacks=[checkpointer])


if __name__ == '__main__':
    skip_lines = sys.argv[1]
    batch_num = sys.argv[2]
    field_info_file = sys.argv[3]
    data_file = sys.argv[4]
    weight_file = sys.argv[5]
    train_network(int(skip_lines), int(batch_num), field_info_file, data_file, weight_file)

