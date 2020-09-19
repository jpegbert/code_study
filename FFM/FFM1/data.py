import pandas as pd
import numpy as np
from collections import Counter
import pickle

data_path = './train_sample.csv'
dataset = pd.read_csv(data_path)#读取训练集数据
dataset['click'] = dataset['click'].map(lambda x: -1 if x == 0 else x)#将训练数据中的点击与否进行映射，0变为-1
click = set()
C1 = set()
C16 = set()
C18 = set()

# build category data
# for data in dataset:
data = dataset.copy() #复制训练集

click_v = set(data['click'].values) #以列表返回中所有的值，删除重复的数据，变为一个集合{-1,1}
click = click | click_v #并集

C1_v = set(data['C1'].values)
C1 = C1 | C1_v #{1008, 1010, 1001, 1002, 1005, 1007}

C16_v = set(data['C16'].values)
C16 = C16 | C16_v #{480, 50, 250, 36, 20, 90}

C18_v = set(data['C18'].values)
C18 = C18 | C18_v #{0, 1, 2, 3}

#类别数据的fields
category_encoding_fields = ['C1', 'C18', 'C16']

feature2field = {}
field_index = 0
ind = 0
for field in category_encoding_fields:
    field_dict = {}
    field_sets = eval(field)#返回表达式的值

    for value in list(field_sets):

        field_dict[value] = ind    #对于不同的值赋予不同的索引
        feature2field[ind] = field_index#在字典中给索引不同的值，从0开始
        ind += 1
    field_index += 1
    print(field_dict)
    print(feature2field)
    with open('./' + field + '.pkl', 'wb') as f:
        pickle.dump(field_dict, f)


#至此，C1.pkl数据为{1008: 0, 1010: 1, 1001: 2, 1002: 3, 1005: 4, 1007: 5}，1008在feature2field对应的键为0.。。
  #   C18.pkl数据为{0: 6, 1: 7, 2: 8, 3: 9}
  #   C16.pkl数据为{480: 10, 50: 11, 20: 14, 36: 13, 250: 12, 90: 15}
  #   click.pkl数据为{1: 16, -1: 17}
  #以上四个文件为不同特征值对应的索引值
  #   feature2field{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2}
  #不同索引值对应的field

click_dict = {}
click_sets = click #{1，-1}
for value in list(click_sets):
    click_dict[value] = ind
    ind += 1

#click_dict={1: 16, -1: 17}
with open('./' + 'click' + '.pkl', 'wb') as f:
    pickle.dump(click_dict, f)

with open('./feature2field.pkl', 'wb') as f:
    pickle.dump(feature2field, f)


# 将所有的特征转换为one-hot类型
def transfer_data(sample, fields_dict, array_length):
    array = np.zeros([array_length])
    for field in fields_dict:
        # get index of array
        if field == 'click':
            field_value = sample[field]
            ind = fields_dict[field][field_value]#得到索引
            if ind == (array_length - 1):
                array[ind] = -1
            else:
                array[ind + 1] = 1
        else:
            field_value = sample[field]
            ind = fields_dict[field][field_value]
            array[ind] = 1
    return array


def get_batch(x, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < x.shape[0] else x.shape[0]
    return x.iloc[start:end, :] #选取x中的start到end行
