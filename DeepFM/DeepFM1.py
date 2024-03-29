import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import DeepFM
from deepctr.utils import SingleFeat


if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')
    #拆分稀疏和稠密特征
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']
    # 1.类别特征的编码与稠密特征做归一化
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    # 2.统计稀疏特征类别特征个数，记录稠密特征类目
    sparse_feature_list = [SingleFeat(feat, data[feat].nunique()) for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0,) for feat in dense_features]
    # 3.生成模型输入特征
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
                        [train[feat.name].values for feat in dense_feature_list]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
                       [test[feat.name].values for feat in dense_feature_list]
    # 4.定义模型、预测、评估模型
    model = DeepFM({"sparse": sparse_feature_list, "dense": dense_feature_list}, task='binary')
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
    history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

