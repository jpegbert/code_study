import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder


class GradientBoostingWithLR(object):
    def __init__(self):
        self.gbdt_model = None
        self.lr_model = None
        self.gbdt_encoder = None
        self.X_train_leafs = None
        self.X_test_leafs = None
        self.X_trains = None

    def gbdt_train(self, X_train, y_train):
        """
        定义GBDT模型
        :param X_train:
        :param y_train:
        :return:
        """
        gbdt_model = GradientBoostingClassifier(n_estimators=10, max_depth=6, verbose=0, max_features=0.5)
        # 训练模型
        gbdt_model.fit(X_train, y_train)
        return gbdt_model

    def lr_train(self, X_train, y_train):
        """
        训练LR模型
        :param X_train:
        :param y_train:
        :param X_test:
        :return:
        """
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        return lr_model

    def gbdt_lr_train(self, X_train, y_train, X_test):
        """
        训练GBDT+LR模型
        :param X_train:
        :param y_train:
        :param X_test:
        :return:
        """
        self.gbdt_model = self.gbdt_train(X_train, y_train)
        # 使用GBDT的apply方法对原有的特征进行编码
        self.X_train_leafs = self.gbdt_model.apply(X_train)[:, :, 0]
        # 对特征进行one-hot编码
        self.gbdt_encoder = OneHotEncoder(categories='auto')
        self.gbdt_encoder.fit((self.X_train_leafs))
        self.X_trains = self.gbdt_encoder.fit_transform(self.X_train_leafs)

        # 采用LR进行训练
        self.lr_model = self.lr_train(self.X_trains, y_train)
        return self.lr_model

    def gbdt_lr_pred(self, model, X_test, y_test):
        """
        预测及AUC评估
        :param model:
        :param X_test:
        :param y_test:
        :return:
        """
        self.X_test_leafs = self.gbdt_model.apply(X_test)[:, :, 0]
        (train_rows, cols) = self.X_train_leafs.shape
        X_train_all = self.gbdt_encoder.fit_transform(np.concatenate((self.X_train_leafs, self.X_test_leafs)))
        y_pred = model.predict_proba(X_train_all[train_rows:])[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        print("GBDT + LR AUC score: %.5f" % auc_score)
        return auc_score

    def model_assessment(self, model, X_test, y_test, model_name="GBDT"):
        """
        模型评估
        :param model:
        :param X_test:
        :param y_test:
        :param model_name:
        :return:
        """
        y_pred = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        print("%s AUC score: %.5f" % (model_name, auc_score))
        return auc_score


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
def load_data():
    """
    调用sklearn的iris数据集，将多类数据构造成二分类数据，同时切分训练集和测试集
    """
    iris_data = load_iris()
    X = iris_data['data']
    y = iris_data["target"] == 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test


# X_train, X_test, y_train, y_test = load_data()
#
# gbdt_lr = GradientBoostingWithLR()
# gbdt_lr_model = gbdt_lr.gbdt_lr_train(X_train, y_train, X_test)
















