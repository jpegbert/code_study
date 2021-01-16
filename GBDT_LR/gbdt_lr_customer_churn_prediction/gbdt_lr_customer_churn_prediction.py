from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


"""
GBDT+LR模型 电信客户流失预测
"""


class ChurnPredWithGBDTAndLR:
    def __init__(self):
        self.file = "new_churn.csv"
        self.data = self.load_data()
        self.train, self.test = self.split()

    # 加载数据
    def load_data(self):
        return pd.read_csv(self.file)

    # 拆分数据集
    def split(self):
        train, test = train_test_split(self.data, test_size=0.1, random_state=40)
        return train, test

    # 模型训练
    def train_model(self):
        lable = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.train.columns if x not in [lable, ID]]
        x_train = self.train[x_columns]
        y_train = self.train[lable]

        # 创建gbdt模型 并训练
        gbdt = GradientBoostingClassifier()
        gbdt.fit(x_train, y_train)

        # 模型融合
        gbdt_lr = LogisticRegression()
        enc = OneHotEncoder()
        print(gbdt.apply(x_train).shape)
        print(gbdt.apply(x_train).reshape(-1, 100).shape)

        # 100为n_estimators，迭代次数
        enc.fit(gbdt.apply(x_train).reshape(-1, 100))
        gbdt_lr.fit(enc.transform(gbdt.apply(x_train).reshape(-1, 100)), y_train)

        return enc, gbdt, gbdt_lr

    # 效果评估
    def evaluate(self, enc, gbdt, gbdt_lr):
        lable = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.test.columns if x not in [lable, ID]]
        x_test = self.test[x_columns]
        y_test = self.test[lable]

        # gbdt 模型效果评估
        gbdt_y_pred = gbdt.predict_proba(x_test)
        new_gbdt_y_pred = list()
        for y in gbdt_y_pred:
            # y[0] 表示样本label=0的概率 y[1]表示样本label=1的概率
            new_gbdt_y_pred.append(1 if y[1] > 0.5 else 0)
        print("GBDT-MSE: %.4f" % mean_squared_error(y_test, new_gbdt_y_pred))
        print("GBDT-Accuracy : %.4g" % metrics.accuracy_score(y_test.values, new_gbdt_y_pred))
        print("GBDT-AUC Score : %.4g" % metrics.roc_auc_score(y_test.values, new_gbdt_y_pred))

        gbdt_lr_y_pred = gbdt_lr.predict_proba(enc.transform(gbdt.apply(x_test).reshape(-1, 100)))
        new_gbdt_lr_y_pred = list()
        for y in gbdt_lr_y_pred:
            # y[0] 表示样本label=0的概率 y[1]表示样本label=1的概率
            new_gbdt_lr_y_pred.append(1 if y[1] > 0.5 else 0)
        print("GBDT_LR-MSE: %.4f" % mean_squared_error(y_test, new_gbdt_lr_y_pred))
        print("GBDT_LR-Accuracy : %.4g" % metrics.accuracy_score(y_test.values, new_gbdt_lr_y_pred))
        print("GBDT_LR-AUC Score : %.4g" % metrics.roc_auc_score(y_test.values, new_gbdt_lr_y_pred))


if __name__ == "__main__":
    pred = ChurnPredWithGBDTAndLR()
    enc, gbdt, gbdt_lr = pred.train_model()
    pred.evaluate(enc, gbdt, gbdt_lr)
