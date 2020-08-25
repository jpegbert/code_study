from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from GBDT_LR.GradientBoostingWithLR import GradientBoostingWithLR


def load_data():
    """
    调用sklearn的iris数据集，将多类数据构造成二分类数据，同时切分训练集和测试集
    """
    iris_data = load_iris()
    X = iris_data['data']
    y = iris_data["target"] == 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()

gbdt_lr = GradientBoostingWithLR()
gbdt_lr_model = gbdt_lr.gbdt_lr_train(X_train, y_train, X_test)
gbdt_lr.model_assessment(gbdt_lr.gbdt_model, X_test, y_test)
gbdt_lr.gbdt_lr_pred(gbdt_lr_model, X_test, y_test)
