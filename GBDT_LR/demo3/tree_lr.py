import numpy as np
import random
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost.sklearn import XGBClassifier


np.random.seed(10)
X, Y = make_classification(n_samples=1000, n_features=30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=233, test_size=0.5)
X_train, X_train_lr, Y_train, Y_train_lr = train_test_split(X_train, Y_train, random_state=233, test_size=0.2)
print(X_train.shape, X_train_lr.shape, Y_train.shape, Y_train_lr.shape) # (400, 30) (100, 30) (400,) (100,)


def RandomForestLR():
    """
    RandomForest + LogisticRegression
    """
    RF = RandomForestClassifier(n_estimators=100, max_depth=4)
    RF.fit(X_train, Y_train)
    OHE = OneHotEncoder()
    OHE.fit(RF.apply(X_train))
    LR = LogisticRegression()
    LR.fit(OHE.transform(RF.apply(X_train_lr)), Y_train_lr)
    Y_pred = LR.predict_proba(OHE.transform(RF.apply(X_test)))[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('RandomForest + LogisticRegression: ', auc)
    return fpr, tpr


def XGBoostLR():
    """
    Xgboost + LogisticRegression
    """
    XGB = xgb.XGBClassifier(nthread=4, learning_rate=0.08, n_estimators=100, colsample_bytree=0.5)
    XGB.fit(X_train, Y_train)
    OHE = OneHotEncoder()
    OHE.fit(XGB.apply(X_train))
    LR = LogisticRegression(n_jobs=4, C=0.1, penalty='l2')
    LR.fit(OHE.transform(XGB.apply(X_train_lr)), Y_train_lr)
    Y_pred = LR.predict_proba(OHE.transform(XGB.apply(X_test)))[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('XGBoost + LogisticRegression: ', auc)
    return fpr, tpr


def GBDTLR():
    """
    GradientBoosting + LogisticRegression
    """
    GBDT = GradientBoostingClassifier(n_estimators=10)
    GBDT.fit(X_train, Y_train)
    OHE = OneHotEncoder()
    OHE.fit(GBDT.apply(X_train)[:, :, 0])
    LR = LogisticRegression()
    LR.fit(OHE.transform(GBDT.apply(X_train_lr)[:, :, 0]), Y_train_lr)
    Y_pred = LR.predict_proba(OHE.transform(GBDT.apply(X_test)[:, :, 0]))[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('GradientBoosting + LogisticRegression: ', auc)
    return fpr, tpr


def LR():
    LR = LogisticRegression(n_jobs=4, C=0.1, penalty='l2')
    LR.fit(X_train, Y_train)
    Y_pred = LR.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('LogisticRegression: ', auc)
    return fpr, tpr


def XGBoost():
    XGB = xgb.XGBClassifier(nthread=4, learning_rate=0.08, n_estimators=100, colsample_bytree=0.5)
    XGB.fit(X_train, Y_train)
    Y_pred = XGB.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('XGBoost: ', auc)
    return fpr, tpr


if __name__ == '__main__':
    fpr_xgb_lr, tpr_xgb_lr = XGBoostLR()
    fpr_xgb, tpr_xgb = XGBoost()
    fpr_lr, tpr_lr = LR()
    fpr_rf_lr, tpr_rf_lr = RandomForestLR()
    fpr_gbdt_lr, tpr_gbdt_lr = GBDTLR()

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
    plt.plot(fpr_gbdt_lr, tpr_gbdt_lr, label='GBT + LR')
    plt.plot(fpr_xgb, tpr_xgb, label='XGB')
    plt.plot(fpr_lr, tpr_lr, label='LR')
    plt.plot(fpr_xgb_lr, tpr_xgb_lr, label='XGB + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
    plt.plot(fpr_gbdt_lr, tpr_gbdt_lr, label='GBT + LR')
    plt.plot(fpr_xgb, tpr_xgb, label='XGB')
    plt.plot(fpr_lr, tpr_lr, label='LR')
    plt.plot(fpr_xgb_lr, tpr_xgb_lr, label='XGB + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()
