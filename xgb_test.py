# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

path = '/sas/cxdx/yl/test/'

def load_data():
    train = pd.read_csv(path + 'MNIST.train.csv')
    test = pd.read_csv(path + 'MNIST.test.csv')
    print(train.shape)
    print(test.shape)
    return train, test

def train_split(train):
    train_data = train.values[:, 1:]
    train_label = train.values[:, 0]
    xgb_train, xgb_val, train_label, val_label = train_test_split(train_data, train_label, test_size=0.3)
    return xgb_train, xgb_val, train_label, val_label

def model_fit(xgb_model, dtrain, cv_folds=5, early_stopping_rounds=50):
    xgb_train, xgb_val, train_label, val_label = train_split(train)
    xgb_param = xgb_model.get_xgb_params()
    # print(xgb_param) # dict
    xgb_train = xgb.DMatrix(xgb_train, label=train_label)
    xgb_test = xgb.DMatrix(xgb_val)
    cv_result = xgb.cv(xgb_param, xgb_train, nfold=5, num_boost_round=1000,
                       early_stopping_rounds=early_stopping_rounds)
    print(cv_result)
    print(cv_result.shape[0])

    xgb_model.set_params(n_estimators=cv_result.shape[0])
    # eval_metric[default according to objective] eval_metric='mlogloss'
    xgb_model.fit(dtrain.values[:, 1:], dtrain.values[:, 0])
    dtrain_pred = xgb_model.predict(dtrain.values[:, 1:])
    dtrain_pred_prob = xgb_model.predict_proba(dtrain.values[:, 1:])
    feat = xgb_model.get_booster().get_fscore()
    print(type(feat))
    print('准确率 : %.4g' % metrics.accuracy_score(dtrain.values[:, 0], dtrain_pred))



if __name__ == '__main__':
    train, test = load_data()
    xgb_model = XGBClassifier(booster='gbtree', learning_rate=0.1, n_estimators=1000, max_depth=6,
                              reg_alpha=0.05, min_child_weight=1, gamma=0, subsample=0.8,
                              colsample_bytree=1, objective='multi:softmax', num_class=10,
                              scale_pos_weight=1)
    model_fit(xgb_model, train)




