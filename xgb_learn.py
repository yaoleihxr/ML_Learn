# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV

x = XGBClassifier()
x.get_xgb_params()
x.get_params()

def load_data():
    train = pd.read_csv('E:/train_modified.csv')
    test = pd.read_csv('E:/test_modified.csv')
    print(train.shape)
    print(test.shape)
    return train, test

# alg: model
# predictors: data columns
# target: predict column
def modelfit(alg, dtrain, dtest, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cv_res = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                        nfold=cv_folds, early_stopping_rounds=early_stopping_rounds)
