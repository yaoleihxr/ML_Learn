# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

path = '/sas/cxdx/yl/test/'
# path = 'E:/'

def load_data():
    train = pd.read_csv(path + 'MNIST.train.csv')
    test = pd.read_csv(path + 'MNIST.test.csv')
    print(train.shape)
    print(test.shape)
    print(train['label'].value_counts())
    return train, test

def train_split(train):
    train_data = train.values[:, 1:]
    train_label = train.values[:, 0]
    data_train, data_val, train_label, val_label = train_test_split(train_data, train_label, test_size=0.3)
    return data_train, data_val, train_label, val_label

def model_fit(dtrain, param_set=None):
    gbdt_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0,
                                            min_samples_split=2, min_samples_leaf=1, max_depth=3,
                                            max_features='sqrt')
    if param_set:
        gbdt_model.set_params(**param_set)

    gbdt_model.fit(dtrain.values[:, 1:], dtrain.values[:, 0])
    dtrain_pred = gbdt_model.predict(dtrain.values[:, 1:])
    print(gbdt_model.feature_importances_)
    print('准确率 : %.4g' % metrics.accuracy_score(dtrain.values[:, 0], dtrain_pred))
    return gbdt_model

def grid_search(gbdt_model, param_search, dtrain):
    gbdt_gs = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0,
                                            min_samples_split=2, min_samples_leaf=1, max_depth=3,
                                            max_features='sqrt')
    param_set = gbdt_model.get_params()
    gbdt_gs.set_params(**param_set)

    gsearch = GridSearchCV(estimator=gbdt_gs, param_grid=param_search, cv=5)
    gsearch.fit(dtrain.values[:, 1:], dtrain.values[:, 0])
    print(gsearch.cv_results_)
    print(gsearch.best_score_)
    print(gsearch.best_params_)


if __name__ == '__main__':
    train, test = load_data()
    gbdt_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0,
                                            min_samples_split=2, min_samples_leaf=1, max_depth=3,
                                            max_features='sqrt')
    model_fit(train, {'n_estimators':100})

    param_search = {'n_estimators': range(50, 220, 30)}
