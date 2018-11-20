# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import test_data

def model_fit(dtrain, trainlabel, dtest, testlabel, param_set=None):
    rf_model = RandomForestClassifier(criterion='gini', max_depth=None, max_features='sqrt',
                                      max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
                                      n_estimators=10, oob_score=True, random_state=10)
    if param_set:
        rf_model.set_params(**param_set)
    print(rf_model)

    rf_model.fit(dtrain, trainlabel)
    print(rf_model.oob_score_)
    train_pred = rf_model.predict(dtrain)
    train_prob = rf_model.predict_proba(dtrain)
    test_pred = rf_model.predict(dtest)
    test_prob = rf_model.predict_proba(dtest)
    # print(test_prob[:3])
    # print(rf_model.feature_importances_)
    print('训练准确率 : %.4g' % accuracy_score(trainlabel, train_pred))
    print('测试准确率 : %.4g' % accuracy_score(testlabel, test_pred))
    return rf_model

def grid_search(rf_model, param_search, dtrain, labels):
    rf_gs = RandomForestClassifier(n_estimators=10, oob_score=True, random_state=10)
    rf_gs.set_params(**rf_model.get_params())

    gsearch = GridSearchCV(estimator=rf_gs, param_grid=param_search, cv=5)
    gsearch.fit(dtrain, labels)
    print(gsearch.cv_results_)
    print(gsearch.best_score_)
    print(gsearch.best_params_)
    return gsearch.best_params_


if __name__ == '__main__':
    train, test = test_data.load_data()
    dtrain, dtest, trainlabel, testlabel = test_data.train_split(train)
    param = {'max_features': 'sqrt', 'n_estimators':30}
    rf_model = model_fit(dtrain, trainlabel, dtest, testlabel, param)

    param_search = {'n_estimators':range(30, 100, 20)}
    param = grid_search(rf_model, param_search, dtrain, trainlabel)

    rf_model.set_params(**param)
    param_search = {'max_depth': range(3, 15, 2), 'max_sample_split': range(10, 100, 20)}
    param = grid_search(rf_model, param_search, dtrain, trainlabel)

    rf_model.set_params(**param)
    param_search = {'max_sample_leaf': range(10, 60, 10)}
    param = grid_search(rf_model, param_search, dtrain, trainlabel)

    rf_model.set_params(**param)
    print(rf_model)
    rf_model = model_fit(dtrain, trainlabel, dtest, testlabel, rf_model.get_params())


