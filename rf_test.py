# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import test_data

def model_fit(dtrain, dlabel, param_set=None):
    rf_model = RandomForestClassifier(n_estimators=10, oob_score=True, random_state=10)
    if param_set:
        rf_model.set_params(**param_set)

    rf_model.fit(dtrain, dlabel)
    print(rf_model.oob_score_)
    dpred = rf_model.predict(dtrain)
    dpred_prob = rf_model.predict_proba(dtrain)
    print(rf_model.feature_importances_)
    print('准确率 : %.4g' % accuracy_score(dlabel, dpred))
    print('AUC : %.4g' % roc_auc_score(dlabel, dpred))
    return rf_model

if __name__ == '__main__':
    train, test = test_data.load_data()
    dtrain = train.values[:, 1:]
    dlabel = train.values[:, 0]
    rf_model = model_fit(dtrain, dlabel)


