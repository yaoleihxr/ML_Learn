# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


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
        print(xgb_param)
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cv_res = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                        nfold=cv_folds, early_stopping_rounds=early_stopping_rounds)
        print(cv_res)
        print(cv_res.shape[0])
        alg.set_params(n_estimators=cv_res.shape[0])

    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
    dtrain_pred = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    feat = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    print(feat)
    print('准确率 : %.4g' % metrics.accuracy_score(dtrain[target].values, dtrain_pred))
    print('AUC 得分 (训练集): %f' % metrics.roc_auc_score(dtrain[target], dtrain_predprob))


# GridSearch
def grid_search(param, data, label):
    gsearch = GridSearchCV(estimator=XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
                                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                           param_grid=param, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(data, label)
    print(gsearch.cv_results_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)


if __name__ == '__main__':
    train, test = load_data()
    target = 'Disbursed'
    IDcol = 'ID'
    predictors = [x for x in train.columns if x not in [target, IDcol]]
    print(predictors)
    xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=6, reg_alpha=0.05,
                         min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
    modelfit(xgb1, train, test, predictors, target, True)

    # param_test1={'max_depth':range(2,10,2),'min_child_weight':range(1,6,2)}
    # grid_search(param_test1, train[predictors], train[target])

    # param_test2 = {'max_depth': [6,8,10], 'min_child_weight': [3,4,5]}
    # grid_search(param_test2, train[predictors], train[target])