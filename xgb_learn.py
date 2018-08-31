# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV


def load_data():
    train = pd.read_csv('E:/train_modified.csv')
    test = pd.read_csv('E:/test_modified.csv')