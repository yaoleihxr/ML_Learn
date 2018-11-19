# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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