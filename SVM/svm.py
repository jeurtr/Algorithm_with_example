# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm

# data iris digits
iris = datasets.load_iris()
digits = datasets.load_digits()

# fit model
clf = svm.SVC()
x, y = iris.data, iris.target
clf.fit(x, y)
