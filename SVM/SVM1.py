# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from skleanrn import svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

data = []
labels = []
with open('data\\1.txt') as infile:
    for line in infile:
        tokens = line.strip().split('')
        data.append([float(tk) for tk in token[:-1]])
        labels.append(token[-1])
