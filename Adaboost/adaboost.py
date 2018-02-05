# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np


def loadSimpData():
    datMat = np.matrix([[1., 2.1],
                        [2., 1, 1],
                        [1.2, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels
