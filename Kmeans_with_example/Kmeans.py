# !/usr/bin/python
# -*- coding:utf-8 -*-
from numpy import *


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        datMat.append(fltLine)
    return dataMat


def disEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, J]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, l)
    return centroids
