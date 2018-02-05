def sort(a):
    l = []
    for i in a:
        if i not in l:
            l.append(i)
    return l


from math import log
from collections import Counter

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    L=[]
    for featVec in dataSet:
    	
        currentLabel = featVec[-1]
        L.append(currentLabel)
        labelCounts=Counter(L)
            # labelCounts[currentLabel] = 0
            # labelCounts[currentLabel] += 1

    return labelCounts
