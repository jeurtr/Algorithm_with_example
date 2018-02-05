#!/usr/bin/python
#-*- coding:utf-8 -*-


# FP树的类定义
class treeNode:

    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print '' * ind, self.name, '', self.count
        for child in self.children.values():
            child.disp(ind + 1)


# FP树构建函数
def createTree(dataSet, minsup=1):  # 设定最小支持度为1
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in headerTable.keys():
        '''移除不满足最小支持度的元素项'''
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:

        return None, None

    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null set', 1, None)
    for tranSet, count in dataSet.items():
        locaID = {}
        for item in tranSet:
            if item in freqItemSet:
                locaID[item] = headerTable[item][0]
            if len(locaID) > 0:
                orderedItems = [v[0] for v in sorted(locaID.items(),
                                                     key=lambda p:p[1], reverse=True)]
                updateTree(orderedItems, retTree, headerTable, count)
        return retTree, headerTable

    def updateTree(item, inTree, headerTable, count):
        if items[0] in inTree.children:
            inTree.children[items[0]].inc(count)
        else:
            inTree.children[items[0]] = treeNode(items[0], count, inTree)
            if headerTable[items[0]][1] == None:
                headerTable[item[0]][1] = inTree.children[items[0]]
            else:
                updateTree(headerTable[items[0]][1], inTree.children[items[0]])
        if len(items) > 1:
            updateTree(items[1::], inTree.children[
                       items[0]], headerTable, count)

    def updateHeader(nodeToTest, targetNode):
        while(nodeToTest.nodeLink != None):
            nodeToTest = nodeToTest.nodeLink
        nodeToTest.nodeLink = targetNode
