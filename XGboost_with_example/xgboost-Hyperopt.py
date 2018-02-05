# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from random import shuffle
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import cross_val_score
import pickle
import time
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def loadFile():
    target = 'Disbursed'
    IDcol = 'ID'
    predictors = [x for x in traindata.columns if x not in [target, IDcol]]

    trainfilePath = r"C:\Users\dell\Desktop\train_modified.csv"
    testfilePath = r"C:\Users\dell\Desktop\test_modified.csv"
    traindata = pd.read_csv(trainfilePath)
    testdata = pd.read_csv(testfilePath)

    return traindata, testdata

data = loadFile()
train_x = traindata[predictors]
train_y = traindata[target]
# labels = label.reshape((1, -1))
# label = labels.tolist()[0]
# minmaxscaler = MinMaxScaler()
# attrs = minmaxscaler.fit_transform(attrs)

# index = range(0, len(label))
# shuffle(index)
# trainIndex = index[:int(len(label) * 0.7)]
# print len(trainIndex)
# testIndex = index[int(len(label) * 0.7):]
# print len(testIndex)
# attr_train = attrs[trainIndex, :]
# print attr_train.shape
# attr_test = attrs[testIndex, :]
# print attr_test.shape
# label_train = labels[:, trainIndex].tolist()[0]
# print len(label_train)
# label_test = labels[:, testIndex].tolist()[0]
# print len(label_test)
# print np.mat(label_train).reshape((-1, 1)).shape


def GBM(argsDict):
    max_depth = argsDict["max_depth"] + 5
    n_estimators = argsDict['n_estimators'] * 5 + 50
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"] + 1
    print "max_depth:" + str(max_depth)
    print "n_estimator:" + str(n_estimators)
    print "learning_rate:" + str(learning_rate)
    print "subsample:" + str(subsample)
    print "min_child_weight:" + str(min_child_weight)
    global attr_train, label_train

    gbm = xgb.XGBClassifier(nthread=4,  # 进程数
                            max_depth=max_depth,  # 最大深度
                            n_estimators=n_estimators,  # 树的数量
                            learning_rate=learning_rate,  # 学习率
                            subsample=subsample,  # 采样数
                            min_child_weight=min_child_weight,  # 孩子数
                            max_delta_step=10,  # 10步不降则停止
                            objective="binary:logistic")

    metric = cross_val_score(gbm, attr_train, label_train,
                             cv=5, scoring="roc_auc").mean()
    print metric
    return -metric

space = {"max_depth": hp.randint("max_depth", 15),
         # [0,1,2,3,4,5] -> [50,]
         "n_estimators": hp.randint("n_estimators", 10),
         # [0,1,2,3,4,5] -> 0.05,0.06
         "learning_rate": hp.randint("learning_rate", 6),
         # [0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "subsample": hp.randint("subsample", 4),
         "min_child_weight": hp.randint("min_child_weight", 5),
         }
algo = partial(tpe.suggest, n_startup_jobs=1)
best = fmin(GBM, space, algo=algo, max_evals=4)

print(best)
print(GBM(best))


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(
            dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print(("Accuracy : %.4f") % metrics.accuracy_score(
        dtrain['Disbursed'].values, dtrain_predictions))
    print(("AUC Score (Train): %f") % metrics.roc_auc_score(
        dtrain['Disbursed'], dtrain_predprob))

#     Predict on testing data:
    dtest['predprob'] = alg.predict_proba(dtest[predictors])[:, 1]
    results = test_results.merge(dtest[['ID', 'predprob']], on='ID')
    print ('AUC Score (Test): %f' % metrics.roc_auc_score(results['Disbursed'], results['predprob']))

    feat_imp = pd.Series(alg.booster().get_fscore()
                         ).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


xgb4 = XGBClassifier(
    learning_rate=0.01,
    n_estimators=5000,
    max_depth=4,
    min_child_weight=6,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.005,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)

modelfit(xgb4, train, test, predictors)
