# -*- coding: utf-8 -*-
"""
Created on Wed May 17 22:32:25 2017

@author: vrtjso
"""

import numpy as np
import pandas as pd
#from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
#from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, VotingClassifier
#from sklearn import decomposition
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import Imputer
#from xgboost import XGBRegressor
import xgboost as xgb
import operator
#import lightgbm as lgb
import gc
from Utils import RMSLE, eval_metric, objective

#XGBoost
#initialization
trainDf = pd.read_csv('train_featured.csv')
testDf = pd.read_csv('test_featured.csv')
Xtrain = trainDf.drop(['price_doc','w'], 1).values
Ytrain = trainDf['price_doc'].values
# Ytrain = Ytrain * 0.97 # multiplier
f_names = trainDf.drop(['price_doc','w'],1).columns
w = trainDf.w.values
dtrain_all = xgb.DMatrix(Xtrain,label=Ytrain, feature_names=f_names, weight=w)


### Xgboost ###
##Customized CV
#min CV on 0.02: 0.4576
#min CV on 0.05(new):233 (230 with weight)
params = {'eta':0.05, 'max_depth':5, 'subsample':0.8, 'colsample_bytree':0.8, 'min_child_weight':1,
          'gamma':0, 'silent':1, 'objective':'reg:linear', 'eval_metric':'rmse'}

xgbcv = xgb.cv(params, dtrain_all, 10000, nfold=8, show_stdv=False, early_stopping_rounds=100, verbose_eval=100)

xgbcv.iloc[-1,[0,2]].to_csv('xgb_CV_score.log', header=True, index=True, sep=' ',
        mode='a', float_format = '%.4f')
print('The CV score is', xgbcv.iloc[-1,0])
print('The best number of rounds is', xgbcv.shape[0])

num_round = xgbcv.shape[0]
# num_round = 386
bst = xgb.train(params, dtrain_all, num_round)
Xtest = testDf.values
dtest = xgb.DMatrix(Xtest, feature_names = f_names)
prediction = bst.predict(dtest)
#CreateOutput(prediction)

output = pd.read_csv('test.csv')
output = output[['id']]
output['price_doc'] = prediction
output.to_csv('Submission.csv',index=False)

fs = bst.get_fscore()
sorted_fs = sorted(fs.items(), key=operator.itemgetter(1),reverse = False)


'''
#Grid Search
GSparams = {'max_depth':[4,5,6], 'colsample_bytree':[0.6,0.8,1]} # grid search params
params = {'eta': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'silent':1,
          'min_child_weight': 1, 'gamma': 0, 'objective': 'reg:linear', 'eval_metric': 'rmse'} # default params
gs = '' # str to store GridSearch result
items = sorted(GSparams.items())
for i in items[0][1]:
    for j in items[1][1]:
        params[items[0][0]] = i
        params[items[1][0]] = j
        xgbcv = xgb.cv(params, dtrain_all, 10000, nfold = 8, early_stopping_rounds = 50)
        gs = ('test_score:%.4f, train_score:%.4f, best_round:%d, %s:%f, %s:%f \n'
        %(xgbcv.iloc[-1,0], xgbcv.iloc[-1,2], xgbcv.shape[0], items[0][0], i, items[1][0], j))
        f = open('xgb_GridSearch.log', mode = 'a')
        f.write(gs)
        f.close()
'''

#Fit xgboost
#split = 24000
#indices = np.random.permutation(Xtrain.shape[0])
#train_id, test_id = indices[:split], indices[split:]
#x_train, y_train, x_valid, y_valid = Xtrain[train_id], Ytrain[train_id], Xtrain[test_id], Ytrain[test_id]
#d_train = xgb.DMatrix(x_train, label=y_train)
#d_valid = xgb.DMatrix(x_valid, label=y_valid)
#watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#params = {'eta':0.05, 'max_depth':4, 'subsample':0.8, 'colsample_bytree':0.8, 'min_child_weight':2,
#          'gamma':0, 'objective':'reg:linear', 'eval_metric': 'rmse'}
#partial_bst = xgb.train(params, d_train, 1000, early_stopping_rounds=20, evals = watchlist, verbose_eval=30)


#plt.scatter(Ytrain,prediction)
#plt.plot(range(12,19),range(12,19),color='red')
