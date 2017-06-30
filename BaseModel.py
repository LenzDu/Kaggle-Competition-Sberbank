# author: vrtjso

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, VotingClassifier, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
import time

trainDf = pd.read_csv('train_featured.csv')
testDf = pd.read_csv('test_featured.csv')

# Dealing with nan: Imputed by -999
# trainDf.fillna(-999, inplace=True)
# testDf.fillna(-999, inplace=True)

Xtrain = trainDf.drop(['price_doc','w'], 1).values
Ytrain = trainDf['price_doc'].values
# Ytrain = Ytrain * 0.97 # Multiplier

# Dealing with nan: Imputed by mean
imp = Imputer(strategy='mean', axis=0, copy = False)
Xtrain = imp.fit_transform(Xtrain)

cv = ShuffleSplit(n_splits=6, test_size=0.2)
# RF = RandomForestRegressor(n_estimators=500, max_features=0.2)
# scores_RF = cross_val_score(RF, Xtrain, Ytrain, cv=cv, n_jobs=1, scoring='neg_mean_squared_error')
# print(np.sqrt(-scores_RF.mean()))
# ETR = ExtraTreesRegressor(n_estimators=500, max_features=0.1,max_depth=10)
# scores_ETR = cross_val_score(ETR, Xtrain, Ytrain, cv=cv, n_jobs=3, scoring='neg_mean_squared_error')
# print(np.sqrt(-scores_ETR.mean()))
# Ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),n_estimators=200)
# scores_Ada = cross_val_score(Ada, Xtrain, Ytrain, cv=cv, n_jobs=5, scoring='neg_mean_squared_error')
# print(np.sqrt(-scores_Ada.mean()))
MLP = MLPRegressor(hidden_layer_sizes=(100,),alpha=0.0001,max_iter=500,early_stopping=True)
scores_MLP = cross_val_score(MLP, Xtrain, Ytrain, cv=cv, n_jobs=1, scoring='neg_mean_squared_error')
print(np.sqrt(-scores_MLP.mean()))
# trainDf_lasso = trainDf.drop(['price_doc','w','sub_area','material'], 1)
# for c in trainDf_lasso.columns:
#     if c.startswith('ID'): trainDf_lasso.drop(c,1,inplace=True)
# Xtrain_Lasso = trainDf_lasso.values
# La = Lasso(max_iter=2000)
# scores_La = cross_val_score(La, Xtrain_Lasso, Ytrain, cv=cv, n_jobs=1, scoring='neg_mean_squared_error')
# print(np.sqrt(-scores_La.mean()))
# GBR = GradientBoostingRegressor(n_estimators=100,max_depth=3,max_features=None)
# scores_GBR = cross_val_score(GBR, Xtrain, Ytrain, cv=cv, n_jobs=1, scoring='neg_mean_squared_error')
# print(np.sqrt(-scores_GBR.mean()))

# f = open(r'base_model_score.log',mode='a') #record CV result
# f.write('RF:' + str(np.sqrt(-scores_RF.mean())) + '\n')
# f.write('ETR:' + str(np.sqrt(-scores_ETR.mean())) + '\n')
# f.write('Ada:' + str(np.sqrt(-scores_Ada.mean())) + '\n')
# f.close()

### GridSearch to find the best params ###
# estimator = GridSearchCV(Ada, dict(base_estimator__max_depth=[3,5,10,15], n_estimators=[100,150,200,500]),
#                          cv = cv, n_jobs = 5, scoring='neg_mean_squared_error')
# estimator = GridSearchCV(Ada, dict(n_estimators=[300,500], max_features=[0.1,0.2,0.3,0.4]),
#                          cv = cv, n_jobs = 5, scoring='neg_mean_squared_error')
# estimator.fit(Xtrain,Ytrain)

# f = open('GS_Adaboost.log',mode='a')
# f.write(time.asctime() + '\n')
# for i, param in enumerate(estimator.cv_results_['params']):
#     f.write(str(param) + ': ')
#     f.write(str(np.sqrt(-estimator.cv_results_['mean_test_score'][i])) + '\n')
# f.write('Best Parameters: '  + str(estimator.best_params_) + '\n')
# f.write('Best Score: ' + str(np.sqrt(-estimator.best_score_))+'\n\n')
# f.close()


"""
# Make prediction
RF.fit(Xtrain, Ytrain)
Xtest = testDf.values
# Xtest = imp.fit_transform(Xtest)
prediction = RF.predict(Xtest)

output = pd.read_csv('test.csv')
output = output[['id']]
output['price_doc'] = prediction
output.to_csv(r'Ensemble\Submission_RF.csv',index=False)
"""