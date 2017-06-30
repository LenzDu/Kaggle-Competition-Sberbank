# author: vrtjso
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from sklearn.preprocessing import StandardScaler
from Utils import CreateOutput

trainDf = pd.read_csv('train_featured.csv')
Xtrain = trainDf.drop(['price_doc','w'],1)
w = trainDf.w.values
Ytrain = trainDf.price_doc
# scaler = StandardScaler().fit(Ytrain)
# Ytrain = scaler.transform(Ytrain)
Ytrain = Ytrain * 0.0000001
train = lgb.Dataset(Xtrain, Ytrain, weight=w)
del Xtrain, Ytrain; gc.collect()

#min CV on 0.1(new): 2350 (num_leave:15, min_data:30)
#min CV 0.1 normalized: 0.511 (num_leave:15, min_data:30)
params = {'objective':'regression','metric':'rmse',
          'learning_rate':0.1,'max_depth':-1,'sub_feature':0.7,'sub_row':1,
          'num_leaves':15,'min_data':30,'max_bin':20,
          'bagging_fraction':0.9,'bagging_freq':40,'verbosity':-1}
lgbcv = lgb.cv(params, train, 10000, nfold=6, early_stopping_rounds=50,
               verbose_eval=50, show_stdv=False)['rmse-mean']
print('The final CV score:', lgbcv[-1])

# best_round = len(lgbcv)
# bst = lgb.train(params, train, best_round)
# fs = bst.feature_importance()
# f_name = bst.feature_name()
# f = dict()
# for i in range(0,len(fs)):
#    f[f_name[i]]=fs[i]

# Xtest = pd.read_csv('test_featured.csv')
# prediction = bst.predict(Xtest) * 10000000
# # prediction = scaler.inverse_transform(prediction)
# output = pd.read_csv('test.csv')
# output = output[['id']]
# output['price_doc'] = prediction
# output.to_csv(r'Ensemble\Submission_lgb.csv',index=False)
