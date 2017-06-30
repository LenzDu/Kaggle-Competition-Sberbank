# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:01:55 2017

@author: vrtjso
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

#Data importing
trainDf = pd.read_csv('train.csv').set_index('id')
testDf = pd.read_csv('test.csv').set_index('id')
fix = pd.read_excel('BAD_ADDRESS_FIX.xlsx').set_index('id')
testDf['isTrain'] = 0
trainDf['isTrain'] = 1
allDf = pd.concat([trainDf,testDf])
allDf.update(fix, filter_func = lambda x:np.array([True]*x.shape[0]))#update fix data

allDf['time'] = pd.to_datetime(allDf['timestamp'])
macro = pd.read_csv('macro.csv')
allDf = allDf.join(macro.set_index('timestamp'), on='timestamp')
#areaDf = allDf.drop(['time','full_sq','price_doc','life_sq','floor','max_floor','material','build_year','num_room','kitch_sq','state','product_type'],1)

#plot
sns.distplot(trainDf['log_price']) #dist of price
sns.distplot(trainDf.ix[trainDf['log_price']>np.log1p(20000000),'log_price']) #range 0 ~ 2e7
plt.close()
plt.scatter(trainDf.full_sq,trainDf.log_price)

#corr
corr = trainDf.corr()
price_corr = corr.price_doc

lm = sns.lmplot('num_room','log_price', data = trainDf)
axe = lm.axes
axe[0,0].set_xlim(0,7.5)
axe[0,0].set_ylim(0,80000000)
plt.show()

#dealing with macro data
macro = macro.loc[365:2343,:] #drop data before 2011 and after 2016.6
macro_full = macro.loc[:,macro.count()==1979] # drop nan columns
macro_missing = macro.loc[:2190,macro.count()==1826]
pca = PCA(2)
pca.fit(macro_full.drop('timestamp',1).values)
eco_1 = pca.transform(macro_full.drop('timestamp',1))
pca = PCA(1)
pca.fit(macro_missing.values)
eco_2 = pca.transform(macro_missing)
eco = pd.DataFrame()
eco['timestamp'] = macro['timestamp']
eco['index1'] = eco_1[:,0]
eco['index2'] = eco_1[:,1]