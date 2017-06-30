# -*- coding: utf-8 -*-
"""
Created on Wed May 17 23:07:54 2017

@author: vrtjso
"""
import numpy as np
import pandas as pd
import random
from operator import le, eq
from sklearn.decomposition import PCA
from sklearn import model_selection, preprocessing

def RMSLE(y, yfit):
    n = len(y)
    s = 0
    for i in range(0,n):
        s += (np.log(yfit[i] + 1) - np.log(y[i] + 1)) ** 2
    RMSLE = np.sqrt(s/n)
    return RMSLE


#Objective function used for xgb
def objective(yfit, dtrain):
    y = dtrain.get_label()
    g = 2 * (np.log(yfit + 1) - np.log(y + 1)) / (yfit + 1)
    h = (2 - 2 * np.log(yfit + 1) + 2 * np.log(y + 1)) / ((yfit + 1) ** 2)
    #n = dtrain.num_row()
    #g = []
    #h = []
    #for i in range(0,n):
    #    g.append(2 * (np.log(yfit[i] + 1) - np.log(y[i] + 1)) / (yfit[i] + 1))
    #    h.append((2 - 2 * np.log(yfit[i] + 1) + 2 * np.log(y[i] + 1)) / ((yfit[i] + 1) ** 2))
    return g, h

#Metric used for xgb cv
def eval_metric(yfit, dtrain):
    y = dtrain.get_label()
    return 'error', RMSLE(y,yfit)

def CreateOutput(prediction):
    output = pd.read_csv('test.csv')
    output = output[['id']]
    output['price_doc'] = prediction
    output.to_csv('Submission.csv',index=False)
    
    
#load data
def loadTraindata(takeLog=True):
    #filename = 'train.csv'
    filename = 'train_featured.csv'
    rawDf = pd.read_csv(filename)
    Ytrain = rawDf['price_doc'].values
    Xtrain = rawDf.drop(['price_doc','w'], 1).values
    return Ytrain, Xtrain

def loadTestdata():
    #filename = 'test.csv'
    filename = 'test_featured.csv'
    rawDf = pd.read_csv(filename)
    Xtest = rawDf.values
    return Xtest

#load random small part of data for fast model testing
def loadSample(n=300):
    #filename = 'train.csv'
    #filename = 'train_cleaned.csv'
    filename = 'train_featured.csv'
    size = pd.read_csv(filename).shape[0]
    skip = sorted(random.sample(range(1,size+1),size-n))
    rawDf = pd.read_csv(filename, skiprows = skip)

    Ytrain = rawDf['log_price'].values
    Xtrain = rawDf.drop(['log_price'], 1).values
    return Ytrain, Xtrain

#Used to undersample strange values
def sample_vals(df, price_value, ratio, condition):
    indices = condition(df.price_doc, price_value) & (df.product_type == 0)
    df_resampled = df.loc[indices].sample(frac=ratio)
    df_remaining = df.loc[~indices]
    df_new = pd.concat([df_resampled, df_remaining], axis=0)
    return df_new

#Encoding dummy variables
def Encoding(TestEncoding = True):
    filename = 'test.csv' if TestEncoding else 'train.csv'
    rawDf = pd.read_csv(filename)
              
    #Drop variable with no use and small variance, and sub area
    rawDf = rawDf.drop(["id","ID_metro","ID_railroad_station_walk","ID_railroad_station_avto",
            "ID_big_road1", "ID_big_road2", "ID_railroad_terminal", "ID_bus_terminal"],1)
    rawDf = rawDf.drop(["culture_objects_top_25_raion","oil_chemistry_raion","railroad_terminal_raion",
                 "nuclear_reactor_raion", "build_count_foam", "big_road1_1line","railroad_1line",
                 "office_sqm_500", "trc_sqm_500", "cafe_count_500_price_4000", "cafe_count_500_price_high",
                 "mosque_count_500", "leisure_count_500", "office_sqm_1000", "trc_sqm_1000",
                 "cafe_count_1000_price_high", "mosque_count_1000", "cafe_count_1500_price_high",
                 "mosque_count_1500", "cafe_count_2000_price_high"],1)
    #rawDf = rawDf.drop('sub_area',1)
    
    result = rawDf
    for i in range(1,rawDf.shape[1]): #Do not encode timestamp
        if rawDf.ix[:,i].dtype == np.object:
            varName = rawDf.columns[i]
            if varName == 'sub_area':
                dummy_ranks = pd.get_dummies(rawDf[varName], prefix = varName)
            else:
                dummy_ranks = pd.get_dummies(rawDf[varName], prefix = varName, drop_first=True)
            result = pd.concat([result, dummy_ranks], axis=1)
            result = result.drop(varName, 1)
    varName = 'material' #special case
    dummy_ranks = pd.get_dummies(rawDf[varName], prefix = varName)
    result = pd.concat([result, dummy_ranks], axis=1)
    result = result.drop(varName, 1)
    outputFile = 'test_encoded.csv' if TestEncoding else 'train_encoded.csv'
    result.to_csv(outputFile,index=False)
    #return result

#用PCA合并同一个系列高度相关的feature
def FeatureCombination(Df,s='',num_feature=2): 
    feature_set = []
    for c in Df.columns:
        if c.startswith(s): feature_set.append(c)
    print('combining', len(feature_set), 'features')
    data = Df[feature_set].values

    for c in Df.columns:
        if Df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(Df[c].values))
            Df[c] = lbl.transform(list(Df[c].values))
            
    imp = preprocessing.Imputer()
    data = imp.fit_transform(data)
    data = preprocessing.scale(data)
    pca = PCA(num_feature)
    pca.fit(data)
    print('explained_variance_ratio_:', pca.explained_variance_ratio_)
    trans = pca.transform(data)
    for i in range(0,num_feature):
        Df[s+'_%d'%(i+1)] = trans[:,i]
    Df.drop(feature_set,1,inplace=True)
    return Df