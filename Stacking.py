# author: vrtjso

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# 封装一下lightgbm让其可以在stacking里面被调用
class LGBregressor(object):
    def __init__(self,params):
        self.params = params

    def fit(self, X, y, w):
        y /= 10000000
        # self.scaler = StandardScaler().fit(y)
        # y = self.scaler.transform(y)
        split = int(X.shape[0] * 0.8)
        indices = np.random.permutation(X.shape[0])
        train_id, test_id = indices[:split], indices[split:]
        x_train, y_train, w_train, x_valid, y_valid,  w_valid = X[train_id], y[train_id], w[train_id], X[test_id], y[test_id], w[test_id],
        d_train = lgb.Dataset(x_train, y_train, weight=w_train)
        d_valid = lgb.Dataset(x_valid, y_valid, weight=w_valid)
        partial_bst = lgb.train(self.params, d_train, 10000, valid_sets=d_valid, early_stopping_rounds=50)
        num_round = partial_bst.best_iteration
        d_all = lgb.Dataset(X, label = y, weight=w)
        self.bst = lgb.train(self.params, d_all, num_round)

    def predict(self, X):
        return self.bst.predict(X) * 10000000
        # return self.scaler.inverse_transform(self.bst.predict(X))

# 封装一下xgboost让其可以在stacking里面被调用
class XGBregressor(object):
    def __init__(self, params):
        self.params = params

    def fit(self, X, y, w=None):
        if w==None:
            w = np.ones(X.shape[0])
        split = int(X.shape[0] * 0.8)
        indices = np.random.permutation(X.shape[0])
        train_id, test_id = indices[:split], indices[split:]
        x_train, y_train, w_train, x_valid, y_valid,  w_valid = X[train_id], y[train_id], w[train_id], X[test_id], y[test_id], w[test_id],
        d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        partial_bst = xgb.train(self.params, d_train, 10000, early_stopping_rounds=50, evals = watchlist, verbose_eval=100)
        num_round = partial_bst.best_iteration
        d_all = xgb.DMatrix(X, label = y, weight=w)
        self.bst = xgb.train(self.params, d_all, num_round)

    def predict(self, X):
        test = xgb.DMatrix(X)
        return self.bst.predict(test)

# This object modified from Wille on https://dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/
class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, trainDf, testDf):
        X = trainDf.drop(['price_doc', 'w'], 1).values
        y = trainDf['price_doc'].values
        w = trainDf['w'].values
        T = testDf.values

        X_fillna = trainDf.drop(['price_doc', 'w'], 1).fillna(-999).values
        T_fillna = testDf.fillna(-999).values

        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True))
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            print('Training base model ' + str(i+1) + '...')
            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                print('Training round ' + str(j+1) + '...')
                if clf not in [xgb1,lgb1]: # sklearn models cannot handle missing values.
                    X = X_fillna
                    T = T_fillna
                X_train = X[train_idx]
                y_train = y[train_idx]
                w_train = w[train_idx]
                X_holdout = X[test_idx]
                # w_holdout = w[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train, w_train)
                y_pred = clf.predict(X_holdout)
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)
            S_test[:, i] = S_test_i.mean(1)
        self.S_train, self.S_test, self.y = S_train, S_test, y  # for diagnosis purpose
        self.corr = pd.concat([pd.DataFrame(S_train),trainDf['price_doc']],1).corr() # correlation of predictions by different models.
        # cv_stack = ShuffleSplit(n_splits=6, test_size=0.2)
        # score_stacking = cross_val_score(self.stacker, S_train, y, cv=cv_stack, n_jobs=1, scoring='neg_mean_squared_error')
        # print(np.sqrt(-score_stacking.mean())) # CV result of stacking
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)
        return y_pred

if __name__ == "__main__":
    trainDf = pd.read_csv('train_featured.csv')
    testDf = pd.read_csv('test_featured.csv')

    params1 = {'eta':0.05, 'max_depth':5, 'subsample':0.8, 'colsample_bytree':0.8, 'min_child_weight':1,
              'gamma':0, 'silent':1, 'objective':'reg:linear', 'eval_metric':'rmse'}
    xgb1 = XGBregressor(params1)
    params2 = {'booster':'gblinear', 'alpha':0,# for gblinear, delete this line if change back to gbtree
               'eta':0.1, 'max_depth':2, 'subsample':1, 'colsample_bytree':1, 'min_child_weight':1,
              'gamma':0, 'silent':1, 'objective':'reg:linear', 'eval_metric':'rmse'}
    xgb2 = XGBregressor(params2)
    RF = RandomForestRegressor(n_estimators=500, max_features=0.2)
    ETR = ExtraTreesRegressor(n_estimators=500, max_features=0.3, max_depth=None)
    Ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=15),n_estimators=200)
    GBR = GradientBoostingRegressor(n_estimators=200,max_depth=5,max_features=0.5)
    LR =LinearRegression()

    params_lgb = {'objective':'regression','metric':'rmse',
              'learning_rate':0.05,'max_depth':-1,'sub_feature':0.7,'sub_row':1,
              'num_leaves':15,'min_data':30,'max_bin':20,
              'bagging_fraction':0.9,'bagging_freq':40,'verbosity':0}
    lgb1 = LGBregressor(params_lgb)

    E = Ensemble(5, xgb2, [xgb1,lgb1,RF,ETR,Ada,GBR])
    prediction = E.fit_predict(trainDf, testDf)
    output = pd.read_csv('test.csv')
    output = output[['id']]
    output['price_doc'] = prediction
    output.to_csv(r'Ensemble\Submission_Stack.csv',index=False)

    # corr = pd.concat([pd.DataFrame(S_train),trainDf['price_doc']],1).corr() # extract correlation
    # 1: 2434 2: 2421
