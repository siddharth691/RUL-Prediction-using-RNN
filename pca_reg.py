#Code for pca + support vector regression

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import numpy
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.decomposition import PCA

data = pd.read_csv('hist.csv')
engine101 = pd.read_csv('Engine101.csv')
engine102 = pd.read_csv('Engine102.csv')
engine103 = pd.read_csv('Engine103.csv')
engine104 = pd.read_csv('Engine104.csv')
engine124 = pd.read_csv('Engine124.csv')
engine125 = pd.read_csv('Engine125.csv')
engine134 = pd.read_csv('Engine134.csv')

engine134.loc[:,'unit number'] = 34
engine103.loc[:,'unit number'] = 3

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)

    return scaler, train_scaled, test_scaled

#Preparing the engine data for predicting RUL
def prepare_engine(engine, scaler, pca):
    
    #Removing unit number and time in cycles column
    engine = engine.iloc[:,2:len(engine.columns)]
    engine = engine.values
    
    #Scaling using trained scaler
    engine = engine.reshape(engine.shape[0], engine.shape[1])
    engine = scaler.transform(engine)
    
    #Data compression using PCA
    engine = pca.transform(engine)

    return engine

#Assigning the same prediction variable RUL to each engine 
unit_2RUL = data.loc[:,['unit number', 'RUL']].groupby('unit number').count().to_dict()['RUL']
data.loc[:,'RUL_correct'] = data.loc[:,'unit number']
data.replace({"RUL_correct":unit_2RUL}, inplace = True)
data.drop('RUL', axis = 1, inplace= True)
data.rename(columns= {'RUL_correct':'RUL'}, inplace = True)

#Splitting training and test data (last 20 engines)
def get_test(df):
    if(df['unit number'].values[0]>80):
        return df

def get_train(df):
    if(df['unit number'].values[0]<=80):
        return df
        
test = data.groupby('unit number').apply(get_test).reset_index(drop = True).dropna(axis= 0)
train = data.groupby('unit number').apply(get_train).reset_index(drop = True).dropna(axis = 0)


#Removing unit number and time in cycles column
#For train
train_series_x = train.iloc[:,2:len(data.columns)-1]
train_series_y = train.iloc[:,-1]
#For test
test_series_x = test.iloc[:,2:len(data.columns)-1]
test_series_y = test.iloc[:,-1]

raw_test_x = test_series_x.values
raw_train_x = train_series_x.values

raw_test_y = test_series_y.values
raw_train_y = train_series_y.values

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(raw_train_x, raw_test_x)

#Renaming
train_x = train_scaled
test_x = test_scaled
train_y = raw_train_y
test_y = raw_test_y


#PreProcessing(Data Compression)
pca = PCA()
pca.fit(train_x)
VarianceRatio = pca.explained_variance_ratio_

#Finding number of principal components
#Retaining 99% of total variance 
TotVar = 0 
for idx, Var in enumerate(VarianceRatio):
    TotVar += Var 
    if(TotVar>=0.99): 
        break 
print('Total Retained Variance: {}'.format(TotVar)) 
print('Number of principal components {}'.format(idx))

#Again fitting pca for 14 principal components
pcaRed = PCA(n_components = 14,whiten= True)
pcaRed.fit(train_x)
#Compressing training and testing data using fitted data
train_x = pcaRed.transform(train_x)
test_x = pcaRed.transform(test_x)

#concatenating train and test for grid search
X = np.concatenate((train_x, test_x), axis = 0)
y = np.concatenate((train_y, test_y), axis = 0)

#Grid search
clf = SVR(kernel = 'rbf')
parameters = {'C':[0.01, 0.1, 1, 10, 20, 50,100, 500], 'gamma': [0.01, 0.1, 1, 10, 50, 100]}
gs = GridSearchCV(clf, parameters)
gs.fit(X, y)
best_params = gs.best_params_
print("Best Parameters: {}".format(gs.best_params_))
print("Best test cross validation R2 score {}".format(np.max(gs.cv_results_['mean_train_score'])))

#Fitting the regression model with best parameters
clf = SVR(kernel = 'rbf', C = best_params['C'], gamma = best_params['gamma'])
clf.fit(X, y)

#Preparing test engine data
engine101 = prepare_engine(engine101, scaler, pcaRed)
engine102 = prepare_engine(engine102, scaler, pcaRed)
engine103 = prepare_engine(engine103, scaler, pcaRed)
engine104 = prepare_engine(engine104, scaler, pcaRed)
engine124 = prepare_engine(engine124, scaler, pcaRed)
engine125 = prepare_engine(engine125, scaler, pcaRed)
engine134 = prepare_engine(engine134, scaler, pcaRed)

#Predicting the RUL for new engines
#Negative RUL means zero RUL
e1 = clf.predict(engine101)
e2 = clf.predict(engine102)
e3 = clf.predict(engine103)
e4 = clf.predict(engine104)
e24 = clf.predict(engine124)
e25 = clf.predict(engine125)
e34 = clf.predict(engine134)
print("Predicted RUL for engine 101: {}".format(int(np.mean(e1) - len(e1))))
print("Predicted RUL for engine 102: {}".format(int(np.mean(e2) - len(e2))))
print("Predicted RUL for engine 103: {}".format(int(np.mean(e3) - len(e3))))
print("Predicted RUL for engine 104: {}".format(int(np.mean(e4) - len(e4))))
print("Predicted RUL for engine 124: {}".format(int(np.mean(e24) - len(e24))))
print("Predicted RUL for engine 125: {}".format(int(np.mean(e25) - len(e25))))
print("Predicted RUL for engine 134: {}".format(int(np.mean(e34) - len(e34))))