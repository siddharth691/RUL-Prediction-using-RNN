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
from __future__ import print_function
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

#Reading the data
data = pd.read_csv('hist.csv')

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
def prepare_engine(engine, scaler):
    
    #Removing unit number and time in cycles column
    engine = engine.iloc[:,2:len(engine.columns)]
    engine = engine.values
    
    #Scaling using trained scaler
    engine = engine.reshape(engine.shape[0], engine.shape[1])
    engine = scaler.transform(engine)

    return engine

#Assigning the same prediction variable RUL to each engine 
unit_2RUL = data.loc[:,['unit number', 'RUL']].groupby('unit number').count().to_dict()['RUL']
data.loc[:,'RUL_correct'] = data.loc[:,'unit number']
data.replace({"RUL_correct":unit_2RUL}, inplace = True)
data.drop('RUL', axis = 1, inplace= True)
data.rename(columns= {'RUL_correct':'RUL'}, inplace = True)

#Making number of rows in each test engine divisible by batch_size = 4
def delete_extra_rows(df):
    if (len(df['unit number'])%4 != 0):
        return df.iloc[0:len(df)-(len(df)%4),:]
    
    else:
        return df
data = data.groupby('unit number').apply(delete_extra_rows).reset_index(drop= True)

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


timesteps = 1
input_dim = train_x.shape[1]
batch_size = 4
size_latent = 6
size_linear_layer = 5

#Reshaping the input to be fit to be input to the LSTM
train_x = train_x.reshape(np.shape(train_x)[0], timesteps, input_dim)

inputs = Input(batch_shape=(batch_size,timesteps, input_dim))

encoded = LSTM(size_latent, return_sequences = True)(inputs)
encoded = LSTM(size_latent,return_sequences = True)(encoded)
encoded = LSTM(size_latent,return_sequences = True)(encoded)
encoded = Dense(size_linear_layer, activation = 'relu')(encoded)

decoded = Dense(size_latent, activation = 'relu')(encoded)
decoded =  LSTM(size_latent,return_sequences = True)(decoded)                                   
decoded =  LSTM(size_latent,return_sequences = True)(decoded)
decoded =  LSTM(size_latent, return_sequences = True)(decoded)
decoded = Dense(input_dim, activation = 'relu')(decoded)

sequence_autoencoder = tf.keras.models.Model(inputs, decoded)

encoder = tf.keras.models.Model(inputs,encoded)

sequence_autoencoder.compile(loss='mean_squared_error', optimizer='Adam')

history_data = sequence_autoencoder.fit(train_x, train_x, epochs=100, batch_size=batch_size, shuffle=False)


#Plotting the training loss with epochs
fig = plt.figure()
plt.plot(history_data.history['loss'])
plt.title('Training loss')
plt.xlabel('epoch')

#For training the regression model for predicting RUL

#Train data
#Calculating the embeddings which would be input to the regression classifier
#Calculating the output for each batch (RUL)

embeddings = encoder.predict(train_x, batch_size= batch_size)
start_batch = 0
mean_embeddings_train = []
output_train = []
while(start_batch < train_x.shape[0] - 1):
    mean_embeddings_train.append(list(np.mean(embeddings[start_batch: start_batch + batch_size, :], axis = 0)[0,:]))
    output_train.append(train_y[start_batch: start_batch + batch_size][0])
    start_batch += batch_size

    
#Test data
#Reshaping test data to be of same shape to be input to the encoder
test_x = test_x.reshape(np.shape(test_x)[0], timesteps, input_dim)

embeddings = encoder.predict(test_x, batch_size= batch_size)
start_batch = 0
mean_embeddings_test = []
output_test = []
while(start_batch < test_x.shape[0] - 1):
    mean_embeddings_test.append(list(np.mean(embeddings[start_batch: start_batch + batch_size, :], axis = 0)[0,:]))
    output_test.append(test_y[start_batch: start_batch + batch_size][0])
    start_batch += batch_size

#Combing training and testing embeddings and outputs for grid search Cross validation
mean_embeddings_train.extend(mean_embeddings_test)
output_train.extend(output_test)

clf = SVR(kernel = 'rbf')
parameters = {'C':[50,100], 'gamma': [0.1]}
gs = GridSearchCV(clf, parameters)
gs.fit(mean_embeddings_train, output_train)
best_params = gs.best_params_
print("Best Parameters: {}".format(gs.best_params_))
print("Best test cross validation R2 score {}".format(np.max(gs.cv_results_['mean_train_score'])))


#Fitting the regression model with best parameters
clf = SVR(kernel = 'rbf', C = best_params['C'], gamma = best_params['gamma'])
clf.fit(mean_embeddings_train, output_train)

engine_tul = {}
#Walk forward validation method for predicting the RUL for testing

for engine_no in [101, 102, 103, 104, 124, 125, 134]:
    
    #Initializing the engine tul dictionary
    engine_tul[str(engine_no)]= []
    
    
    #Reading the engine file
    file_name = 'Engine' + str(engine_no) + '.csv'
    engine = pd.read_csv(file_name)
    
    if(engine_no == 103):
        engine.loc[:,'unit number'] = 3
        
    #Preparing the engine for the prediction
    engine = prepare_engine(engine, scaler)
    
    print('\n')
    print('Current Engine: {}'.format(engine_no))
    
    
    start_batch = 0
    predicted_rul = []
    while(start_batch + batch_size < len(engine) -1):


        #Get current batch from the full engine data
        cur_engine = engine[start_batch: start_batch + batch_size,:]

        #Reshaping the current engine batch to be input to the Autoencoder
        cur_engine = cur_engine.reshape(cur_engine.shape[0], timesteps, cur_engine.shape[1])

        #Calculating the embedding vector for the current batch
        embeddings = encoder.predict(cur_engine, batch_size = batch_size)

        #Taking the mean of the embeddings for this batch
        mean_embeddings = np.mean(embeddings, axis = 0)

        #Using regression model to predict the output RUL for current batch
        cur_rul = clf.predict(mean_embeddings)

        #Appending the cur_rul to the previous batch rul vector
        predicted_rul.append(cur_rul)

        #RUL prediction at the current timestamp
        print("TULprediction at current cycle {}: {}".format(start_batch + batch_size + 1, np.mean(predicted_rul)))
        
        engine_tul[str(engine_no)].append(np.mean(predicted_rul))
        
        #Updating the start_batch
        start_batch+=batch_size


#RUL prediction
#Negative RUL means 0 RUL

print('RUL for engine 101: {}'.format(int(np.mean(engine_tul['101']) - 31)))
print('RUL for engine 102: {}'.format(int(np.mean(engine_tul['102']) - 49)))
print('RUL for engine 103: {}'.format(int(np.mean(engine_tul['103']) - 126)))
print('RUL for engine 104: {}'.format(int(np.mean(engine_tul['104']) - 106)))
print('RUL for engine 124: {}'.format(int(np.mean(engine_tul['124']) - 186)))
print('RUL for engine 125: {}'.format(int(np.mean(engine_tul['125']) - 48)))
print('RUL for engine 134: {}'.format(int(np.mean(engine_tul['134']) - 395)))
