/*
* Software Name : SYRROCA
* Version: 1.0
* SPDX-FileCopyrightText: Copyright (c) 2021 Orange
* SPDX-License-Identifier: BSD-3-Clause
*
* This software is distributed under the BSD 3-Clause "New" or "Revised" License,
* the text of which is available at https://spdx.org/licenses/BSD-3-Clause.html
* or see the "license.txt" file for more details.
*
* Author: Alessio Diamanti
*/


# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:44:40 2020

@author: Alessio Diamanti
"""

## For reproducibility
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.layers import RepeatVector
from keras.callbacks import ModelCheckpoint
from keras import Sequential
from keras.layers import  LSTM
import re
from keras.callbacks import EarlyStopping
from keras.layers.core import Dropout

import pre_processing
import utility_classes

def create_model(feature_size, lookback):
    #feature_size = feature_size_cpu
    #lookback = 2
    hidden_layer_size = int(feature_size * 0.8)
    hidden_layer_size2 = int(feature_size * 0.6)
    lstm_autoencoder = Sequential()
    # Encoder
    lstm_autoencoder.add(
        LSTM(hidden_layer_size, activation='elu', input_shape=(lookback, feature_size), return_sequences=True,
             name='encode1'))
    lstm_autoencoder.add(Dropout(0.2, name='dropout_encode_1'))
    lstm_autoencoder.add(LSTM(hidden_layer_size2, activation='elu', name='encode2', return_sequences=False))
    lstm_autoencoder.add(RepeatVector(lookback))
    lstm_autoencoder.add(LSTM(hidden_layer_size, activation='elu', return_sequences=True, name='dencode1'))
    lstm_autoencoder.add(Dropout(0.2, name='dropout_dencode_1'))
    lstm_autoencoder.add(LSTM(feature_size, activation='linear', return_sequences=True, name='dencode2'))
    lstm_autoencoder.summary()
    lstm_autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return lstm_autoencoder
def gen_scored_train(squaredErrorDf, layer):
    scored = pd.DataFrame()
    scored['Loss_mae'] = np.mean(squaredErrorDf, axis=1)
    plt.figure()
    ax = sns.distplot(scored['Loss_mae'], bins=6000,
                      hist_kws={"histtype": "step", "linewidth": 3,
                                "alpha": 1, "color": "g"}, hist=True, kde=False)
    ax.set_title(layer + ' MSE probability distribution')
    #ax.set_xscale('log')
    thresholds = [scored['Loss_mae'].quantile(q=0.999)]
    scored['Threshold1'] = thresholds[0]
    scored['Anomaly1'] = scored['Loss_mae'] > scored['Threshold1']
    ax.axvline(scored['Loss_mae'].quantile(q=0.999),color='red',linewidth=5)
    #    for e in thresholds:
    #        ax.axvline(e,color='blue',linewidth=5)

    return [thresholds, scored]
def LSTM_dataset_gen(regEx, dataFrameArray, lookback):
    scaler = MinMaxScaler([0, 1])
    columnToUse_X = dataFrameArray[0].filter(regex=regEx)
    columnToUse_X = columnToUse_X.filter(regex='openims')
    if regEx == 'memory':
        columnToUse_X.drop(list(columnToUse_X.filter(regex='container_memory_max_usage_bytes')), axis=1, inplace=True)
    oderedColumn_X = pre_processing.reset_Names_ca(columnToUse_X.columns)
    oderedColumn_X = [el.replace('"', '') for el in oderedColumn_X]
    columnToUse_X.columns = oderedColumn_X
    concat_df = columnToUse_X
    for dfA in dataFrameArray:
        colToUseTemp = dfA.filter(regex=regEx)  # dfA.filter(regex=regExString)
        colToUseTemp = colToUseTemp.filter(regex='openims')
        cols2 = pre_processing.reset_Names_ca(colToUseTemp.columns)
        cols2 = [el.replace('"', '') for el in cols2]
        colToUseTemp.columns = cols2
        colToUseTemp = colToUseTemp[oderedColumn_X]
        concat_df = pd.concat([concat_df, colToUseTemp], ignore_index=True, sort=False)
    feature_size_cpu = len(concat_df.columns)
    input_feature_resampeled_normalized = concat_df.values
    input_feature_resampeled_normalized = scaler.fit_transform(input_feature_resampeled_normalized)
    ## LSTM data format and rescaling
    input_data_train = input_feature_resampeled_normalized  # [len(input_feature_resampeled_normalized)-1000:len(input_feature_resampeled_normalized)]
    # lookback = 2
    X_train_A_look = []
    for i in range(len(input_data_train) - lookback - 1):
        t = []
        for j in range(0, lookback):
            t.append(input_data_train[[(i + j)], :])
        X_train_A_look.append(t)
    X = np.array(X_train_A_look)
    X_train = X.reshape(X.shape[0], lookback, feature_size_cpu)
    X_train_X = np.flip(X_train, 0)
    return [X_train_X, oderedColumn_X, scaler, feature_size_cpu]
def LSTM_dataset_gen_test(regEx, dataFrame, lookback, oderedColumn, scaler, feature_size):
    columnToUseT = dataFrame.filter(regex=regEx)  # use same columns as in training phase
    columnToUseT = columnToUseT.filter(regex='openims')
    colsTest = reset_Names(columnToUseT.columns)
    colsTest = [el.replace('"', '') for el in colsTest]
    columnToUseT.columns = colsTest
    columnToUseT = columnToUseT[oderedColumn]
    input_feature_rescaled_normalized = scaler.transform(columnToUseT)

    ##LSTM format training
    X_test_A_look = []
    for i in range(len(input_feature_rescaled_normalized) - lookback):
        t = []
        for j in range(0, lookback):
            t.append(input_feature_rescaled_normalized[[(i + j)], :])
        X_test_A_look.append(t)

    X = np.array(X_test_A_look)
    X_test = X.reshape(X.shape[0], lookback, feature_size)
    X_test_X = np.flip(X_test, 0)
    return X_test_X
def save_models(models, level):
    for idx, m in enumerate(models):
        model_json = m.to_json()
        with open('models/'+level+'/'+str(idx)+'.json', "w") as json_file:
            json_file.write(model_json)

type = "../SystemState/Cadvisor/"  # Cadvisor/ Physical/

df_arr = []
## Load file archtypes
filesType = open('../SystemState/Cadvisor/MetricsTypes/ca.txt', 'r')
### per file name-types
names = []
types = []
### per file type indexes
countersIndexes = []
gaugesIndexes = []
untypedIndexes = []
summaryIndexes = []

lines = filesType.readlines()
tempTypes = []
tempNames = []
for line in lines:
    if len(re.findall('# TYPE', line)) != 0:
        tempTypes.append(line.split('# TYPE ')[1].split(' ')[1].split('\n')[0])
        tempNames.append(line.split('# TYPE ')[1].split(' ')[0])
names.append(tempNames)
types.append(tempTypes)

## Load dfs
fileMetrics = [open(type  + 'DaysNew/' + file, 'r') for file in os.listdir(type+'DaysNew/')]
counter = 0
for file_handler in fileMetrics:  # iterates on days files
    df = pd.DataFrame()
    df = pd.read_csv(file_handler, sep=';', header=0, low_memory=False, index_col=None, error_bad_lines=False)
    df = df.select_dtypes(exclude=['object'])
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='timestamp')), axis=1, inplace=True)  # spec start
    df.drop(list(df.filter(regex='spec')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='start')), axis=1, inplace=True)
    df.drop(df.filter(regex='container_memory_mapped_file').filter(regex='dns-dep').columns, axis=1, inplace=True)
    df.drop(df.filter(regex='container_memory_cache').filter(regex='dns-dep').columns, axis=1, inplace=True)
    df = pre_processing.smooth(df, '30s')  # resampling to match sipp log frequency
    ## Metric types identification
    countersTemp = []
    gaugesTemp = []
    untypedTemp = []
    summaryTemp = []
    for j in range(0, len(
            names[counter])):  # for the Cadvisor we retained only "openIms" namespace related metrics
        if types[counter][j] == 'counter':
            currGropu = df.filter(regex=names[counter][j] + '({+|$)').columns
            if len(currGropu) != 0:  # to deal with not retained metrics during xml writing after scraping
                countersTemp.append(currGropu)
        if types[counter][j] == 'gauge':
            currGropu = df.filter(regex=names[counter][j] + '({+|$)').columns
            if len(currGropu) != 0:  # to deal with not retained metrics during xml writing after scraping
                gaugesTemp.append(currGropu)
        if types[counter][j] == 'untyped':
            currGropu = df.filter(regex=names[counter][j] + '({+|$)').columns
            if len(currGropu) != 0:  # to deal with not retained metrics during xml writing after scraping
                untypedTemp.append(currGropu)
        if types[counter][j] == 'summary':
            currGropu = df.filter(regex=names[counter][j] + '({|$|_sum|_count)').columns
            if len(currGropu) != 0:  # to deal with not retained metrics during xml writing after scraping
                summaryTemp.append(currGropu)

    ## container_fs_usage_bytes si marked as gauge but is a counter
    to_swap = [[e,idx] for idx,e in enumerate(gaugesTemp) if 'container_fs_usage_bytes' in e[0]] # to_swap[1] is the index we want to remove
    del gaugesTemp[to_swap[0][1]]
    countersTemp.append(to_swap[0][0])
    to_swap = [[e, idx] for idx, e in enumerate(gaugesTemp) if
               'memory' in e[0]]  # to_swap[1] is the index we want to remove
    for e in to_swap:
        countersTemp.append(e[0])
    for elem in summaryTemp:
        for t in range(0, len(elem)):
            df.drop(list(df.filter(regex=elem[t])), axis=1, inplace=True)
    # Counters  pre-processing
    for i in range(len(countersTemp)):
        for j in range(0, len(countersTemp[i])):
            df[countersTemp[i][j]] = df[countersTemp[i][j]].diff()
    for i in range(len(gaugesTemp)):  # tries to catch wrongly marked gauges
        for j in range(0, len(gaugesTemp[i])):
            if '_total' in gaugesTemp[i][j]:
                df[gaugesTemp[i][j]] = df[gaugesTemp[i][j]].diff()
    df = df.fillna(value=0)  # first elem will be Nan after diff
    reverse_df = df.iloc[::-1]  # reverse df for lookback
    df_arr.append(reverse_df)

##train datasets

lookback = 2

## Regular expressions

cpu = 'cpu'
memory = 'memory'
disk = '_fs_'
network = 'network'

groupsRegex = [cpu, network, memory, disk]
lstmobjts = []

for idx,e in enumerate(groupsRegex):
    X_train_x, oderedColumn_x, scaler_x, feature_size_x = LSTM_dataset_gen(e, df_arr, lookback)
    lstmobjts.append(utility_classes.lstmObjts(X_train_x, oderedColumn_x, scaler_x, feature_size_x))

# Save lstmObjts to reuse in test phase
import pickle
for idx, e in enumerate(lstmobjts):
    with open('models/pickled/ca/'+str(idx), 'wb') as file:
        pickle.dump(e,file)


groups = ['cpu','network','memory','fs']
models = []
for idx,e in enumerate(lstmobjts):
    model_x = create_model(e.feature_size_x, lookback)
    models.append(model_x)

lstm_autoencoder_history = []
for idx,m in enumerate(models):
    filepath = str(groups[idx])+"weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=10, save_best_only=True, mode='min')
    ##Early stopping
    es = EarlyStopping(monitor='loss', patience=10, mode='min')
    callbacks_list = []
    callbacks_list.append(checkpoint)
    callbacks_list.append(es)
    lstm_autoencoder_history.append(m.fit(lstmobjts[idx].X_train_x, lstmobjts[idx].X_train_x, epochs=2000,
                                            batch_size=int(lstmobjts[idx].X_train_x.shape[0] / len(df_arr)), verbose=2,
                                            callbacks=callbacks_list).history)


save_models(models, 'ca')

