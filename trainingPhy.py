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
import os
from sklearn.preprocessing import MinMaxScaler
from keras.layers import  RepeatVector
from keras.callbacks import ModelCheckpoint
from keras import Sequential
from keras.layers import LSTM
import re
from keras.callbacks import EarlyStopping
from keras.layers.core import Dropout

import pre_processing
import utility_classes

def create_model(feature_size, lookback):
    hidden_layer_size = int(feature_size * 0.8)

    hidden_layer_size2 = int(feature_size * 0.6)
    lstm_autoencoder = Sequential()
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
    thresholds = [scored['Loss_mae'].quantile(q=0.999)]
    scored['Threshold1'] = thresholds[0]
    scored['Anomaly1'] = scored['Loss_mae'] > scored['Threshold1']
       return [thresholds, scored]
def dropNonImsInterfaces(df):
    cali_features = df.filter(regex='cali(.+?)*')
    toPurge = []
    for e in cali_features.columns:
        if not ('calie416ff25c02' in e or 'calid6c429c7d10' in e or 'cali3606066a66f' in e or 'cali3dedcfa9ee3' in e or 'califd4e2eb44ca' in e):
            toPurge.append(e)
    for e in df.columns:
        if 'eno2' in e or 'eno3' in e or 'eno4' in e:
            toPurge.append(e)
    df.drop(toPurge, axis=1, inplace=True)
    return df
def LSTM_dataset_gen(regEx, dataFrameArray, lookback):
    scaler = MinMaxScaler([0, 1])
    columnToUse_X = dataFrameArray[0].filter(regex=regEx)
    columnToUse_X.drop(list(columnToUse_X.filter(regex='max')), axis=1, inplace=True)
    columnToUse_X.drop(list(columnToUse_X.filter(regex='min')), axis=1, inplace=True)
    columnToUse_X.drop(list(columnToUse_X.filter(regex='error')), axis=1, inplace=True)
    columnToUse_X.drop(list(columnToUse_X.filter(regex='node_network_info')), axis=1, inplace=True)
    columnToUse_X.drop(list(columnToUse_X.filter(regex='node_netstat_Icmp6_OutMsgs')), axis=1, inplace=True)
    columnToUse_X.drop(list(columnToUse_X.filter(regex='node_network_mtu_bytes')), axis=1, inplace=True)
    columnToUse_X.drop(list(columnToUse_X.filter(regex='node_network_iface_id')), axis=1, inplace=True)
    columnToUse_X.drop(list(columnToUse_X.filter(regex='node_netstat_Udp6_OutDatagrams')), axis=1, inplace=True)
    columnToUse_X.drop(list(columnToUse_X.filter(regex='node_netstat_Udp6_InDatagrams')), axis=1, inplace=True)
    columnToUse_X.drop(list(columnToUse_X.filter(regex='node_network_iface_link')), axis=1, inplace=True)
    columnToUse_X.drop(list(columnToUse_X.filter(regex='node_sockstat_UDP_inuse')), axis=1, inplace=True)
    if regEx == 'node_network|node_netstat|node_sockstat':
        columnToUse_X = dropNonImsInterfaces(columnToUse_X)
    concat_df = columnToUse_X
    for dfA in dataFrameArray[1:]:
        colToUseTemp = dfA[columnToUse_X.columns]  # dfA.filter(regex=regExString)
        # colToUseTemp.drop(list(colToUseTemp.filter(regex='max')), axis=1, inplace=True)
        # colToUseTemp.drop(list(colToUseTemp.filter(regex='min')), axis=1, inplace=True)
        # colToUseTemp.drop(list(colToUseTemp.filter(regex='error')), axis=1, inplace=True)

        concat_df = pd.concat([concat_df, colToUseTemp], ignore_index=True, sort=False)
    feature_size_x = len(concat_df.columns)
    input_feature_resampeled_normalized = concat_df.values
    input_feature_resampeled_normalized = scaler.fit_transform(input_feature_resampeled_normalized)
    ## LSTM data format and rescaling
    input_data_train = input_feature_resampeled_normalized
    X_train_A_look = []
    for i in range(len(input_data_train) - lookback - 1):
        t = []
        for j in range(0, lookback):
            t.append(input_data_train[[(i + j)], :])
        X_train_A_look.append(t)
    X = np.array(X_train_A_look)
    X_train = X.reshape(X.shape[0], lookback, feature_size_x)
    X_train_X = np.flip(X_train, 0)
    return [X_train_X, columnToUse_X.columns, scaler, feature_size_x]

type = "../SystemState/Physical/"  # Cadvisor/ Physical/
df_arr = []

## Load file archtypes
filesType = open('../SystemState/Physical/MetricsTypes/phy.txt', 'r')

### per file name-types
typesFileNames = []
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
    ## Common pre-processing
    # Timestamp, Unnamed and obj dropping
    df = df.select_dtypes(exclude=['object'])
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='timestamp')), axis=1, inplace=True)  # spec start
    df.drop(list(df.filter(regex='spec')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='start')), axis=1, inplace=True)

    #### FILTER FOR MEMORY STABILITY ####
    to_filter = ['node_memory_Inactive_bytes', 'node_memory_SReclaimable_bytes', 'node_memory_Inactive_file_bytes',
                 'node_memory_Active_file_bytes', 'node_memory_DirectMap4k_bytes','node_memory_Active_bytes',
                 'node_memory_DirectMap2M_bytes','node_memory_Slab_bytes','node_memory_Shmem_bytes']

    [df.drop(list(df.filter(regex=e)), axis=1, inplace=True) for e in to_filter]

    df = pre_processing.smooth(df, '30s')  # resampling to match sipp log frequency

    ## Metric types identification
    countersTemp = []
    gaugesTemp = []
    untypedTemp = []
    summaryTemp = []
    for j in range(0, len(names[counter])):
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

    ###############
    # Manual adjustement for wrong typed metrics
    ###############

    ## all node_netstat are marked as untyped but instead are counter
    r = re.compile("node_netstat")
    newlist = list(filter(r.match, [e.values[0] for e in untypedTemp]))
    [untypedTemp.remove(e) for e in newlist]
    [countersTemp.append([e]) for e in newlist]

    # drop summary and untyped
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

lookback = 2

cpu = 'node_cpu|node_context_switch|node_temp'
memory = 'node_edac|node_vm_stat|node_memory'
disk = 'node_disk|node_filesystem|node_filefd'
network = 'node_network|node_netstat|node_sockstat'

groupsRegex = [cpu, network, memory, disk]
lstmobjts = []

for idx,e in enumerate(groupsRegex):
    X_train_x, oderedColumn_x, scaler_x, feature_size_x = LSTM_dataset_gen(e, df_arr, lookback)
    lstmobjts.append(utility_classes.lstmObjts(X_train_x, oderedColumn_x, scaler_x, feature_size_x))


import pickle
for idx, e in enumerate(lstmobjts):
    with open('models/pickled/phy/'+str(idx), 'wb') as file:
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
    lstm_autoencoder_history = m.fit(lstmobjts[idx].X_train_x, lstmobjts[idx].X_train_x, epochs=2000,
                                            batch_size=int(lstmobjts[idx].X_train_x.shape[0] / len(df_arr)), verbose=2,
                                            callbacks=callbacks_list).history

def save_models(models, level):
    for idx, m in enumerate(models):
        model_json = m.to_json()
        with open('models/'+level+'/'+str(idx)+'.json', "w") as json_file:
            json_file.write(model_json)

save_models(models, 'phy')
