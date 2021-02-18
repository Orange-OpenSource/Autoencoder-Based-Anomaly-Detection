##Import utility classes
from utility_classes import lstmObjts
from utility_classes import predictRes
import pre_processing
import logging
import os
import pickle
import pandas as pd
import  numpy as np
import re
from keras.models import model_from_json

## Loads training AEs models as long as models weights
def load_models():
    models_ca = []
    models_phy = []
    for i in range(4):
        json_file = open('models/ca/'+str(i)+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        models_ca.append(loaded_model)
        json_file_phy = open('models/phy/' + str(i) + '.json', 'r')
        loaded_model_json_phy = json_file_phy.read()
        json_file_phy.close()
        loaded_model_phy = model_from_json(loaded_model_json_phy)
        models_phy.append(loaded_model_phy)
    models_ca[0].load_weights('Weights/CADVISORLast/cpu.hdf5')  ####cpu : no_max.hdf5
    models_ca[1].load_weights('Weights/CADVISORLast/network.hdf5')  ####cpu : no_max.hdf5
    models_ca[2].load_weights('Weights/CADVISORLast/memory.hdf5')  ####cpu : no_max.hdf5
    models_ca[3].load_weights('Weights/CADVISORLast/fs.hdf5')  ####cpu : no_max.hdf5
    models_phy[0].load_weights('Weights/PHYSICALLast/cpu.hdf5')  ####cpu : no_max.hdf5
    models_phy[1].load_weights('Weights/PHYSICALLast/network.hdf5')  ####cpu : no_max.hdf5
    models_phy[2].load_weights('Weights/PHYSICALLast/memory.hdf5')  ####cpu : no_max.hdf5
    models_phy[3].load_weights('Weights/PHYSICALLast/fs.hdf5')  ####cpu : no_max.hdf5
    return [models_ca, models_phy]

# Utility function to flatten 3D arrays in 2d ones
def flatten(X):
    '''
    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, 0,]
    return (flattened_X)

# generate the scored df for the training phase
def gen_scored_train(squaredErrorDf):
    scored = pd.DataFrame()
    scored['Loss_mae'] = np.mean(squaredErrorDf, axis=1)
    thresholds = [scored['Loss_mae'].quantile(q=0.999)] ##we use 0.999 as through fit we cannot find a well fitting unimodal function, so threshold placement has to be done "visually"
    scored['Threshold1'] = thresholds[0]
    scored['Anomaly1'] = scored['Loss_mae'] > scored['Threshold1']
    return [thresholds, scored]

# generate the scored df for the test phase
def gen_scored(xShape, squaredErrorDf, thresholdS):
    scored = pd.DataFrame(index=range(0, xShape))
    scored['Loss_mae'] = np.mean(squaredErrorDf, axis=1)
    scored['Threshold1'] = thresholdS
    scored['Anomaly1'] = scored['Loss_mae'] > scored['Threshold1']
    # scored['Threshold2'] = thresholdS[1]
    # scored['Anomaly2'] = scored['Loss_mae'] > scored['Threshold2']
    # scored['Threshold3'] = thresholdS[2]
    # scored['Anomaly3'] = scored['Loss_mae'] > scored['Threshold3']
    return scored


# prepear the dataset in the format LSTMs accepts
def lstm_dataset_gen_test(regEx, dataFrame, lookback, oderedColumn, scaler, type):

    ###print(columnToUseT.columns.values)
    if type == 'ca':
        columnToUseT = dataFrame.filter(regex=regEx)  # use same columns as in training phase
        columnToUseT = columnToUseT.filter(regex='openims')
        ###print(columnToUseT)
        colsTest = reset_Names_ca(columnToUseT.columns)
        colsTest = [el.replace('"', '') for el in colsTest]
        columnToUseT.columns = colsTest
        columnToUseT = columnToUseT[oderedColumn] #takes desired columns and re-order according to training order
        ###print('-----------**********')
        ###print(columnToUseT.columns)
    else:
        columnToUseT = dataFrame.filter(regex=regEx)  # use same columns as in training phase
        colsTest = reset_Names_phy2(columnToUseT.columns.values)
        #colsTest = [el.replace('"', '') for el in colsTest]
        ###print(colsTest)
        columnToUseT.columns = colsTest
        columnToUseT = columnToUseT[oderedColumn]  # use same columns as in training phase
        ###print(columnToUseT.shape)
    input_feature_rescaled_normalized = scaler.transform(columnToUseT)

    ##LSTM format training
    X_test_A_look = []
    for i in range(len(input_feature_rescaled_normalized) - lookback):
        t = []
        for j in range(0, lookback):
            t.append(input_feature_rescaled_normalized[[(i + j)], :])
        X_test_A_look.append(t)

    X = np.array(X_test_A_look)
    #    ##print(len(X))
    X_test = X.reshape(X.shape[0], lookback, len(oderedColumn))
    X_test_X = np.flip(X_test, 0)
    return X_test_X

def reset_Names_phy(columns):
    #columnsNew = []
    for idx,col in enumerate(columns):
        if re.search('cali3606066a66f', col) is not None:
            columns[idx] = col.replace('cali3606066a66f', 'cali203e397a51e')
        elif re.search('calid6c429c7d10',col) is not None:
            columns[idx] = col.replace('calid6c429c7d10', 'cali6e9c3ba3209')
        elif re.search('calie416ff25c02',col) is not None:
            columns[idx] = col.replace('calie416ff25c02', 'cali2375d59d0fe')#
        elif re.search('cali3dedcfa9ee3',col) is not None:
            columns[idx] = col.replace('cali3dedcfa9ee3', 'cali6d7ce0f49eb')
        elif re.search('califd4e2eb44ca',col) is not None:
            columns[idx] = col.replace('califd4e2eb44ca', 'calid1fbb161cbb')
        # elif re.search('cali',col) is not None:
        #     columns[idx] = re.sub('cali(.+?)*"', 'calixxx', 'node_network_transmit_packets_total{device="cali0ecf3cd1604"}')
    return columns

def reset_Names_phy2(columns):
    #columnsNew = []
    for idx,col in enumerate(columns):
        if re.search('cali203e397a51e', col) is not None:
            columns[idx] = col.replace('cali203e397a51e', 'cali3606066a66f')
        elif re.search('cali6e9c3ba3209',col) is not None:
            columns[idx] = col.replace('cali6e9c3ba3209', 'calid6c429c7d10')
        elif re.search('cali2375d59d0fe',col) is not None:
            columns[idx] = col.replace('cali2375d59d0fe', 'calie416ff25c02')#
        elif re.search('cali6d7ce0f49eb',col) is not None:
            columns[idx] = col.replace('cali6d7ce0f49eb', 'cali3dedcfa9ee3')
        elif re.search('calid1fbb161cbb',col) is not None:
            columns[idx] = col.replace('calid1fbb161cbb', 'califd4e2eb44ca')
        # elif re.search('cali',col) is not None:
        #     columns[idx] = re.sub('cali(.+?)*"', 'calixxx', 'node_network_transmit_packets_total{device="cali0ecf3cd1604"}')
    return columns

def reset_Names_ca(columns):
    ###print('AAAAA')
    columnsNew = []
    for col in columns:
        ##print(col)
        # col= 'container_memory_mapped_file{container="",container_name="",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b",image="",name="",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"} 3.989504e+06 1581501170989'
        # col2 = 'container_memory_mapped_file{container="POD",container_name="POD",id="/kubepods/besteffort/pod3067e0c2-81d0-47a6-bdc3-2a458ca24b3b/77e9f81f120a4cd1106add9fe98523d1537ce2f576490fb58ba711496f6e0b38",image="k8s.gcr.io/pause:3.1",name="k8s_POD_dns-deployment-6d486c4fb6-slsq6_openims_3067e0c2-81d0-47a6-bdc3-2a458ca24b3b_4",namespace="openims",pod="dns-deployment-6d486c4fb6-slsq6",pod_name="dns-deployment-6d486c4fb6-slsq6"} 0 1581501168012'
        colNew = col.split('{')[0]
        # colNew = colNew+'{'+ col.split('pod_name="')[1].split('-')[0]+','+col.split('container=')[1].split(',')[0]+'}'
        if col.split('container=')[1].split(',')[0] != '"POD"' and col.split('container=')[1].split(',')[0] != '""':  # to contour errors on pod.yaml
            colNew = colNew + '{' + col.split('pod="')[1].split('-')[0] + ',' + \
                     col.split('pod="')[1].split('-')[0]
        else:
            colNew = colNew + '{' + col.split('pod="')[1].split('-')[0] + ',' + \
                     col.split('container=')[1].split(',')[0]
        ## take in account interface for network related metrics
        if len(col.split('interface="')) != 1:
            colNew = colNew + ',' + col.split('interface="')[1].split('"')[0]
        if len(col.split('failure_type="')) != 1:
            colNew = colNew + ',' + col.split('failure_type="')[1].split('"')[0]
        if len(col.split('scope="')) != 1:
            colNew = colNew + ',' + col.split('scope="')[1].split('"')[0]
        ## take in account interface for network related metrics
        if len(col.split('device="')) != 1:
            colNew = colNew + ',' + col.split('device="')[1].split('"')[0]


        colNew = colNew + '}'

        columnsNew.append(colNew)
    return columnsNew


#logging.basicConfig(level=numeric_level, ...)

models_ca, models_phy = load_models()
to_test = 'pcscfCpuInc_1hx10new' #pcscfCpuInc_1hx10new hssCpuInc_1hx15New overloadLac1312_22_03 pcktLoss16_03
file_ca = [open('../SystemState/Cadvisor/'+to_test+'/' + file, 'r') for file in os.listdir('../SystemState/Cadvisor/'+to_test+'/')]# hssCpuInc_1hx15New pcscfCpuInc_1hx10new
file_phy = [open('../SystemState/Physical/'+to_test+'/' + file, 'r') for file in os.listdir('../SystemState/Physical/'+to_test+'/')]
ca_dataset =  pre_processing.load_dataset(file_ca[0], 'ca','../SystemState/Cadvisor/MetricsTypes/') # ca_dataset is reversed
phy_dataset = pre_processing.load_dataset(file_phy[0], 'phy','../SystemState/Physical/MetricsTypes/') #phy_dataset is reversed

## RegEx used to filter for group of resources
groupsRegex_ca = ['cpu', 'network', 'memory', '_fs_']
groupsRegex_phy = ['node_cpu|node_context_switch|node_temp', 'node_network|node_netstat|node_sockstat', 'node_edac|node_vm_stat|node_memory', 'node_disk|node_filesystem|node_filefd']

# Load some training information into lstmObj_x objets
##  - X_train_x is the training dataset ready to be used as LSTM-AEs input
##  - feature_size_x is the number of used features
##  - oderedColumn_x is the odered list of feature names
##  - scaler_x is the sclaler used to re-scale training dataset
lstmObj_ca = []
lstmObj_phy= []
# virtual layer
for i in range(4):
    with open('models/pickled/ca/' + str(i), 'rb') as file:
        unpikcled =  pickle.load(file)
        lstmObj_ca.append(unpikcled)
#physical layer
for i in range(4):
    with open('models/pickled/phy/' + str(i), 'rb') as file:
        unpikcled =  pickle.load(file)
        lstmObj_phy.append(unpikcled)


## Compute scoreds, errorDfSquare and predicted for the TRAINING phase for each AEs.
## predictedsResTrain_ca is the list for virtual layer and predictedsResTrain_phy is the list for the physical layer )
##
##       - scored: DataFrame containing following columns:
##          -'Loss_mae': MSE
#           -'Threshold1': Computed threshold
#           -'Anomaly1': True if the sample is an anomaly
##       - errorDfSquare: matrix containing squared errors per-feature and per-timestamp
##       - predicted: the predicted tensor produced in the training phase

predictedsResTrain_ca = []
for idx, e in enumerate(lstmObj_ca):
    predicted = models_ca[idx].predict(e.X_train_x)
    train_retrans_x = (flatten(e.X_train_x))
    reconstructed_train_retrans_x = (flatten(predicted))
    errorDfSquare_train_x = pd.DataFrame(np.power(train_retrans_x - reconstructed_train_retrans_x, 2),
                                        columns=lstmObj_ca[idx].oderedColumn_x)
    thresh_x , scored_train_x = gen_scored_train(errorDfSquare_train_x)
    # scored_test_x.set_index(df2.index[0:df2.shape[0] - 2], inplace=True)
    predictedsResTrain_ca.append(predictRes(scored_train_x, errorDfSquare_train_x, predicted))

predictedsResTrain_phy = []
for idx, e in enumerate(lstmObj_phy):
    predicted = models_phy[idx].predict(e.X_train_x)
    train_retrans_x = (flatten(e.X_train_x))
    reconstructed_train_retrans_x = (flatten(predicted))
    errorDfSquare_train_x = pd.DataFrame(np.power(train_retrans_x - reconstructed_train_retrans_x, 2),
                                        columns=lstmObj_phy[idx].oderedColumn_x)
    thresh_x ,scored_train_x = gen_scored_train(errorDfSquare_train_x)
    # scored_test_x.set_index(df2.index[0:df2.shape[0] - 2], inplace=True)
    predictedsResTrain_phy.append(predictRes(scored_train_x, errorDfSquare_train_x, predicted))

## Generate test input tensors for each AE

lstmDatasetsTest_ca = []
for idx, e in enumerate(groupsRegex_ca):
    X_test_x = lstm_dataset_gen_test(e, ca_dataset, 2, lstmObj_ca[idx].oderedColumn_x, lstmObj_ca[idx].scaler_x,'ca')
    lstmDatasetsTest_ca.append(X_test_x)

lstmDatasetsTest_phy = []
for idx, e in enumerate(groupsRegex_phy):
    X_test_x = lstm_dataset_gen_test(e, phy_dataset, 2, lstmObj_phy[idx].oderedColumn_x, lstmObj_phy[idx].scaler_x, 'phy')
    lstmDatasetsTest_phy.append(X_test_x)


## Compute scoreds, errorDfSquare and predicted for the TEST phase for each AEs.
# Details about obj variables is given for the Training 
predictedsRes_ca = []
predictedsRes_phy = []

for idx, e in enumerate(lstmDatasetsTest_ca):
    predicted = models_ca[idx].predict(e)  # [:1140]
    test_retrans_x = (flatten(lstmDatasetsTest_ca[idx]))  # [:1140] [:420]
    reconstructed_test_retrans_x = (flatten(predicted))
    # fig = plt.figure()
    # plt.plot(reconstructed_test_retrans_x)
    # fig.savefig('reconstructed_test_retrans_x',dpi=500)
    errorDfSquare_test_x = pd.DataFrame(np.power(test_retrans_x - reconstructed_test_retrans_x, 2),
                                        columns=lstmObj_ca[idx].oderedColumn_x)
    scored_test_x = gen_scored(e.shape[0], errorDfSquare_test_x,
                               predictedsResTrain_ca[idx].scored['Threshold1'][0])  # 1140 e.shape[0]
    # scored_test_x.set_index(df2.index[0:df2.shape[0] - 2], inplace=True)
    predictedsRes_ca.append(predictRes(scored_test_x, errorDfSquare_test_x, predicted))

for idx, e in enumerate(lstmDatasetsTest_phy):
    predicted = models_phy[idx].predict(e)
    test_retrans_x = (flatten(lstmDatasetsTest_phy[idx]))
    reconstructed_test_retrans_x = (flatten(predicted))
    errorDfSquare_test_x = pd.DataFrame(np.power(test_retrans_x - reconstructed_test_retrans_x, 2),
                                        columns=lstmObj_phy[idx].oderedColumn_x)
    scored_test_x = gen_scored(e.shape[0], errorDfSquare_test_x,
                               predictedsResTrain_phy[idx].scored['Threshold1'][0])
    # scored_test_x.set_index(df2.index[0:df2.shape[0] - 2], inplace=True)
    predictedsRes_phy.append(predictRes(scored_test_x, errorDfSquare_test_x, predicted))
