##Import utility classes
from utility_classes import lstmObjts
import pre_processing
import logging
import os
import pickle
import pandas as pd
import  numpy as np
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
#logging.basicConfig(level=numeric_level, ...)
models_ca, models_phy = load_models()
to_test = 'pcscfCpuInc_1hx10new' #pcscfCpuInc_1hx10new hssCpuInc_1hx15New overloadLac1312_22_03 pcktLoss16_03
file_ca = [open('../SystemState/Cadvisor/'+to_test+'/' + file, 'r') for file in os.listdir('../SystemState/Cadvisor/'+to_test+'/')]# hssCpuInc_1hx15New pcscfCpuInc_1hx10new
file_phy = [open('../SystemState/Physical/'+to_test+'/' + file, 'r') for file in os.listdir('../SystemState/Physical/'+to_test+'/')]
ca_dataset =  pre_processing.load_dataset(file_ca[0], 'ca','../SystemState/Cadvisor/MetricsTypes/') # ca_dataset is reversed
phy_dataset = pre_processing.load_dataset(file_phy[0], 'phy','../SystemState/Physical/MetricsTypes/') #phy_dataset is reversed
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