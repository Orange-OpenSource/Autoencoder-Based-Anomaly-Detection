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


import os
import re
import pandas as pd


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

def load_models(): # returns models_ca[] and models_phy
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

def smooth(dfa, interval):  # '60s'
    timeRange = pd.timedelta_range(start='0 seconds', end=str((dfa.shape[0] - 1) * 5) + ' seconds', freq='5s')
    dfa['timestampo'] = timeRange
    indexer = pd.TimedeltaIndex(dfa['timestampo'], freq='5s')
    dfa.set_index(indexer, inplace=True)
    dfa_res = dfa.resample(interval).mean()
    dfa_res = dfa_res.select_dtypes(exclude=['object'])
    return dfa_res

def pre_process(df,metricsTypePath,dtype):
    df = smooth(df, '30s')  # resampling to match sipp log frequency
    if dtype == 'ca':
        filesType_ca = [open(metricsTypePath + file, 'r') for file in os.listdir(metricsTypePath)]
    else:
        filesType_ca = [open(metricsTypePath + file, 'r') for file in os.listdir(metricsTypePath)]
    ### per file name-types
    names_ca = []
    types_ca = []
    # typesFileNames.append(filesType_ca[0].name.split('/')[1].split(".")[0])
    lines_ca = filesType_ca[0].readlines()
    for line in lines_ca:
        if len(re.findall('# TYPE', line)) != 0:
            # ##print(line)
            types_ca.append(line.split('# TYPE ')[1].split(' ')[1].split('\n')[0])
            names_ca.append(line.split('# TYPE ')[1].split(' ')[0])
    ## Metric types identification
    countersTemp = []
    gaugesTemp = []
    untypedTemp = []
    summaryTemp = []
    for j in range(0, len(names_ca)):  # for the Cadvisor we retained only "openIms" namespace related metrics
        if types_ca[j] == 'counter':
            currGropu = df.filter(regex=names_ca[j] + '({+|$)').columns
            if len(currGropu) != 0:  # to deal with not retained metrics during xml writing after scraping
                countersTemp.append(currGropu)
        if types_ca[j] == 'gauge':
            currGropu = df.filter(regex=names_ca[j] + '({+|$)').columns
            if len(currGropu) != 0:  # to deal with not retained metrics during xml writing after scraping
                gaugesTemp.append(currGropu)
        if types_ca[j] == 'untyped':
            currGropu = df.filter(regex=names_ca[j] + '({+|$)').columns
            if len(currGropu) != 0:  # to deal with not retained metrics during xml writing after scraping
                untypedTemp.append(currGropu)
        if types_ca[j] == 'summary':
            currGropu = df.filter(regex=names_ca[j] + '({|$|_sum|_count)').columns
            if len(currGropu) != 0:  # to deal with not retained metrics during xml writing after scraping
                summaryTemp.append(currGropu)
    if dtype == 'phy':
        ###############
        # Manual adjustement for wrong typed metrics
        ###############
        r = re.compile("node_netstat")
        newlist = list(filter(r.match, [e.values[0] for e in untypedTemp]))
        [untypedTemp.remove(e) for e in newlist]
        [countersTemp.append([e]) for e in newlist]
    else:
        to_swap = [[e, idx] for idx, e in enumerate(gaugesTemp) if
                   'container_fs_usage_bytes' in e[0]]  # to_swap[1] is the index we want to remove
        del gaugesTemp[to_swap[0][1]]
        countersTemp.append(to_swap[0][0])

        to_swap = [[e, idx] for idx, e in enumerate(gaugesTemp) if
                   'memory' in e[0]]  # to_swap[1] is the index we want to remove

        for e in to_swap:
            countersTemp.append(e[0])
    # drop summary
    for elem in summaryTemp:
        for t in range(0, len(elem)):
            df.drop(list(df.filter(regex=elem[t])), axis=1, inplace=True)
    # for elem in untypedTemp:
    #     for t in range(0, len(elem)):
    #         df.drop(list(df.filter(regex=elem[t])), axis=1, inplace=True)

    # Counters  pre-processing
    for i in range(len(countersTemp)):
        for j in range(0, len(countersTemp[i])):
            df[countersTemp[i][j]] = df[countersTemp[i][j]].diff()
    for i in range(len(gaugesTemp)):  # tries to catch wrongly marked gauges
        for j in range(0, len(gaugesTemp[i])):
            if '_total' in gaugesTemp[i][j]:
                df[gaugesTemp[i][j]] = df[gaugesTemp[i][j]].diff()
                # ##print(gaugesTemp[i][j])
    df = df.fillna(value=0)  # first elem will be Nan after diff
    reverse_df = df.iloc[::-1]  # reverse df for lookback

    return reverse_df

def load_dataset(file_handler, dtype, metricsTypePath): #return the pre-processed dataset 'Cadvisor/MetricsTypes/'
    df = pd.read_csv(file_handler, sep=';', header=0, low_memory=False, index_col=None, error_bad_lines=False)
    ## Common pre-processing
    # Timestamp, Unnamed and obj dropping
    df = df.select_dtypes(exclude=['object'])
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='timestamp')), axis=1, inplace=True)  # spec start

    reverse_df = pre_process(df,metricsTypePath,dtype)

    return reverse_df

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

def names_from_indexes(errorClasses_arr_x, predictedRes): # get a list of typed error classes and replace features indexes in errordfsquare with feature name
    # errorClasses_arr_x = errorClasses_arr_ca_train
    # predictedRes = predictedsResTrain_ca
    for index,e in enumerate(errorClasses_arr_x):
        for clasS in e:
            patternInt = clasS.pattern
            patternString = [predictedRes[index].errorDfSquare.iloc[:,u].name for u in patternInt]
            ##print(patternString)
            clasS.set_patternString(patternString)

def reset_features_name(names,type):

    cpu = 'cpu|context_switch|temp'
    memory = 'edac|vm_stat|memory'
    disk = 'disk|filesystem|filefd'
    network = 'network|netstat|sockstat'
    new_names = []
    for item in names:
        if re.search('container_(.+?)_',item) == None: # as base for multiserver, for the moment there is no difference
        #names = ['node_context_switches_total','node_cpu_scaling_frequency_hertz{cpu="1"}', 'node_cpu_scaling_frequency_hertz{cpu="2"}', 'node_cpu_scaling_frequency_hertz{cpu="25"}']
            item_new_type = item.split('node_')[1]
            if re.search(cpu, item_new_type):
                item_new_type = 'CPU_server1'
            elif re.search(memory, item_new_type):
                item_new_type = 'MEM_server1'
            elif re.search(disk, item_new_type):
                item_new_type = 'FS_server1'
            else:
                item_new_type = 'NET_server1'
            # ##print(item_new_type)
            new_names.append(item_new_type)
        else:
        #names = ['container_cpu_system_seconds_total{pcscf,}', 'container_cpu_system_seconds_total{dns,}', 'container_cpu_usage_seconds_total{pcscf,}', 'container_cpu_system_seconds_total{dns,dns}', 'container_cpu_user_seconds_total{pcscf,}', 'container_cpu_system_seconds_total{pcscf,pcscf}']
            item_new_type = re.search('container_(.+?)_',item).group(1)
            item_new_container = re.search('{(.+?),',item).group(1)#strip out differences between container level and pod level anomalies
            new_names.append(item_new_type + '_' +item_new_container)
    new_names_str = str()
    for e in sorted(list(dict.fromkeys(new_names))): #sorted is used to take in account for permutations
        new_names_str += e +'__'
    return new_names_str

