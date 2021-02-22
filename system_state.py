import  numpy as np
from utility_classes import ErrorClass
import pydot
import re
import networkx as nx
# Takes a list of typed error classes and adds feature names in errordfsquare basing on features indexes
def names_from_indexes(errorClasses_arr_x, predictedRes): 
    for index,e in enumerate(errorClasses_arr_x):
        for clasS in e:
            patternInt = clasS.pattern
            patternString = [predictedRes[index].errorDfSquare.iloc[:,u].name for u in patternInt]
            clasS.set_patternString(patternString)

#
def get_clean_classes(scored, errorDfSquare):
    perEccr = []  # perEccr[i] list of %contribution to MSI for i-th anomaly
    timeIndexes = []  # time index of most contributing error feature
    errIndexes = []  # sorted index of most error contirubuting feature
    errorSums = []
    for q in range(0, scored.shape[0]):  # scored.shape[0]
        if scored['Anomaly1'][q]:
            timeIndexes.append(q)
            sumErr = (errorDfSquare.iloc[q, :].sum())
            perEccrTemp = [errQ / sumErr for errQ in errorDfSquare.iloc[q, :]]
            perEccr.append(perEccrTemp)
            perEccrTempSorted = np.flip(np.sort(np.asanyarray(perEccrTemp)))  # sort ascending % error
            tempSum = 0
            stopIndex = 0
            for w in range(0, len(perEccrTempSorted)):#search first stopIndex features that reaches 0.9% of total error
                tempSum += perEccrTempSorted[w]
                if tempSum >= 0.9:
                    stopIndex = w
                    errorSums.append(tempSum)
                    break
            errIndexesTemp = []
            j = 0
            while j < stopIndex + 1:
                found = np.where(perEccrTemp == perEccrTempSorted[j])[0]
                if len(found) > 1:
                    j += len(found)
                    for e in found:
                        errIndexesTemp.append(e)# look for index of the j-th perEccrTempSorted in perEccrTemp
                else:
                    errIndexesTemp.append(found[0])
                    j += 1
            errIndexes.append(errIndexesTemp)

    # Some errors are characterized by permutation of same metrics, that is
    # the same metrics characterize the deviation but the magnitude of the influence of each metric is different
    # Considering permutation we just catch type of deviation, neglecting each metric contribution
    # errorClasses will contain a pattern and the list of timestamps where this pattern happens

    errorClasses = []
    for w in range(0, len(errIndexes)):
        currClass = errIndexes[w]
        if currClass not in [c.pattern for c in errorClasses]:  # if it is a not known error class
            indexes = []  # collect indexes of this pattern class
            indexes.append(w)
            for q in range(w + 1, len(errIndexes)):
                if currClass == errIndexes[q]:
                    indexes.append(q)
            ia = np.asarray([el for el in indexes], dtype=int)
            errorClasses.append(ErrorClass(currClass, np.array(timeIndexes)[ia]))

    # here we take care of permutations
    scannedClasses = []
    usedIndex = []  # root Classes
    permUsedIndex = []  # permUsedIndex[i] := list of permutations of usedIndex[i] root class
    for w in range(0, len(errorClasses)):
        currClass = errorClasses[w].pattern
        if currClass not in scannedClasses:
            usedIndex.append(w)
            scannedClasses.append(currClass)
            sortedCurrClass = np.sort(currClass)
            permIndex = []
            for q in range(w + 1, len(errorClasses)):
                if len(errorClasses[q].pattern) == len(sortedCurrClass):
                    if np.all(np.sort(errorClasses[q].pattern) == sortedCurrClass):  # found a permutation
                        permIndex.append(q)
                        scannedClasses.append(errorClasses[q].pattern)
            permUsedIndex.append(permIndex)

    cleanClasses = []  # cleanClasses[i] contains root class and all its permutations
    for l in range(0, len(permUsedIndex)):
        if len(permUsedIndex[l]) == 0:
            cleanClasses.append(errorClasses[usedIndex[l]])
        else:
            pattern = errorClasses[usedIndex[l]].pattern
            indexesN = errorClasses[usedIndex[l]].timeIndex
            for y in range(len(permUsedIndex[l])):
                indexesN = np.append(indexesN, errorClasses[permUsedIndex[l][y]].timeIndex)
            newErrClass = ErrorClass(pattern, indexesN)
            cleanClasses.append(newErrClass)

    ## Groups frequency
    for i in range(len(cleanClasses)):
        cleanClasses[i].set_freq(len(cleanClasses[i].timeIndex) / len(timeIndexes) * 100)
    return cleanClasses

# Takes two time steps and returns classes indexes in given list
def get_classes(i,j,classList):
    i_class_index = -1
    j_class_index = -1
    for idx, e in enumerate(classList):
        if i in e.timeIndex:
            i_class_index = idx
        if j in e.timeIndex:
            j_class_index = idx
    return  [i_class_index, j_class_index]

#
def get_per_layer_errorClasses_arr(errorClasses_arr_x): # pass a copy of the array
    errorClasses_ca_train = [e.copy() for e in errorClasses_arr_x[0]] # start with cpu related classes
    current_times = [elem for c in errorClasses_arr_x[0] for elem in c.timeIndex]
    for i in range(len(errorClasses_arr_x)):
        for j in range(i+1,len(errorClasses_arr_x)):
            for errorC in errorClasses_arr_x[j]:
                for t in errorC.timeIndex:
                    if t in current_times: # t is of type i and j -> remove t from his Class in errorClasses_ca_train and add the new class

                        c1 = get_classes(t,t,errorClasses_ca_train) #retreive Class c1 with the same t
                        c1_timeIndex = errorClasses_ca_train[c1[0]].timeIndex #retreive c1 timeindexes
                        c1_StringPattern = errorClasses_ca_train[c1[0]].patternString #retreive c1 patternString

                        c2_timeIndex = errorC.timeIndex
                        c2_StringPattern = errorC.patternString

                        errorClasses_ca_train[c1[0]].set_timeIndex(
                            np.delete(c1_timeIndex, np.argwhere(c1_timeIndex == t))) #delete c1_timeIndex from c1
                        errorC.set_timeIndex(
                            np.delete(c2_timeIndex, np.argwhere(c2_timeIndex == t))) #delete c1_timeIndex from errorC

                        if len(errorClasses_ca_train[c1[0]].timeIndex) == 0: #if there is no more residual timeStep
                            errorClasses_ca_train.remove(errorClasses_ca_train[c1[0]])

                        #now we add merged class

                        errorClasses_ca_train.append(ErrorClass([],[t],c1_StringPattern+c2_StringPattern))
                    else:
                        # errorClasses_ca_train.append(errorC)
                        current_times.append(t)
                if not len(errorC.timeIndex) == 0:  # if there is residual timeStep add errorC
                    errorClasses_ca_train.append(errorC)
    to_del = []
    petternsStrings = [e.patternString for e in errorClasses_ca_train]
    for q,e in enumerate(petternsStrings):
        if q not in to_del:
            indices = [i for i, x in enumerate(petternsStrings) if x == e]
            if len(indices) != 1:
                for index in indices[1:]:
                    errorClasses_ca_train[indices[0]].timeIndex.append([el for el in errorClasses_ca_train[index].timeIndex][0])
                for t in indices[1:]:
                    to_del.append(t)
    to_del_elem = [errorClasses_ca_train[t] for t in to_del]
    [errorClasses_ca_train.remove(e)for e in to_del_elem]
    return errorClasses_ca_train

#
def reset_features_name(names,type):
    cpu = 'cpu|context_switch|temp'
    memory = 'edac|vm_stat|memory'
    disk = 'disk|filesystem|filefd'
    network = 'network|netstat|sockstat'
    new_names = []
    for item in names:
        if re.search('container_(.+?)_', item) is None: # as base for multiserver, for the moment there is no difference
            item_new_type = item.split('node_')[1]
            if re.search(cpu, item_new_type):
                item_new_type = 'CPU_server1'
            elif re.search(memory, item_new_type):
                item_new_type = 'MEM_server1'
            elif re.search(disk, item_new_type):
                item_new_type = 'FS_server1'
            else:
                item_new_type = 'NET_server1'
            new_names.append(item_new_type)
        else:
            item_new_type = re.search('container_(.+?)_',item).group(1)
            item_new_container = re.search('{(.+?),',item).group(1)#strip out differences between container level and pod level anomalies
            new_names.append(item_new_type + '_' +item_new_container)
    new_names_str = str()
    for e in sorted(list(dict.fromkeys(new_names))): #sorted is used to take in account for permutations
        new_names_str += e +'__'
    return new_names_str

#
def merge_virt_phy(errorClasses_ca_x,errorClasses_phy_x):
    errorClasses_phy_x = [e.copy() for e in errorClasses_phy_x]
    errorClasses_all_x = [e.copy() for e in errorClasses_ca_x]
    current_times = [elem for c in errorClasses_all_x for elem in c.timeIndex]

    for idx,errorC in enumerate(errorClasses_phy_x):
        for t in errorC.timeIndex:
            if t in current_times:  # t is of type i and j -> remove t from his Class in errorClasses_ca_train and add the new class
                c1 = get_classes(t, t, errorClasses_all_x)  # retreive Class c1 with the same t
                c1_timeIndex = errorClasses_all_x[c1[0]].timeIndex  # retreive c1 timeindexes
                c1_StringPattern = errorClasses_all_x[c1[0]].patternString  # retreive c1 patternString

                c2_timeIndex = errorC.timeIndex
                c2_StringPattern = errorC.patternString

                errorClasses_all_x[c1[0]].set_timeIndex(
                    np.delete(c1_timeIndex, np.argwhere(c1_timeIndex == t)))  # delete c1_timeIndex from c1
                errorC.set_timeIndex(
                    np.delete(c2_timeIndex, np.argwhere(c2_timeIndex == t)))  # delete c1_timeIndex from errorC

                if len(errorClasses_all_x[c1[0]].timeIndex) == 0:  # if there is no more residual timeStep
                    errorClasses_all_x.remove(errorClasses_all_x[c1[0]])
                # now we add merged class
                errorClasses_all_x.append(ErrorClass([], [t], c1_StringPattern + c2_StringPattern))
            else:
                # errorClasses_ca_train.append(errorC)
                current_times.append(t)
        if not len(errorC.timeIndex) == 0:  # if there is residual timeStep add errorC
            errorClasses_all_x.append(errorC)
    return errorClasses_all_x

#
def get_node_dimentions(node):
    return len(node.timeIndex)

#
def get_graph(errorClasses_arr_x,size,type): # list of errorClasses, index of the list (0 for cpu etc...), size the number of samples
    pen_width = 0.1
    timSteps_train = [eI for c in errorClasses_arr_x for eI in c.timeIndex]
    timSteps_train.sort()
    graph = pydot.Dot(graph_type='digraph')
    graph.add_node(pydot.Node("Nominal", style="filled",xlabel= (size-len(timSteps_train)),shape='doublecircle'))
    for idx,e in enumerate(errorClasses_arr_x):

        to_reset = []
        for it in e.patternString:
            to_reset.append(it)#predictedsRes.errorDfSquare.iloc[:,it].name)
        resetted = reset_features_name(to_reset,type)
        if len(graph.get_node(resetted[:-2])) == 0:
            height = get_node_dimentions(e)
            node_to_add = pydot.Node(resetted[:-2], xlabel=height)
            node_to_add.set('labeldistance', 0.5)
            graph.add_node(node_to_add) # adds node 'a'
        else:
            curr_node = graph.get_node(resetted[:-2])[0]
            curr_node.set('xlabel',curr_node.get('xlabel')+get_node_dimentions(e))
    countstrans2_train = 0
    for idx,e in enumerate(timSteps_train):
        if not idx > len(timSteps_train)-2:
            if timSteps_train[idx+1] == e+1: #transition from a deviate state to another
                classes = get_classes(e,timSteps_train[idx+1],errorClasses_arr_x)
                te_reset0 = [elem for elem in errorClasses_arr_x[classes[0]].patternString]
                te_reset1 = [elem for elem in errorClasses_arr_x[classes[1]].patternString]
                if not reset_features_name(te_reset0,type) == reset_features_name(te_reset1,type):
                    countstrans2_train = countstrans2_train + 1
                    label = reset_features_name(te_reset0,type)[:-2]+'->'+reset_features_name(te_reset1,type)[:-2]
                    edge = pydot.Edge(reset_features_name(te_reset0,type)[:-2],
                                      reset_features_name(te_reset1,type)[:-2],penwidth= pen_width,
                                      label=label)
                    if not label in  [e.get_label() for e in graph.get_edges()] :
                        graph.add_edge(edge)
                    else:
                        curr_edge = graph.get_edge(reset_features_name(te_reset0,type)[:-2],reset_features_name(te_reset1,type)[:-2])[0]
                        curr_edge.set_penwidth(curr_edge.get_penwidth()+0.2)

    for i in range(size):
        if not i in timSteps_train: # i is not anomaly
            curr_class = get_classes(i, i-1, errorClasses_arr_x)
            if not curr_class[1] == -1: #add transition towards the nominal state
                te_reset0 = [elem for elem in errorClasses_arr_x[curr_class[1]].patternString]
                label = reset_features_name(te_reset0, type)[:-2] + '->' + 'Nominal'
                edge = pydot.Edge(reset_features_name(te_reset0,type)[:-2],
                                  'Nominal',label=label,penwidth= pen_width)
                if not label in [e.get_label() for e in graph.get_edges()]:
                    graph.add_edge(edge)
                else:
                    curr_edge = graph.get_edge(reset_features_name(te_reset0, type)[:-2], 'Nominal')[0]
                    curr_edge.set_penwidth(curr_edge.get_penwidth() + 0.2)
                countstrans2_train = countstrans2_train + 1
            if i+1 in timSteps_train: # add transition from the nominal state
                to_go_to = [elem for elem in errorClasses_arr_x[get_classes(i, i+1, errorClasses_arr_x)[1]].patternString]
                label = 'Nominal' + '->' + reset_features_name(to_go_to, type)[:-2]
                edge = pydot.Edge('Nominal',reset_features_name(to_go_to,type)[:-2],label=label,penwidth= pen_width)
                if not label in [e.get_label() for e in graph.get_edges()]:
                    graph.add_edge(edge)
                else:
                    curr_edge = graph.get_edge('Nominal',reset_features_name(to_go_to,type)[:-2])[0]
                    curr_edge.set_penwidth(curr_edge.get_penwidth() + 0.2)
                countstrans2_train = countstrans2_train + 1
    return graph

#
def get_purged_grpah(grpah, cut):
    G=nx.nx_pydot.from_pydot(grpah)
    nodes = list(grpah.get_nodes())
    to_remove = []
    for idx,node in enumerate(nodes):
        # ##print(idx)
        ###print(node.get_name())
        #purge nodes we just go in from nominal and then we go back to nominal after 1 timestep
        if node.get('xlabel') <=cut :#and list(G.successors(node.get_name())) == ['Nominal'] and list(G.predecessors(node.get_name())) == ['Nominal']:
            to_remove.append(node.get_name())
            # ##print(node.get_name())
    G.remove_nodes_from(to_remove)
    return G



