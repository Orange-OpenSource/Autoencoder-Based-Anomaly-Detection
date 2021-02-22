import networkx as nx
import matplotlib.pyplot as plt

# Generate the new state label. Returns the current label and updates silently dictMapping
def generate_label(node_name,dictMapping):
    # global countLabels
    # global dictMapping
    if not node_name in dictMapping and not node_name == 'Nominal':
        dictMapping.update({node_name:'S' + str(len(dictMapping))})
    return dictMapping[node_name]

#
def draw_networx_graph(G, graph,fname,to_div,sign,prog,pad,pad_leg,dictMapping):
    """
    Plots network graph with networkx
    Keyword arguments:
    G               The graph to plot
    graph           Pydot graph to extract metadata
    fname           File name to save to
    to_div          Quotient to compute percentage/per-mille
    sign            "‰" or "%"
    prog            Graphviz program to compute node position
    pad             Offset from node center to position node name and frequency inside node circle
    pad_leg         Offset to position the legend
    dictMapping     Dictionary with labels mapping
    """
    nodesG_old = G.nodes
    nodesG = [node.replace('__', ' & ') for node in G.nodes]
    mappingN = dict(zip([node for node in G.nodes], nodesG))
    G = nx.relabel_nodes(G, mappingN)
    plt.figure(figsize=(9.6, 5.952))
    pos = nx.nx_agraph.graphviz_layout(G, prog=prog)
    pos3Nom = {'Nominal': (pos['Nominal'][0], pos['Nominal'][1] + pad)}
    pos2 = {elem: (pos[elem][0], pos[elem][1] - pad) for idx, elem in enumerate(pos)}
    pos3 = {elem: (pos[elem][0], pos[elem][1] + pad) for idx, elem in enumerate(pos) if idx != 0}
    node_labels = [generate_label(node,dictMapping) for node in list(G.nodes)]
    mappingNom = {nodesG[0]: node_labels[0]}
    mapping = dict(zip(nodesG[1:], node_labels[1:]))
    mappingXL = dict(
        zip(nodesG,
            [str(round(n.get('xlabel') / to_div, 2)) + sign for n in graph.get_nodes() if n.get_name() in nodesG_old]))
    weights = []
    for e in list(G.edges):
        src = e[0]
        dst = e[1]
        to_append = graph.get_edge(src.replace(' & ', '__'), dst.replace(' & ', '__'))[0].get_penwidth()
        print(src,dst, to_append)
        weights.append(to_append)
    print(weights)
    for idx, n in enumerate(G.nodes):
        if idx == 0:
            nx.draw_networkx_nodes(G, pos=pos, nodelist=[n], label=dictMapping[n] + ' ' + n.replace('__', ' & '),
                                   node_size=2200, node_color='#FFFFFF', edgecolors='#000000')
        else:
            nx.draw_networkx_nodes(G, pos=pos, nodelist=[n], label=dictMapping[n] + ' ' +
                                   n.replace('__', ' & ').upper().replace('NETWORK','NET').replace('MEMORY', 'MEM'),
                                   node_size=2200, node_color='#FFFFFF',edgecolors='#000000')
        nx.draw_networkx_labels(list(G.nodes)[0], pos3Nom, mappingNom, font_size=14, font_color='#008000')
        nx.draw_networkx_labels(list(G.nodes)[:1], pos3, mapping, font_size=14, font_color='#FF0000')
        nx.draw_networkx_labels(G, pos2, mappingXL, font_size=11)
    nx.draw_networkx_edges(G, pos=pos, width=weights, connectionstyle='arc3, rad = 0.2', min_target_margin=26,
                           label=weights, min_source_margin=10)
    lgd = plt.legend(bbox_to_anchor=(0.5, pad_leg), loc='lower center', prop=dict(weight='bold', size=9), fontsize=10,
                     handlelength=0, handletextpad=0, fancybox=True, ncol=2)
    for item in lgd.legendHandles:
        item.set_visible(False)
    plt.savefig(fname + ".png", format="PNG", dpi=600, bbox_extra_artists=[lgd], bbox_inches='tight')
    return dictMapping

# Plots network graph with networkx. Just a commodity method to plot bigger circle for the training (long per-mille string)
def draw_networx_graph_train(G, graph,fname,to_div,sign,prog,pad,pad_leg,dictMapping):
    nodesG_old = G.nodes
    nodesG = [node.replace('__', ' & ') for node in G.nodes]
    mappingN = dict(zip([node for node in G.nodes], nodesG))
    G = nx.relabel_nodes(G, mappingN)
    plt.figure(figsize=(9.6,5.952))
    pos = nx.nx_agraph.graphviz_layout(G, prog=prog)  # nx.spring_layout(G)
    pos3Nom = {'Nominal':(pos['Nominal'][0], pos['Nominal'][1] + pad)}
    pos2 = {elem: (pos[elem][0], pos[elem][1] - pad) for idx,elem in enumerate(pos)}
    pos3 = {elem: (pos[elem][0], pos[elem][1] + pad) for idx,elem in enumerate(pos) if idx != 0}

    node_labels = [generate_label(node,dictMapping) for node in list(G.nodes)]
    mappingNom = {nodesG[0]: node_labels[0]}
    mapping = dict(zip(nodesG[1:], node_labels[1:]))

    mappingXL = dict(
        zip(nodesG, [str(round(n.get('xlabel')/to_div,2))+sign  for n in graph.get_nodes() if n.get_name() in nodesG_old ]))
    weights = []
    for e in list(G.edges):
        src = e[0]
        dst = e[1]
        weights.append(graph.get_edge(src.replace(' & ', '__'), dst.replace(' & ', '__'))[0].get_penwidth())
    for idx, n in enumerate(G.nodes):
        if idx == 0:
            nx.draw_networkx_nodes(G, pos=pos, nodelist=[n], label=dictMapping[n] + ' ' + n.replace('__', ' & '),
                                   node_size=3000, node_color='#FFFFFF', edgecolors='#000000')
        else:
            nx.draw_networkx_nodes(G, pos=pos, nodelist=[n], label=dictMapping[n]+' '+
                    n.replace('__', ' & ').upper().replace('NETWORK','NET').replace('MEMORY','MEM'),
                                node_size=2200, node_color='#FFFFFF', edgecolors='#000000' )
        nx.draw_networkx_labels(list(G.nodes)[0], pos3Nom, mappingNom, font_size=14, font_color='#008000')
        nx.draw_networkx_labels(list(G.nodes)[:1], pos3, mapping, font_size=14, font_color='#FF0000')
        nx.draw_networkx_labels(G, pos2, mappingXL, font_size=11)
    nx.draw_networkx_edges(G, pos=pos, width=weights, connectionstyle='arc3, rad = 0.2',min_target_margin = 26,label=weights,min_source_margin=10 )
    lgd = plt.legend(bbox_to_anchor=(0.5, pad_leg), loc='lower center', prop=dict(weight='bold',size=9),fontsize=10,handlelength=0, handletextpad=0, fancybox=True,ncol=2)
    for item in lgd.legendHandles:
        item.set_visible(False)
    plt.savefig(fname+".png", format="PNG", dpi=600, bbox_extra_artists=[lgd], bbox_inches='tight')
    return dictMapping
