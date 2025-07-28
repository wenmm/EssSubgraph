import pandas as pd
import networkx as nx
import numpy as np
import h5py, os, sys
np.set_printoptions(precision=1, suppress=True)
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath('/home/hwen6/gongju/EMOGI/EMOGI'))
import gcnPreprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils.convert import from_networkx

import torch
from torch_geometric.data import Data
import pickle

# params
test_ratio = .3
pos_distr = (50, 60) # mean and variance
neg_distr = (45, 60) # mean and variance
num_features = 150
NETWORK_TYPE = 'random'

for node_number in [10000, 15000, 20000, 25000,  30000]:
    for edge_number in [200000,400000,600000,800000,1000000,1200000, 1400000, 1600000, 1800000]:
        G = nx.gnm_random_graph(node_number, edge_number)
        pyg_graph = from_networkx(G)
        nx.write_edgelist(G, './simulation/network_{}.edgelist_{}'.format(node_number, edge_number))
        x = list(np.random.choice(list(nx.nodes(G)), size=10000).reshape(-1, 10))
        insert_positions = [list(i) for i in x]
        network = G
        with open('./simulation/insert_positions_{}_{}.txt'.format(node_number, edge_number), 'w') as f:
            count = 1
            f.write('# subnetwork insert positions. Each row corresponds to a subnetwork and each column to a position in it.\n')
            for m in insert_positions:
                f.write('Subnetwork {}: '.format(count))
                for i in m:
                    f.write('{}\t'.format(i))
                    f.write('\n')
                    count += 1

        num_nodes = network.number_of_nodes()
        all_motif_nodes = np.array(insert_positions).reshape(-1)
        features = np.random.normal(loc=neg_distr[0], scale=np.sqrt(neg_distr[1]),
                                size=(num_nodes, num_features))
        features[all_motif_nodes] = np.random.normal(loc=pos_distr[0], scale=np.sqrt(pos_distr[1]),
                                              size=(all_motif_nodes.shape[0], num_features))
        not_motif_mems = np.array([i for i in np.arange(num_nodes) if not i in all_motif_nodes])
        features[features < 0] = 0 # no negative gene expression

        y = np.array([1 if i in all_motif_nodes else 0 for i in np.arange(num_nodes)]).reshape(-1, 1)
        mask = np.ones(num_nodes, dtype=np.uint)


        y_train, train_mask, y_test, test_mask = gcnPreprocessing.train_test_split(y, mask, 0.25)

        y = y.squeeze()

        train_mask = train_mask.astype(np.bool)
        test_mask = test_mask.astype(np.bool)
        node_names = list(np.arange(num_nodes))

        data = Data(x=torch.tensor(features, dtype=torch.float), y=torch.tensor(y, dtype=torch.float), edge_index=pyg_graph.edge_index, tr_mask=train_mask, te_mask = test_mask, node_names=node_names)


        fname ="./simulation/" +'simulation_network_nodes_{}_edges_{}.pkl'.format(node_number, edge_number)
        with open(fname, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


                       


