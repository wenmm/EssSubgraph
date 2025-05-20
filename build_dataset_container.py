import torch_geometric.transforms as T


import pandas as pd
import json
import numpy as np
from functools import reduce
import torch
from torch_geometric.utils import remove_self_loops, to_undirected

from typing import Optional, Callable, List
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from numpy import random
import sys
import scipy.io
import pandas as pd
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import cluster, datasets


def generate_5CV_set(drivers,nondrivers,randseed):
    """
    Generate 5CV splits.
    :param drivers: List of canonical driver genes(positive samples)
    :param nondrivers: List of nondriver genes(negative samples)
    :param randseed: Random seed
    :return: 5CV splits sorted in a dictionary
    """
    # StratifiedKFold
    X, y = drivers + nondrivers, np.hstack(([1]*len(drivers), [0]*len(nondrivers)))
    skf = StratifiedKFold(n_splits=8,shuffle=True,random_state=randseed)
    X_5CV = {}
    cv_idx=1
    for train, test in skf.split(X, y):
        # train/test sorts the sample indices in X list.
        # For each split, we should convert the indices in train/test to names
        train_set=[]
        test_set=[]
        for i in train:
            train_set.append(X[i])
        for i in test:
            test_set.append(X[i])


        X_5CV['train_%d' % cv_idx] = torch.tensor(train_set)
        X_5CV['test_%d' % cv_idx] = torch.tensor(test_set)
        cv_idx += 1
    return X_5CV




delimiter = " "
net_name = "./data/string_net.txt"
#net_name = "/home/hwen6/database/string_db/string_v9.1.txt"
#net_name = "/home/hwen6/gongju/one_net_subnetwork/train_valid_G.csv"
#network_name = net_name.split("/")[-1].split("_")[0]
network_name = "string2"

# net_names = ["./data/string_net.txt",
#              "./data/BIOGRID_net.txt",
#              "./data/CPDB_net.txt",
#              "./data/HumanNet_net.txt",
#              "./data/IREF_net.txt",
#              "./data/PathwayCommons_net.txt",
#              "./data/pcnet_net.txt"]




graphs = [pd.read_csv(net_name, delimiter=delimiter, header=None)]
for G in graphs:
    if G.shape[1] < 3:
        G[2] = pd.Series([1.0] * len(G))

labels = []
#label_names = ["./data/mouse_gene_labels.json"]
label_names = ["./data/dep_gene_labels.json"]

for label_name in label_names:
    with open(label_name, "r") as f:
        labels.append(json.load(f))


node_sets = [np.union1d(G[0].values, G[1].values) for G in graphs]
union = reduce(np.union1d, node_sets)

weights = torch.FloatTensor([1.0 for G in graphs])

masks = torch.FloatTensor([np.isin(union, nodes) for nodes in node_sets])
masks = torch.t(masks)

mapper = {name: idx for idx, name in enumerate(union)}

for idx, name in enumerate(union):
    print(idx,name)


e_lst = pd.read_table(filepath_or_buffer='./data/dep_common_essential_genes', sep='\t', header=None, index_col=None,
                  names=['essential'])

#e_lst = pd.read_table(filepath_or_buffer='./data/Mouse_essential_genes', sep='\t', header=None, index_col=None,
#                  names=['essential'])

e_lst = e_lst['essential'].values.tolist()

# Nonessential genes (negative samples)
ne_lst = pd.read_table(filepath_or_buffer='./data/dep_non_essential_genes', sep='\t', header=None, index_col=None, names=['nonessential'])

#ne_lst = pd.read_table(filepath_or_buffer='./data/Mouse_viable_genes', sep='\t', header=None,
#               index_col=None, names=['nonessential'])
ne_lst = ne_lst['nonessential'].values.tolist()

e_idx = [mapper[i] for i in e_lst if i in mapper and i in labels[0]]
ne_idx = [mapper[i] for i in ne_lst if i in mapper  and i in labels[0]]


n = 1
k_sets_net = dict()
for k in np.arange(0,10): # Randomly generate 5CV splits for ten times
    k_sets_net[k] = []
    randseed = (k+1)%100+(k+1)*5
    cv = generate_5CV_set(e_idx,ne_idx,randseed)
    for cv_idx in np.arange(1,6):
        a = cv["train_%d" % cv_idx]
        b = cv["test_%d" % cv_idx]
        random.shuffle(a)
        random.shuffle(b)
        test_mask = b
        train_mask = a[:int(len(cv["train_1"])/10*9)]
        valid_mask = a[int(len(cv["train_1"])/10*9):]
        k_sets_net[k].append((train_mask, valid_mask, test_mask))
        n += 1


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
#gene_feature = pd.read_csv('/home/hwen6/public_data/TCGA/XENA_pancancer/tcga_RSEM_gene_fpkm_gname_pc150.csv', sep=',',index_col=0)
#gene_feature = pd.read_csv('/home/hwen6/gongju/DeepHE/data/gene_dna_pep_network_feature.csv', sep=',',index_col=0)
gene_feature = pd.read_csv('/home/hwen6/database/gene_expression_prediction/cancer_full_expression_pc50.csv', sep=',',index_col=0)
#gene_feature = pd.read_csv('/home/hwen6/gongju/DeepHE/data/gene_dna_pep_feature_gname_uniq.csv', sep=',',index_col=0)
gene_feature_index = pd.DataFrame(gene_feature,index=union).fillna(0)
feat_raw = scaler.fit_transform(np.abs(gene_feature_index))

final_labels = []

for curr_labels in labels:

# Remove nodes from labels not in `self.union`
    labels = {node: labels_ for node, labels_ in curr_labels.items() if node in union}
    labels_mh = labels.values()
    labels_mh = pd.DataFrame(labels_mh, index=labels.keys())
    labels_mh = labels_mh.reindex(union)
    #labels_mh = torch.FloatTensor(labels_mh.values)
    final_labels.append(labels_mh.values)

x = torch.tensor(feat_raw, dtype=torch.float).contiguous()

y = torch.tensor(final_labels[0], dtype=torch.long)
#edge_index = torch.tensor(edge_index, dtype=torch.int64).contiguous()
edge_type = torch.tensor(weights, dtype=torch.float)

pyg_graphs = []

G[[0, 1]] = G[[0, 1]].applymap(lambda node: mapper[node])
edge_index = torch.LongTensor(G[[0, 1]].values.T)
weights = torch.FloatTensor(G[2].values)
edge_index = torch.LongTensor(G[[0, 1]].values.T)

# Remove existing self loops and add self loops from `union` nodes,
# updating `weights` accordingly
edge_index, weights = remove_self_loops(edge_index, edge_attr=weights)
edge_index, weights = to_undirected(edge_index, edge_attr=weights)
union_idxs = list(range(len(union)))
self_loops = torch.LongTensor([union_idxs, union_idxs])
edge_index = torch.cat([edge_index, self_loops], dim=1)
weights = torch.cat([weights, torch.Tensor([1.0] * len(union))])

for i in range(10):
    for cv_run in range(5):
        train_mask, valid_mask, test_mask = k_sets_net[i][cv_run]

        # Create PyG `Data` object

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=weights, y=y)
        pyg_graph.edge_weight = weights
        pyg_graph.num_nodes = len(union)

        pyg_graph.train_mask = train_mask
        pyg_graph.valid_mask = valid_mask
        pyg_graph.test_mask = test_mask


        pyg_graph = T.ToSparseTensor(remove_edge_index=True)(pyg_graph)

        pyg_graph.k_sets_net = k_sets_net

        pyg_graphs.append(pyg_graph)

import pickle
with open('20250520_test_cancer_full_expression_pc50_{}.pkl'.format(network_name), 'wb') as f:
    pickle.dump(pyg_graphs, f, pickle.HIGHEST_PROTOCOL)