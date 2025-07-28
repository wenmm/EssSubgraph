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
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import StratifiedKFold
import numpy as np
import math

def generate_5CV_set_unbalanced_names(drivers, nondrivers, randseed, fold, trainingProp):

    np.random.seed(randseed)
    X = drivers + nondrivers
    y = np.hstack(([1] * len(drivers), [0] * len(nondrivers)))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=randseed)
    X_5CV = {}
    cv_idx = 1

    for train_idx, test_idx in skf.split(X, y):
        train_genes = [X[i] for i in train_idx]
        train_labels = [y[i] for i in train_idx]

        # Separate ESS and NESS
        train_ess = [g for g, l in zip(train_genes, train_labels) if l == 1]
        train_ness_all = [g for g, l in zip(train_genes, train_labels) if l == 0]

        # Subsample NESS to match fold ratio
        n_ess = len(train_ess)
        n_ness = math.ceil(n_ess * fold)
        if n_ness <= len(train_ness_all):
            train_ness = list(np.random.choice(train_ness_all, n_ness, replace=False))
        else:
            train_ness = list(np.random.choice(train_ness_all, n_ness, replace=True))

        # Shuffle both
        np.random.shuffle(train_ess)
        np.random.shuffle(train_ness)

        # Split into training and test
        def split_list(lst, train_p):
            train_n = math.ceil(len(lst) * train_p)
            return lst[:train_n], lst[train_n:]

        ess_train, ess_test = split_list(train_ess, trainingProp)
        ness_train, ness_test = split_list(train_ness, trainingProp)

        train_all = ess_train + ness_train
        test_all = ess_test + ness_test

        np.random.shuffle(train_all)
        np.random.shuffle(test_all)

        X_5CV[f'train_{cv_idx}'] = train_all
        X_5CV[f'test_{cv_idx}'] = test_all
        cv_idx += 1

    return X_5CV




delimiter = " "
net_name = "./data/string_net.txt"
#net_name = "string_v9.1.txt"
#network_name = net_name.split("/")[-1].split("_")[0]
network_name = "string_graph"

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

label_dict = {}

with open('./data/Essential_genes','r') as f:
    for line in f:
        raw = line.rstrip().split()
        label_dict[raw[0]] = [1]

with open('./data/Non_essential_genes','r') as f:
    for line in f:
        raw = line.rstrip().split()
        label_dict[raw[0]] = [0]


node_sets = [np.union1d(G[0].values, G[1].values) for G in graphs]
union = reduce(np.union1d, node_sets)

weights = torch.FloatTensor([1.0 for G in graphs])

mapper = {name: idx for idx, name in enumerate(union)}

e_lst = pd.read_table(filepath_or_buffer='./data/Essential_genes', sep='\t', header=None, index_col=None,
                  names=['essential'])
e_lst = e_lst['essential'].values.tolist()

# Nonessential genes (negative samples)
ne_lst = pd.read_table(filepath_or_buffer='./data/Non_essential_genes', sep='\t', header=None,
               index_col=None, names=['nonessential'])
ne_lst = ne_lst['nonessential'].values.tolist()

e_idx = [mapper[i] for i in e_lst if i in mapper and i in label_dict]
ne_idx = [mapper[i] for i in ne_lst if i in mapper  and i in label_dict]


import random  
import numpy as np  

n = 1
k_sets_net_unbalanced = dict()
for k in np.arange(0,10): # Randomly generate 5CV splits for ten times
    k_sets_net_unbalanced[k] = []
    randseed = (k+1)%100+(k+1)*5
    cv = generate_5CV_set_unbalanced_names(e_idx,ne_idx,randseed, fold=4, trainingProp=0.8)
    for cv_idx in np.arange(1,6):
        a = cv["train_%d" % cv_idx].copy()
        b = cv["test_%d" % cv_idx].copy()
        random.shuffle(a)
        random.shuffle(b)
        test_mask = b
        train_mask = a[:int(len(cv["train_%d" % cv_idx])/10*9)]
        valid_mask = a[int(len(cv["train_%d" % cv_idx])/10*9):]
        k_sets_net_unbalanced[k].append((torch.tensor(train_mask), torch.tensor(valid_mask), torch.tensor(test_mask)))
        n += 1


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
gene_feature = pd.read_csv('./data/cancer_full_expression_pc50.csv', sep=',',index_col=0)
gene_feature_index = pd.DataFrame(gene_feature,index=union).fillna(0)
feat_raw = scaler.fit_transform(np.abs(gene_feature_index))
x = torch.tensor(feat_raw, dtype=torch.float).contiguous()

pos_idx = e_idx
neg_idx = ne_idx

length = len(union)
# Initialize y with zeros
y = [2] * length

# Set positive indices to 1
for i in pos_idx:
    y[i] = 1

# Set negative indices to 0
for i in neg_idx:
    y[i] = 0


y = torch.tensor(y, dtype=torch.long)
#edge_index = torch.tensor(edge_index, dtype=torch.int64).contiguous()
edge_type = torch.tensor(weights, dtype=torch.float)




pyg_graphs = []
for i in range(10):
    for cv_run in range(5):
        #gene_df = pd.read_csv('./data/cancer_full_expression.tsv', sep='\t')
        train_mask, valid_mask, test_mask = k_sets_net_unbalanced[i][cv_run]
        #print("Train mask empty?", len(train_mask) == 0)
        #print("Valid mask empty?", len(valid_mask) == 0) 
        # print("train_mask",len(train_mask), "valid_mask", len(valid_mask), "test_mask", len(test_mask)) 
        train_valid_idx_list = set(train_mask.tolist() + valid_mask.tolist()) 
        train_valid_gnames = {gname for gname, idx in mapper.items() if idx in train_valid_idx_list}

        train_valid_G = G[G[0].isin(train_valid_gnames) & G[1].isin(train_valid_gnames)]

        if len(train_valid_gnames) != 0:
            edge_type = torch.tensor(weights, dtype=torch.float)


            train_valid_G[[0, 1]] = train_valid_G[[0, 1]].applymap(lambda node: mapper[node])
            edge_index = torch.LongTensor(train_valid_G[[0, 1]].values.T)
            weights = torch.FloatTensor(train_valid_G[2].values)

            edge_index, weights = remove_self_loops(edge_index, edge_attr=weights)
            edge_index, weights = to_undirected(edge_index, edge_attr=weights)
            union_idxs = list(range(len(union)))
            self_loops = torch.LongTensor([union_idxs, union_idxs])
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            weights = torch.cat([weights, torch.Tensor([1.0] * len(union))])

        # Create PyG `Data` object

            pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=weights, y=y)
            pyg_graph.edge_weight = weights
            pyg_graph.num_nodes = len(union)

            pyg_graph.train_mask = k_sets_net_unbalanced[i][cv_run][0]
            pyg_graph.valid_mask = k_sets_net_unbalanced[i][cv_run][1]
            pyg_graph.test_mask = k_sets_net_unbalanced[i][cv_run][2]


            pyg_graph = T.ToSparseTensor(remove_edge_index=True)(pyg_graph)
            pyg_graphs.append(pyg_graph)




import pickle
with open('subgraph2_cancer_full_expression_pc50_{}.pkl'.format(network_name), 'wb') as f:
    pickle.dump(pyg_graphs, f, pickle.HIGHEST_PROTOCOL)