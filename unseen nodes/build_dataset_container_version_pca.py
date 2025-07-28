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
network_name = "string_pca"

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


import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
import matplotlib.pyplot as plt
from sklearn import preprocessing

#pca = PCA(n_components=50)

#pca_data = gene_feature.drop(['sample'], axis=1)

#pca_data = pd.DataFrame(pca_data,index=union).fillna(0)
#reduced_matrix = pca.fit_transform(pca_data)
#gene_feature = pd.read_csv('./data/cancer_full_expression_pc50.csv', sep=',',index_col=0)
#reduced_matrix = pd.DataFrame(gene_feature)
gene_df = pd.read_csv('cancer_full_expression.tsv', sep='\t')
#reduced_matrix.index = gene_feature['sample']


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


union_new = []
for m in gene_df['sample']:
     if m in union:
          union_new.append(m)

all_data_f = gene_df[gene_df['sample'].isin(union_new)]
all_data_f.index = all_data_f['sample']

all_data_f = pd.DataFrame(all_data_f,index=union).fillna(0)
all_data = all_data_f.drop(['sample'], axis=1)

G_raw = G.copy()
G[[0, 1]] = G[[0, 1]].applymap(lambda node: mapper[node])
edge_index = torch.LongTensor(G[[0, 1]].values.T)
weights = torch.FloatTensor(G[2].values)
edge_index = torch.LongTensor(G[[0, 1]].values.T)

edge_index, weights = remove_self_loops(edge_index, edge_attr=weights)
edge_index, weights = to_undirected(edge_index, edge_attr=weights)
union_idxs = list(range(len(union)))
self_loops = torch.LongTensor([union_idxs, union_idxs])
edge_index = torch.cat([edge_index, self_loops], dim=1)
weights = torch.cat([weights, torch.Tensor([1.0] * len(union))])

pyg_graphs = []
for i in range(2):
    for cv_run in range(5):
        all_data1 = all_data.copy()
        gene_df = pd.read_csv('cancer_full_expression.tsv', sep='\t')
        train_mask, valid_mask, test_mask = k_sets_net_unbalanced[i][cv_run]

        train_valid_idx_list = set(train_mask.tolist() + valid_mask.tolist()) 
        train_valid_gnames = {gname for gname, idx in mapper.items() if idx in train_valid_idx_list}

        if len(train_valid_gnames) != 0:


            train_valid_gene_df = gene_df[gene_df['sample'].isin(train_valid_gnames)]


            
            pca = PCA(n_components=50)

            train_valid_pca_data = train_valid_gene_df.drop(['sample'], axis=1)
            scaler = preprocessing.StandardScaler().fit(train_valid_pca_data)
            Xn = scaler.transform(train_valid_pca_data)


            


        #     #scaler = StandardScaler().fit(train_valid_pca_data)
        #     #train_scaled = scaler.transform(train_valid_pca_data)

        #     #all_scaled = scaler.transform(all_data)
            if len(train_valid_pca_data) != 0:
                 pca.fit(Xn)
        #         print(len(train_valid_pca_data))

                 #reduced_matrix = pca.fit(train_valid_pca_data)

                
                 print("all_data", all_data1)
                 all_data1 = scaler.transform(all_data1)
                 all_pca = pca.transform(all_data1)
                 print("all_pca", all_pca)

                 scaler = preprocessing.MinMaxScaler()

                 all_a = pd.DataFrame(all_pca)
                 all_a.index = all_data_f['sample']
                 #gene_feature_index = pd.DataFrame(all_a).reindex(union, fill_value=0)
                 feat_raw = scaler.fit_transform(np.abs(all_a))
                

                 x = torch.tensor(feat_raw, dtype=torch.float).contiguous()

                
                #edge_index = torch.tensor(edge_index, dtype=torch.int64).contiguous()

                 

                # Create PyG Data object

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