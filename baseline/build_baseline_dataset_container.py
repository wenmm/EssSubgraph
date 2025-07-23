import pandas as pd
import numpy as np
import torch
import pickle
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import argparse, os, sys


def load_network(file_path):
    """
    Load network from file.
    :param file_path: Full pathname of the network file
    :return: net (class: pandas.DataFrame): Edges in the network, nodes (class: pandas.DataFrame): The nodes in the network
    """
    net = pd.read_table(filepath_or_buffer=file_path, header=None,
                        index_col=None, names=['source', 'target'], sep='\t')
    nodes = pd.concat([net['source'], net['target']], ignore_index=True)
    nodes = pd.DataFrame(nodes, columns=['nodes']).drop_duplicates()
    nodes.reset_index(drop=True, inplace=True)
    return net, nodes

def build_customized_feature_matrix(feat_file_lst, network_file, feat_name_lst):
    """
    Build feature matrix on your own data.
    :param feat_file_lst: List of full pathnames of feature files. Each feat_file in feat_file_lst contains two columns, i.e., gene names and feature values.
    :param network_file: Full pathname of network file
    :param feat_name_lst: List of feature names
    :return: Concatenated feature matrix with n rows(genes) and m columns(features) (class: pandas.DataFrame)
    """
    feat_dic = dict()
    # Load gene features from each feat_file
    for i in range(0, len(feat_file_lst)):
        feat_dic[feat_name_lst[i]] = pd.read_csv(feat_file_lst[i], sep=',', index_col=0)
    # Load network from file
    net, net_nodes = load_network(network_file)
    # Normalization by MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    feat_raw = scaler.fit_transform(np.abs(feat_dic[feat_name_lst[0]].reindex(net_nodes['nodes'].values.tolist(), fill_value=0)))
    # Concatenate multiple features to form one feature matrix
    if len(feat_file_lst) > 1:
        for i in range(1,len(feat_file_lst)):
            feat_raw = np.concatenate((feat_raw, scaler.fit_transform(np.abs(feat_dic[feat_name_lst[i]].reindex(net_nodes['nodes'].values.tolist(), fill_value=0)))), axis=1)

    return pd.DataFrame(feat_raw,index=net_nodes['nodes'].values.tolist(),columns=feat_name_lst)

def create_edge_index(network_file,net_features):
    """
    Convert the edges in a network into edges indexed by integer ids, which is necessary to build an object typeof torch_geometric.data.Data.
    :param network_file: Full pathname of the network file
    :param net_features (class: pandas.DataFrame): Concatenated feature matrix with n rows(genes) and m columns(features)
    :return (class: pandas.DataFrame): Edges indexed by integer ids
    """
    net, _ = load_network(network_file)
    node_df = pd.DataFrame({'name':net_features.index.values.tolist(),
                            'id':[i for i in np.arange(0,net_features.shape[0])]})
    net = pd.merge(left=net,right=node_df,how='left',left_on='source',right_on='name')
    net.columns=['source','target','sourcename','sourceid']
    net = pd.merge(left=net, right=node_df, how='left',left_on='target',right_on='name')
    net.columns=['source','target','sourcename','sourceid','targetname','targetid']
    edge_index1 = net.loc[:,['sourceid','targetid']]
    # Treat the graph as undirected graph
    edge_index2 = net.loc[:,['targetid','sourceid']]
    edge_index = pd.concat([edge_index1,edge_index2],axis=0)
    return edge_index

def generate_5CV_set(essentials,nonessentials,randseed):
    """
    Generate 5CV splits.
    :param essentials: List of canonical essential genes(positive samples)
    :param nonessentials: List of nonessential genes(negative samples)
    :param randseed: Random seed
    :return: 5CV splits sorted in a dictionary
    """
    # StratifiedKFold
    X, y = essentials + nonessentials, np.hstack(([1]*len(essentials), [0]*len(nonessentials)))
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=randseed)
    X_5CV = {}
    cv_idx=1
    for train, test in skf.split(X, y):
        # train/test sorts the sample indices in X list.
        # For each split, we should convert the indices in train/test to names
        train_set=[]
        train_label=[]
        test_set=[]
        test_label=[]
        for i in train:
            train_set.append(X[i])
            train_label.append(y[i])
        for i in test:
            test_set.append(X[i])
            test_label.append(y[i])
        X_5CV['train_%d' % cv_idx] = train_set
        X_5CV['test_%d' % cv_idx] = test_set
        X_5CV['train_label_%d' % cv_idx] = train_label
        X_5CV['test_label_%d' % cv_idx] = test_label
        cv_idx = cv_idx + 1
    return X_5CV


from sklearn.model_selection import StratifiedKFold
import numpy as np
import random


def generate_5CV_set_unbalanced_1_to_4(essentials, nonessentials, randseed, fold, trainingProp):

    np.random.seed(randseed)
    X = essentials + nonessentials
    y = np.hstack(([1] * len(essentials), [0] * len(nonessentials)))

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



import sys
network_file = sys.argv[1]
feat_name_lst = [str(i) for i in range(50)]
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
gene_feature = pd.read_csv(sys.argv[2], sep=',',index_col=0)

#gene_feature = pd.read_csv(sys.argv[3], sep=',',index_col=0)

net, net_nodes = load_network(network_file)
gene_feature_index = pd.DataFrame(gene_feature,index=net_nodes['nodes'].values.tolist()).fillna(0)
feat_raw = scaler.fit_transform(np.abs(gene_feature_index))
net_features = pd.DataFrame(feat_raw,index=net_nodes['nodes'].values.tolist(),columns=feat_name_lst)


# Concatenate multiple features to form one feature matrix


# A dataset contains the following data:
# feature: the gene feature matrix
# edge_index: graph edges for training model
# node_name: gene names
# feature_name: feature names
# label: True labels of genes (0 for negative samples and 1 for positive samples),
# k_sets: 5CV splits that randomly generated for ten times
# mask: mask for training a single model without cross-validation
dataset=dict()
dataset['feature'] = torch.FloatTensor(np.array(net_features))
dataset['node_name'] = net_features.index.values.tolist()
# Create edge_index by edges in network file
edge_index = create_edge_index(network_file,net_features)
dataset['edge_index'] = torch.LongTensor(np.array(edge_index).transpose())
dataset['feature_name'] = net_features.columns.values.tolist()

# Generate 10 rounds 5CV splits

# Canonical essential genes (positive samples)
#d_lst = pd.read_table(filepath_or_buffer='/home/hwen6/public_data/Depmap/egs_syms.csv', sep='\t', header=None, index_col=None,
#                      names=['essential'])

d_lst = pd.read_table(filepath_or_buffer='Essential_genes',sep='\t', header=None, index_col=None,names=['essential'])
d_lst = d_lst['essential'].values.tolist()

# Nonessential genes (negative samples)
nd_lst = pd.read_table(filepath_or_buffer='Non_essential_genes', sep='\t', header=None,
                       index_col=None, names=['nonessential'])
nd_lst = nd_lst['nonessential'].values.tolist()

# True labels of genes
labels = []
mask = [] # mask for training a single model without cross-validation
for g in dataset['node_name']:
    if g in d_lst:
        labels.append(1)
    else:
        labels.append(0)
    if (g in d_lst) or (g in nd_lst):
        mask.append(True)
    else:
        mask.append(False)

d_in_net = [] # Canonical essential genes in the network
nd_in_net = [] # Nonessential genes in the network
for g in dataset['node_name']:
    if g in d_lst:
        d_in_net.append(g)
    elif g in nd_lst:
        nd_in_net.append(g)

k_sets_net = dict()
for k in np.arange(0,10): # Randomly generate 5CV splits for ten times
    k_sets_net[k] = []
    randseed = (k+1)%100+(k+1)*5
    #cv = generate_5CV_set(d_in_net,nd_in_net,randseed)
    cv = generate_5CV_set_unbalanced_1_to_4(d_in_net, nd_in_net, randseed, fold = 4, trainingProp = 0.8)
    #cv = generate_5CV_set_unbalanced_test(d_in_net,nd_in_net,randseed,test_neg_ratio=4)
    for cv_idx in np.arange(1,6):
        tr_mask = [] # train mask
        te_mask = [] # test mask
        for g in dataset['node_name']:
            if g in cv['train_%d' % cv_idx]:
                tr_mask.append(True)
            else:
                tr_mask.append(False)
            if g in cv['test_%d' % cv_idx]:
                te_mask.append(True)
            else:
                te_mask.append(False)
        tr_mask = np.array(tr_mask)
        te_mask = np.array(te_mask)
        k_sets_net[k].append((tr_mask,te_mask))


dataset['label'] = torch.FloatTensor(np.array(labels))
dataset['split_set'] = k_sets_net
dataset['mask'] = np.array(mask)
# Save the dataset as pickle file, which can be used for training HGDC
with open(sys.argv[3], 'wb') as f:
    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
