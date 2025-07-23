from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from utils.evaluator import Evaluator

from utils.utils import prepare_folder
from torch_geometric.data import NeighborSampler
#from models import SAGE_NeighSampler, GAT_NeighSampler, GATv2_NeighSampler
from tqdm import tqdm

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd
import pickle
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef,roc_auc_score, confusion_matrix,roc_curve, auc, precision_recall_curve, accuracy_score, f1_score
try:
    import torch
except ImportError:
    torch = None   


    

class SAGE_NeighSampler(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(SAGE_NeighSampler, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout

        self.transform = nn.Sequential(
             nn.Linear(hidden_channels, hidden_channels),
             nn.ReLU(),
             nn.Linear(hidden_channels, 2)
         )
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()        
        
        
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers-1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
            
                
        return self.transform(x).log_softmax(dim=-1)
    
    '''
    subgraph_loader: size = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=**, shuffle=False,
                                  num_workers=12)
    You can also sample the complete k-hop neighborhood, but this is rather expensive (especially for Reddit). 
    We apply here trick here to compute the node embeddings efficiently: 
       Instead of sampling multiple layers for a mini-batch, we instead compute the node embeddings layer-wise. 
       Doing this exactly k times mimics a k-layer GNN.  
    '''
    
    def inference_all(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.batchnorm: 
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return self.transform(x).log_softmax(dim=-1)
    
    def inference(self, x_all, layer_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, ncols=80)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in layer_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm: 
                        x = self.bns[i](x)
                xs.append(x)

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return self.transform(x_all).log_softmax(dim=-1)
    

sage_neighsampler_parameters = {'lr':0.01
            , 'num_layers':3
            , 'hidden_channels':128
            , 'dropout':0.5
            , 'batchnorm': True
            , 'l2':5e-7
            }


def train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv=False):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(data.x[n_id], adjs)
        loss = F.nll_loss(out, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size)

    pbar.close()
    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def test(layer_loader, model, data, split_idx, device, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.inference(data.x, layer_loader, device)
#     out = model.inference_all(data)
    y_pred = out.exp()  # (N,num_classes)   
    
    losses = dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        node_id = node_id.to(device)
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
            
    return losses, y_pred

@torch.no_grad()
def inference_test(layer_loader, model, data, device, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.inference(data.x, layer_loader, device)
#     out = model.inference_all(data)
    y_pred = out.exp()  # (N,num_classes)   
                
    return y_pred

def load_obj(name):
    """
    Load dataset from pickle file.
    :param name: Full pathname of the pickle file
    :return: Dataset type of dictionary
    """
    with open(name, 'rb') as f:
        return pickle.load(f)
      

    


#dataset = XYGraphP1(root='./', name='xydata', transform=T.ToSparseTensor())
def main():
    parser = argparse.ArgumentParser(description='minibatch_gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='sage_neighsampler')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--networks', type=str, default='string')
    
    args = parser.parse_args()
    print(args)
    
    no_conv = False
  
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #dataset = XYGraphP1(root='./', name='xydata', transform=T.ToSparseTensor())
    #dataset = load_obj('/home/hwen6/gongju/one_net_subnetwork/dep_essential_task_subgraph_{}.pkl'.format(args.networks))
    dataset = load_obj(args.dataset)

    nlabels = 3
    AUC = np.zeros(shape=(1,5))
    AUPR = np.zeros(shape=(1, 5))

    for i_ in range(len(dataset)):
        data = dataset[i_]
        data.adj_t = data.adj_t.to_symmetric()

        for i in range(1):
            for cv_run in range(5):
                model_dir = prepare_folder("{}_{}_{}".format(args.networks, i, cv_run), args.model)
                data.train_mask, data.valid_mask, data.test_mask = data.k_sets_net[i][cv_run]
                split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
                train_idx = split_idx['train'].to(device)
                data = data.to(device)
            

                x = data.x
                x = (x-x.mean(0))/x.std(0)
                data.x = x
                if data.y.dim()==2:
                    data.y = data.y.squeeze(1)        
        
                    #split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
            
                    data = data.to(device)

                train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[30,25, 10], batch_size=1024, shuffle=True, num_workers=12)

                layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12)        
        
                if args.model == 'sage_neighsampler':
                    para_dict = sage_neighsampler_parameters
                    model_para = sage_neighsampler_parameters.copy()
                    model_para.pop('lr')
                    model_para.pop('l2')
                    model = SAGE_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)

                print(f'Model {args.model} initialized')


                model.reset_parameters()
                optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
                min_valid_loss = 1e8
                loss_op = nn.BCEWithLogitsLoss()


                for epoch in range(1, args.epochs+1):
                    loss = train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv)
                    losses, out = test(layer_loader, model, data, split_idx, device, no_conv)
                    train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

                    if valid_loss < min_valid_loss:
                        min_valid_loss = valid_loss
                        torch.save(model.state_dict(), model_dir +'model.pt')

                    if epoch % args.log_steps == 0:
                        print(f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_loss:.3f}%, '
                            f'Valid: {100 * valid_loss:.3f}% '
                            f'Test: {100 * test_loss:.3f}%')
                        
                out_ = inference_test(layer_loader, model, data, device, no_conv)
                evaluator = Evaluator('auc')
                evaluator1 = Evaluator('acc')
                evaluator_prauc = Evaluator('prauc')
                evaluator_sepcificity_sensitivity_mcc_f1 = Evaluator('sepcificity_sensitivity_mcc_f1')

                preds_train, preds_valid, preds_test = out_[data.train_mask], out_[data.valid_mask], out_[data.test_mask]
                y_train, y_valid, y_test = data.y[data.train_mask], data.y[data.valid_mask], data.y[data.test_mask]

                evaluator = Evaluator('auc')
                evaluator1 = Evaluator('acc')
                evaluator_prauc = Evaluator('prauc')
                evaluator_sepcificity_sensitivity_mcc_f1 = Evaluator('sepcificity_sensitivity_mcc_f1')

                preds_train, preds_valid, preds_test = out_[data.train_mask], out_[data.valid_mask], out_[data.test_mask]

                y_train, y_valid, y_test = data.y[data.train_mask], data.y[data.valid_mask], data.y[data.test_mask]

                train_auc = evaluator.eval(y_train, preds_train)['auc']
                valid_auc = evaluator.eval(y_valid, preds_valid)['auc']
                test_auc = evaluator.eval(y_test, preds_test)['auc']


                train_acc = evaluator1.eval(y_train, preds_train)['acc']
                valid_acc = evaluator1.eval(y_valid, preds_valid)['acc']
                test_acc = evaluator1.eval(y_test, preds_test)['acc']


                train_prauc = evaluator_prauc.eval(y_train, preds_train)['prauc']
                valid_prauc = evaluator_prauc.eval(y_valid, preds_valid)['prauc']
                test_prauc = evaluator_prauc.eval(y_test, preds_test)['prauc']

                

                precision, recall, _ = precision_recall_curve(y_test.cpu().numpy(), preds_test[:, 1].cpu().numpy())
                prauc = auc(recall,precision)
                

                fpr, tpr, _ = roc_curve(y_true=y_test.cpu().numpy(), y_score=preds_test[:, 1].cpu().numpy(), pos_label=None)
                roc_auc = auc(x=fpr, y=tpr)

                #data2save = [precision, recall, prauc, fpr, tpr, roc_auc]

                AUC[i][cv_run], AUPR[i][cv_run] = test_auc, test_prauc
                #data2save = [precision, recall, test_prauc, fpr, tpr, test_auc]
                #file = open('subgraph2_{}_{}_{}_data_to_plot.pkl'.format(args.networks, i, cv_run),'wb')
                #pickle.dump(data2save, file)
                #file.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

    
