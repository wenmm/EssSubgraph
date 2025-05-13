
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from torch_geometric.data import NeighborSampler
from models import SAGE_NeighSampler
from tqdm import tqdm

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import pickle

eval_metric = 'auc'
import time


sage_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
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
        
            
def main():
    parser = argparse.ArgumentParser(description='minibatch_gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str)
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

    nlabels = 2
    #networks_name = ['BioGrid','CPDB','pcnet','pcnet','pcnet','pcnet','string']
  
    for n in range(len(dataset)):
        data = dataset[n]
        data.adj_t = data.adj_t.to_symmetric()


        model_dir = prepare_folder("dep_pc50_version2_test_{}".format(n), args.model)
        split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
        train_idx = split_idx['train'].to(device)
        data = data.to(device)
            

        x = data.x
        print(x)
        x = (x-x.mean(0))/x.std(0)
        data.x = x
        if data.y.dim()==2:
            data.y = data.y.squeeze(1)        

            #split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
    
            data = data.to(device)

        train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[25, 10], batch_size=1024, shuffle=True, num_workers=12)

        layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=2048, shuffle=False, num_workers=12)        

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


        for epoch in range(1, args.epochs+1):
            loss = train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv)
            losses, out = test(layer_loader, model, data, split_idx, device, no_conv)
            train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss

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
        print("preds_train",preds_train)
        y_train, y_valid, y_test = data.y[data.train_mask], data.y[data.valid_mask], data.y[data.test_mask]

        print("y_train",y_train)
        print("preds_train",preds_train)
        print("y_valid",y_valid)
        print("preds_valid",preds_valid)

        train_auc = evaluator.eval(y_train, preds_train)['auc']
        valid_auc = evaluator.eval(y_valid, preds_valid)['auc']
        test_auc = evaluator.eval(y_test, preds_test)['auc']
        print('train_auc:',train_auc)
        print('valid_auc:',valid_auc)
        print('test_auc:',test_auc)

        train_acc = evaluator1.eval(y_train, preds_train)['acc']
        valid_acc = evaluator1.eval(y_valid, preds_valid)['acc']
        test_acc = evaluator1.eval(y_test, preds_test)['acc']
        print('train_acc:',train_acc)
        print('valid_acc:',valid_acc)
        print('test_acc:',test_acc)

        train_prauc = evaluator_prauc.eval(y_train, preds_train)['prauc']
        valid_prauc = evaluator_prauc.eval(y_valid, preds_valid)['prauc']
        test_prauc = evaluator_prauc.eval(y_test, preds_test)['prauc']
        print('train_prauc:',train_prauc)
        print('valid_prauc:',valid_prauc)
        print('test_prauc:',test_prauc)


        train_sepcificity_sensitivity_mcc_f1 = evaluator_sepcificity_sensitivity_mcc_f1.eval(y_train, preds_train)['sepcificity_sensitivity_mcc_f1']
        valid_sepcificity_sensitivity_mcc_f1 = evaluator_sepcificity_sensitivity_mcc_f1.eval(y_valid, preds_valid)['sepcificity_sensitivity_mcc_f1']
        test_sepcificity_sensitivity_mcc_f1 = evaluator_sepcificity_sensitivity_mcc_f1.eval(y_test, preds_test)['sepcificity_sensitivity_mcc_f1']
        print('train_sepcificity_sensitivity_mcc_f1:',train_sepcificity_sensitivity_mcc_f1)
        print('valid_sepcificity_sensitivity_mcc_f1:',valid_sepcificity_sensitivity_mcc_f1)
        print('test_sepcificity_sensitivity_mcc_f1:',test_sepcificity_sensitivity_mcc_f1)


        
        preds = out_[data.test_mask].cpu().numpy()
        print(out.cpu().numpy())

            #np.save('./model_files/dep_{}/{}/preds.npy'.format(args.network, args.model), out.cpu().numpy())
        

            


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))


