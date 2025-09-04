from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

from utils.utils import prepare_folder
from torch_geometric.data import NeighborSampler
from models import SAGE_NeighSampler
from tqdm import tqdm

from utils.utils import prepare_folder
from utils.evaluator import Evaluator

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


import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef,roc_auc_score, confusion_matrix,roc_curve, auc, precision_recall_curve, accuracy_score, f1_score
try:
    import torch
except ImportError:
    torch = None   



sage_neighsampler_parameters = {'lr':0.01
            , 'num_layers':3
            , 'hidden_channels':128
            , 'dropout':0.5
            , 'batchnorm': False
            , 'l2':5e-7
            }


def train(epoch, train_loader, model, data, train_idx, optimizer, device, loss_op, no_conv=False):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(data.x[n_id], adjs)
        loss = loss_op(out, data.y[n_id[:batch_size]].float())
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size)

    pbar.close()
    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def test(layer_loader, model, data, split_idx, device,  loss_op, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.inference(data.x, layer_loader, device)
#     out = model.inference_all(data)
    y_pred = out
    
    losses = dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        node_id = node_id.to(device)
        losses[key] = loss_op(out[node_id], data.y[node_id].float()).item()
            
    return losses, y_pred

@torch.no_grad()
def inference_test(layer_loader, model, data, device, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.inference(data.x, layer_loader, device)
#     out = model.inference_all(data)
    y_pred = out
                
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
    
    args = parser.parse_args()
    print(args)
    
    no_conv = False
  
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    dataset = load_obj(args.dataset)

    nlabels = 2
    AUC = np.zeros(shape=(1,5))
    AUPR = np.zeros(shape=(1, 5))

    for i_ in range(len(dataset)):
        data = dataset[i_]
        data.adj_t = data.adj_t.to_symmetric()

        for i in range(1):
            for cv_run in range(5):
                model_dir = prepare_folder("subgraph2_{}_{}_{}".format(args.networks, i, cv_run), args.model)
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
                loss_op = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(device))


                for epoch in range(1, args.epochs+1):
                    loss = train(epoch, train_loader, model, data, train_idx, optimizer, device, loss_op, no_conv)
                    losses, out = test(layer_loader, model, data, split_idx, device, loss_op, no_conv)
                    train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']
                    print(f'Epoch: {epoch:02d}, '
                            f'Train: {train_loss:.3f}, ')

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

                preds_train, preds_valid, preds_test = out_[data.train_mask], out_[data.valid_mask], out_[data.test_mask]

                y_train, y_valid, y_test = data.y[data.train_mask], data.y[data.valid_mask], data.y[data.test_mask]


                train_auc = evaluator.eval(y_train, preds_train)['auc']
                valid_auc = evaluator.eval(y_valid, preds_valid)['auc']
                test_auc = evaluator.eval(y_test, preds_test)['auc']


                #train_acc = accuracy_score(y_train.detach().cpu().numpy(), preds_train.detach().cpu().numpy())
                #valid_acc = accuracy_score(y_valid.detach().cpu().numpy(), preds_valid.detach().cpu().numpy())
                #test_acc = accuracy_score(y_test.detach().cpu().numpy(), preds_test.detach().cpu().numpy())

                train_prauc = evaluator_prauc.eval(y_train, preds_train)['prauc']
                valid_prauc = evaluator_prauc.eval(y_valid, preds_valid)['prauc']
                test_prauc = evaluator_prauc.eval(y_test, preds_test)['prauc']

                precision, recall, _ = precision_recall_curve(y_test.cpu().numpy(), preds_test.cpu().numpy())
                prauc = auc(recall,precision)

                fpr, tpr, _ = roc_curve(y_true=y_test.cpu().numpy(), y_score=preds_test.cpu().numpy(), pos_label=None)
                roc_auc = auc(x=fpr, y=tpr)

                #data2save = [precision, recall, prauc, fpr, tpr, roc_auc]

                AUC[i][cv_run], AUPR[i][cv_run] = roc_auc, prauc
                #data2save = [precision, recall, prauc, fpr, tpr, roc_auc]
                #file = open('new_subgraph2_{}_{}_{}_data_to_plot.pkl'.format(args.networks, i, cv_run),'wb')
                #pickle.dump(data2save, file)
                #file.close()


                print('Round--%d CV--%d  AUC: %.5f, AUPR: %.5f' % (i, cv_run + 1, AUC[i][cv_run], AUPR[i][cv_run]))
            print('Round--%d Mean AUC: %.5f, Mean AUPR: %.5f' % (i, np.mean(AUC[i, :]), np.mean(AUPR[i, :])))
        print('%s 10 rounds for 5CV-- Mean AUC: %.4f, Mean AUPR: %.4f' % (args.model, AUC.mean(), AUPR.mean()))


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

    

