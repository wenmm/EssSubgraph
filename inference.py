# dataset name: XYGraphP1
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from torch_geometric.data import NeighborSampler
from models import SAGE_NeighSampler
from tqdm import tqdm

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np
import pickle


sage_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

gat_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             , 'layer_heads':[4,1]
             }

gatv2_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-6
             , 'layer_heads':[4,1]
             }


@torch.no_grad()
def test(layer_loader, model, data, device, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.inference(data.x, layer_loader, device)
#     out = model.inference_all(data)
    y_pred = out.exp()  # (N,num_classes)   
                
    return y_pred

def load_obj( name ):
    """
    Load dataset from pickle file.
    :param name: Full pathname of the pickle file
    :return: Dataset type of dictionary
    """
    with open( name , 'rb') as f:
        return pickle.load(f)
        
            
def main():
    parser = argparse.ArgumentParser(description='minibatch_gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--network', type=str, default='string')
    
    args = parser.parse_args()
    print(args)
    
    no_conv = False
      
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = load_obj(args.dataset)
    
    nlabels = 2
    #networks_name = ['string','CPDB','pcnet','pcnet','pcnet','pcnet','string']
        
    for i in range(len(dataset)):
        data = dataset[i]
        data.adj_t = data.adj_t.to_symmetric()
        

        x = data.x
        x = (x-x.mean(0))/x.std(0)
        data.x = x

        if data.y.dim()==2:
            data.y = data.y.squeeze(1)        
        
            
        data = data.to(device)
            
        layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12)        
        
        if args.model == 'sage_neighsampler':
            para_dict = sage_neighsampler_parameters
            model_para = sage_neighsampler_parameters.copy()
            model_para.pop('lr')
            model_para.pop('l2')
            model = SAGE_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)

        print(f'Model {args.model} initialized')

        #model_file = './model_files/dep_pc50_version1_{}_{}/model.pt'
        model_file = './model_files/dep_pc50_version2_{}_{}/sage_neighsampler/model.pt'.format(args.network, i)
        print('model_file:', model_file)
        model.load_state_dict(torch.load(model_file))

        out = test(layer_loader, model, data, device, no_conv)

        evaluator = Evaluator('auc')
        evaluator1 = Evaluator('acc')
        evaluator_prauc = Evaluator('prauc')
        evaluator_sepcificity_sensitivity_mcc_f1 = Evaluator('sepcificity_sensitivity_mcc_f1')

        preds_train, preds_valid, preds_test = out[data.train_mask], out[data.valid_mask], out[data.test_mask]
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


        
        preds = out[data.test_mask].cpu().numpy()
        print(out.cpu().numpy())
        
        #np.save('./model_files/dep_{}/{}/preds.npy'.format(args.network, args.model), out.cpu().numpy())


if __name__ == "__main__":
    main()
