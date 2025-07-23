import os
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef,roc_auc_score, confusion_matrix,roc_curve, auc, precision_recall_curve, accuracy_score, f1_score
try:
    import torch
except ImportError:
    torch = None   
    
### Evaluator for node property prediction
class Evaluator:
    def __init__(self, eval_metric):
        if eval_metric not in ['acc', 'auc','prauc','sepcificity_sensitivity_mcc_f1']:
            raise ValueError('eval_metric should be acc or auc or prauc or sepcificity_sensitivity_mcc_f1')
            
        self.eval_metric = eval_metric

    def _check_input(self, y_true, y_pred):
        '''
            y_true: numpy ndarray or torch tensor of shape (num_node)
            y_pred: numpy ndarray or torch tensor of shape (num_node, num_tasks)
        '''

        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        ## check type
        if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
            raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')


        return y_true, y_pred

    def eval(self, y_true, y_pred):
        if self.eval_metric == 'auc':
            y_true, y_pred = self._check_input(y_true, y_pred)
            return self._eval_rocauc(y_true, y_pred)
        if self.eval_metric == 'acc':
            y_true, y_pred = self._check_input(y_true, y_pred)
            return self._eval_acc(y_true, y_pred)
        if self.eval_metric == 'prauc':
            y_true, y_pred = self._check_input(y_true, y_pred)
            return self._eval_prauc(y_true, y_pred)


    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC and AP score averaged across tasks
        '''
        

        auc = roc_auc_score(y_true, y_pred)

        return {'auc': auc}

    def _eval_acc(self, y_true, y_pred):

        correct = y_true == y_pred
        acc = float(np.sum(correct))/len(correct)

        return {'acc': acc}
    
    def _eval_prauc(self, y_true, y_pred):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        prauc = auc(recall,precision)

        return {'prauc': prauc}



