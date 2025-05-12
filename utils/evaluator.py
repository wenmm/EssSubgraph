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

        if not y_pred.ndim == 2:
            raise RuntimeError('y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

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
        if self.eval_metric == 'sepcificity_sensitivity_mcc_f1':
            y_true, y_pred = self._check_input(y_true, y_pred)
            return self._eval_sepcificity_sensitivity_mcc_f1(y_true, y_pred)


    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC and AP score averaged across tasks
        '''
        
        if y_pred.shape[1] ==2:
            auc = roc_auc_score(y_true, y_pred[:, 1])
        else:
            onehot_code = np.eye(y_pred.shape[1])
            y_true_onehot = onehot_code[y_true]
            auc = roc_auc_score(y_true_onehot, y_pred)

        return {'auc': auc}

    def _eval_acc(self, y_true, y_pred):
        y_pred = y_pred.argmax(axis=-1)

        correct = y_true == y_pred
        acc = float(np.sum(correct))/len(correct)

        return {'acc': acc}
    
    def _eval_prauc(self, y_true, y_pred):
        precision, recall, _ = precision_recall_curve(y_true, y_pred[:, 1])
        prauc = auc(recall,precision)

        return {'prauc': prauc}
    
    def _eval_sepcificity_sensitivity_mcc_f1(self, y_true, y_pred):
        result = []
        rounded = [round(x[1]) for x in y_pred]
        confusion = confusion_matrix(y_true, rounded)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        sepcificity = TN / float( TN + FP)
        sensitivity = TP / float(FN + TP)
        mcc = matthews_corrcoef(y_true, rounded)
        f1 = f1_score(y_true, rounded)
        #fpr, tpr, thresholds = roc_curve(y_true, y_pred[:, 1])
        result.append({'sepcificity': sepcificity})
        result.append({'sensitivity': sensitivity})
        result.append({'mcc': mcc})
        result.append({'f1': f1})
        #result.append({'auc': auc(fpr, tpr)})
        return {'sepcificity_sensitivity_mcc_f1': result}



