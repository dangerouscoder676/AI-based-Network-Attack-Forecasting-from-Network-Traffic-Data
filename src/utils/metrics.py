import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
import warnings
from typing import Dict, Any, List

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Computes basic classification metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) == 0:
        return {}
        
    acc = accuracy_score(y_true, y_pred)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        report = classification_report(y_true, y_pred, output_dict=True)
    
    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'classification_report': report
    }
    
    unique_classes = np.unique(y_true)
    is_binary = len(unique_classes) <= 2 and (set(unique_classes) <= {0, 1})
    
    if is_binary:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            metrics['specificity'] = tn / (fp + tn) if (fp + tn) > 0 else 0.0
        
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['auc_roc'] = 0.5
                
    return metrics

def compute_binary_metrics(y_true, y_pred, y_prob=None) -> dict:
    return compute_metrics(y_true, y_pred, y_prob)

def compute_stage_metrics(y_true, y_pred) -> dict:
    return compute_metrics(y_true, y_pred)

def confusion_matrix_dict(y_true, y_pred, labels=None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        'matrix': cm.tolist(),
        'labels': labels if labels is not None else np.unique(np.concatenate([y_true, y_pred])).tolist()
    }
