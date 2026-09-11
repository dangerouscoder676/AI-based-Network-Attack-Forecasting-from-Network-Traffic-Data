import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from typing import Dict, Any

class LogisticBaseline:
    """
    Logistic Regression baseline model operating on per-flow features.
    """
    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticBaseline':
        """
        Fit the logistic regression model.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            
        Returns:
            self
        """
        self.model.fit(X, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        return self.model.predict_proba(X)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model and return a dictionary of metrics.
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: True labels of shape (n_samples,)
            
        Returns:
            Dictionary containing metrics: accuracy, f1_macro, f1_weighted, 
            precision, recall, fpr, classification_report.
        """
        y_pred = self.predict(X)
        
        acc = accuracy_score(y, y_pred)
        f1_mac = f1_score(y, y_pred, average='macro', zero_division=0)
        f1_weight = f1_score(y, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y, y_pred, average='macro', zero_division=0)
        rec = recall_score(y, y_pred, average='macro', zero_division=0)
        
        # Calculate FPR (False Positive Rate) per class and average it
        cm = confusion_matrix(y, y_pred)
        
        fpr_list = []
        for i in range(len(cm)):
            # False Positives for class i: sum of column i minus diagonal element
            fp = cm[:, i].sum() - cm[i, i]
            # True Negatives for class i: sum of all elements minus row i and col i
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            if (fp + tn) > 0:
                fpr_list.append(fp / (fp + tn))
            else:
                fpr_list.append(0.0)
                
        fpr_macro = np.mean(fpr_list) if len(fpr_list) > 0 else 0.0
        
        clf_report = classification_report(y, y_pred, zero_division=0, output_dict=True)
        
        return {
            'accuracy': acc,
            'f1_macro': f1_mac,
            'f1_weighted': f1_weight,
            'precision': prec,
            'recall': rec,
            'fpr': fpr_macro,
            'classification_report': clf_report
        }
