import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Optional

TEMPLATE = 'plotly_dark'

def plot_infiltration_timeline(probabilities: List[float], stages: List[str], timestamps: Optional[List[str]] = None) -> go.Figure:
    fig = go.Figure()
    
    x = timestamps if timestamps else list(range(len(probabilities)))
    
    fig.add_trace(go.Scatter(
        x=x, y=probabilities,
        mode='lines+markers',
        name='Infiltration Prob',
        line=dict(color='cyan', width=2),
        marker=dict(size=8, color='cyan')
    ))
    
    for y_val, color, name in [(0.3, 'yellow', 'LOW'), (0.6, 'orange', 'MEDIUM'), 
                               (0.8, 'red', 'HIGH'), (0.95, 'darkred', 'CRITICAL')]:
        fig.add_hline(y=y_val, line_dash="dash", line_color=color, annotation_text=name, 
                      annotation_position="bottom right", opacity=0.5)
                      
    fig.update_layout(
        title='Infiltration Probability Forecast',
        xaxis_title='Time Step',
        yaxis_title='Probability',
        yaxis_range=[0, 1.05],
        template=TEMPLATE
    )
    return fig

def plot_attention_heatmap(attention_matrix: np.ndarray, labels: Optional[List[str]] = None) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=labels, y=labels,
        colorscale='Viridis'
    ))
    fig.update_layout(
        title='Attention Heatmap',
        template=TEMPLATE
    )
    return fig

def plot_shap_waterfall(feature_names: List[str], shap_values: np.ndarray, max_display: int = 15) -> go.Figure:
    abs_shap = np.abs(shap_values)
    indices = np.argsort(abs_shap)[-max_display:]
    
    selected_features = [feature_names[i] for i in indices]
    selected_shap = [shap_values[i] for i in indices]
    
    colors = ['red' if val > 0 else 'blue' for val in selected_shap]
    
    fig = go.Figure(go.Bar(
        x=selected_shap,
        y=selected_features,
        orientation='h',
        marker_color=colors
    ))
    fig.update_layout(
        title='SHAP Feature Importances',
        xaxis_title='SHAP Value',
        yaxis_title='Feature',
        template=TEMPLATE
    )
    return fig

def plot_feature_importance(feature_names: List[str], importances: np.ndarray, top_k: int = 15) -> go.Figure:
    indices = np.argsort(importances)[-top_k:]
    selected_features = [feature_names[i] for i in indices]
    selected_imp = [importances[i] for i in indices]
    
    fig = go.Figure(go.Bar(
        x=selected_imp,
        y=selected_features,
        orientation='h',
        marker_color='cyan'
    ))
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        template=TEMPLATE
    )
    return fig

def plot_confusion_matrix(cm: np.ndarray, labels: List[str]) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels, y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        hoverinfo="z"
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template=TEMPLATE
    )
    return fig

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (area = {auc_score:.2f})',
                             line=dict(color='cyan', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random',
                             line=dict(color='gray', dash='dash')))
    fig.update_layout(
        title='Receiver Operating Characteristic',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis_range=[0.0, 1.0],
        yaxis_range=[0.0, 1.05],
        template=TEMPLATE
    )
    return fig

def plot_training_history(history: dict) -> go.Figure:
    fig = go.Figure()
    for key, values in history.items():
        if 'loss' in key.lower() or 'acc' in key.lower():
            fig.add_trace(go.Scatter(y=values, mode='lines', name=key))
            
    fig.update_layout(
        title='Training History',
        xaxis_title='Epoch',
        yaxis_title='Value',
        template=TEMPLATE
    )
    return fig

def plot_state_trajectory(states: np.ndarray, labels: Optional[np.ndarray] = None) -> go.Figure:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(states)
    
    fig = go.Figure(data=go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode='markers',
        marker=dict(
            color=labels if labels is not None else 'cyan',
            colorscale='Viridis',
            showscale=True if labels is not None else False
        )
    ))
    fig.update_layout(
        title='State Trajectory (PCA)',
        xaxis_title='PC1',
        yaxis_title='PC2',
        template=TEMPLATE
    )
    return fig
