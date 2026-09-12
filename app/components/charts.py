"""
Reusable Plotly chart builders for the NetSeer dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Optional, Dict, Tuple


# â”€â”€â”€ Color Palette â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
COLORS = {
    'primary': '#00ffaa',
    'secondary': '#00aaff',
    'danger': '#ff3366',
    'warning': '#ff9900',
    'caution': '#ffcc00',
    'purple': '#aa66ff',
    'background': '#0a0e1a',
    'card': '#0d1525',
    'text': 'rgba(255,255,255,0.8)',
    'text_muted': 'rgba(255,255,255,0.4)',
    'grid': 'rgba(255,255,255,0.06)',
}

STAGE_COLORS = {
    0: '#00ffaa',   # Benign
    1: '#ffcc00',   # Reconnaissance
    2: '#ff9900',   # Initial Access
    3: '#ff6633',   # Lateral Movement
    4: '#ff3366',   # C2
    5: '#cc00ff',   # Exfiltration
}

STAGE_NAMES = {
    0: 'Benign',
    1: 'Reconnaissance',
    2: 'Initial Access',
    3: 'Lateral Movement',
    4: 'Command & Control',
    5: 'Exfiltration',
}

SEVERITY_COLORS = {
    'LOW': '#00ffaa',
    'MEDIUM': '#ffcc00',
    'HIGH': '#ff9900',
    'CRITICAL': '#ff3366',
}


def _apply_dark_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply consistent dark theme layout to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS['text'], size=16, family='Inter')),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=40, r=30, t=50 if title else 30, b=40),
        font=dict(family='Inter', color=COLORS['text']),
        xaxis=dict(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            showgrid=True,
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(size=11),
        ),
    )
    return fig


def create_infiltration_timeline(
    probabilities: List[float],
    stages: List[int],
    timestamps: Optional[List[str]] = None,
    title: str = "Infiltration Probability Forecast"
) -> go.Figure:
    """Create an infiltration probability timeline with attack stage color bands."""
    x = timestamps if timestamps else list(range(len(probabilities)))

    fig = go.Figure()

    # Threshold bands
    fig.add_hrect(y0=0, y1=0.3, fillcolor="rgba(0,255,170,0.03)", line_width=0)
    fig.add_hrect(y0=0.3, y1=0.6, fillcolor="rgba(255,204,0,0.03)", line_width=0)
    fig.add_hrect(y0=0.6, y1=0.8, fillcolor="rgba(255,153,0,0.05)", line_width=0)
    fig.add_hrect(y0=0.8, y1=1.0, fillcolor="rgba(255,51,102,0.05)", line_width=0)

    # Threshold lines
    for thresh, label, color in [(0.3, 'LOW', '#00ffaa'), (0.6, 'MEDIUM', '#ffcc00'),
                                  (0.8, 'HIGH', '#ff9900'), (0.95, 'CRITICAL', '#ff3366')]:
        fig.add_hline(y=thresh, line_dash="dash", line_color=color, opacity=0.4,
                      annotation_text=label, annotation_position="right",
                      annotation_font_color=color, annotation_font_size=10)

    # Probability line
    colors_per_point = [STAGE_COLORS.get(s, '#00ffaa') for s in stages]
    fig.add_trace(go.Scatter(
        x=x, y=probabilities,
        mode='lines+markers',
        name='Infiltration Probability',
        line=dict(color=COLORS['danger'], width=2.5, shape='spline'),
        marker=dict(color=colors_per_point, size=6, line=dict(width=1, color='white')),
        fill='tozeroy',
        fillcolor='rgba(255,51,102,0.08)',
        hovertemplate='<b>Step %{x}</b><br>Probability: %{y:.3f}<br><extra></extra>',
    ))

    fig = _apply_dark_layout(fig, title, height=380)
    fig.update_yaxes(range=[0, 1.05], title='Probability')
    fig.update_xaxes(title='Time Step')

    return fig


def create_attack_stage_timeline(
    stages: List[int],
    timestamps: Optional[List[str]] = None,
    title: str = "Predicted Attack Stages"
) -> go.Figure:
    """Create a color-coded attack stage timeline."""
    x = timestamps if timestamps else list(range(len(stages)))

    fig = go.Figure()

    for stage_idx in sorted(set(stages)):
        mask = [1 if s == stage_idx else None for s in stages]
        y_vals = [stage_idx if m else None for m in mask]
        fig.add_trace(go.Scatter(
            x=x, y=y_vals,
            mode='markers',
            name=STAGE_NAMES.get(stage_idx, f'Stage {stage_idx}'),
            marker=dict(
                color=STAGE_COLORS.get(stage_idx, '#888'),
                size=12,
                symbol='square',
            ),
            hovertemplate=f'<b>{STAGE_NAMES.get(stage_idx, "Unknown")}</b><br>Step: %{{x}}<extra></extra>',
        ))

    fig = _apply_dark_layout(fig, title, height=250)
    fig.update_yaxes(
        tickvals=list(STAGE_NAMES.keys()),
        ticktext=list(STAGE_NAMES.values()),
        title='Attack Stage',
    )

    return fig


def create_feature_importance_bar(
    feature_names: List[str],
    importances: List[float],
    top_k: int = 15,
    title: str = "Top Contributing Features"
) -> go.Figure:
    """Create a horizontal bar chart of feature importances."""
    # Sort and truncate
    sorted_pairs = sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    sorted_pairs.reverse()  # Bottom-to-top
    names, vals = zip(*sorted_pairs)

    colors = [COLORS['danger'] if v > 0 else COLORS['primary'] for v in vals]

    fig = go.Figure(go.Bar(
        x=list(vals),
        y=list(names),
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
    ))

    fig = _apply_dark_layout(fig, title, height=max(300, top_k * 28))
    fig.update_xaxes(title='Attribution Score')

    return fig


def create_attention_heatmap(
    attention_matrix: np.ndarray,
    title: str = "Temporal Attention Weights"
) -> go.Figure:
    """Create an attention heatmap visualization."""
    seq_len = attention_matrix.shape[0]
    labels = [f't-{seq_len - 1 - i}' if i < seq_len - 1 else 't' for i in range(seq_len)]

    fig = go.Figure(go.Heatmap(
        z=attention_matrix,
        x=labels,
        y=labels,
        colorscale=[
            [0, '#0a0e1a'],
            [0.25, '#0d2540'],
            [0.5, '#00557a'],
            [0.75, '#00aaaa'],
            [1, '#00ffaa'],
        ],
        hovertemplate='Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>',
    ))

    fig = _apply_dark_layout(fig, title, height=400)
    fig.update_xaxes(title='Key (Source)')
    fig.update_yaxes(title='Query (Target)')

    return fig


def create_network_state_heatmap(
    states: np.ndarray,
    feature_names: List[str],
    title: str = "Network State Evolution"
) -> go.Figure:
    """Create a heatmap showing how network state features evolve over time."""
    # Truncate feature names for display
    display_names = [n[:25] + '...' if len(n) > 25 else n for n in feature_names[:30]]
    display_states = states[:, :30] if states.shape[1] > 30 else states

    fig = go.Figure(go.Heatmap(
        z=display_states.T,
        x=[f'W{i}' for i in range(display_states.shape[0])],
        y=display_names,
        colorscale=[
            [0, '#0a0e1a'],
            [0.3, '#003355'],
            [0.6, '#0077aa'],
            [0.8, '#00aaff'],
            [1, '#00ffaa'],
        ],
        hovertemplate='Window: %{x}<br>Feature: %{y}<br>Value: %{z:.3f}<extra></extra>',
    ))

    fig = _apply_dark_layout(fig, title, height=500)

    return fig


def create_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix"
) -> go.Figure:
    """Create a styled confusion matrix heatmap."""
    # Normalize
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig = go.Figure(go.Heatmap(
        z=cm_normalized,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        textfont=dict(size=12, color='white'),
        colorscale=[
            [0, '#0a0e1a'],
            [0.5, '#005577'],
            [1, '#00ffaa'],
        ],
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{text}<br>Rate: %{z:.2%}<extra></extra>',
    ))

    fig = _apply_dark_layout(fig, title, height=450)
    fig.update_xaxes(title='Predicted')
    fig.update_yaxes(title='Actual')

    return fig


def create_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "ROC Curve"
) -> go.Figure:
    """Create a ROC curve with AUC display."""
    fig = go.Figure()

    # Random classifier diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines', line=dict(dash='dash', color=COLORS['text_muted'], width=1),
        name='Random (AUC = 0.5)', showlegend=True,
    ))

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'World Model (AUC = {auc_score:.4f})',
        line=dict(color=COLORS['primary'], width=2.5),
        fill='tonexty',
        fillcolor='rgba(0,255,170,0.1)',
    ))

    fig = _apply_dark_layout(fig, title, height=400)
    fig.update_xaxes(title='False Positive Rate', range=[0, 1])
    fig.update_yaxes(title='True Positive Rate', range=[0, 1])

    return fig


def create_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History"
) -> go.Figure:
    """Create training loss curves."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss', 'Metrics'),
        horizontal_spacing=0.12,
    )

    loss_colors = {
        'total_loss': COLORS['primary'],
        'transition_loss': COLORS['secondary'],
        'infiltration_loss': COLORS['danger'],
        'stage_loss': COLORS['warning'],
        'val_loss': COLORS['purple'],
    }

    for key, values in history.items():
        if 'loss' in key.lower():
            color = loss_colors.get(key, COLORS['text'])
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                name=key.replace('_', ' ').title(),
                line=dict(color=color, width=2),
                mode='lines',
            ), row=1, col=1)
        elif any(m in key.lower() for m in ['f1', 'accuracy', 'precision', 'recall']):
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                name=key.replace('_', ' ').title(),
                line=dict(width=2),
                mode='lines',
            ), row=1, col=2)

    fig = _apply_dark_layout(fig, title, height=380)
    fig.update_xaxes(title_text='Epoch', gridcolor=COLORS['grid'])
    fig.update_yaxes(gridcolor=COLORS['grid'])

    return fig


def create_kill_chain_html(active_stage: int = 0) -> str:
    """Create an HTML kill chain progress visualization."""
    stages = [
        ('Recon', 1),
        ('Initial Access', 2),
        ('Lateral Movement', 3),
        ('C2', 4),
        ('Exfiltration', 5),
    ]

    html = '<div class="kill-chain-container">'
    for name, idx in stages:
        if idx < active_stage:
            css_class = 'kill-chain-passed'
        elif idx == active_stage:
            css_class = 'kill-chain-active'
        else:
            css_class = 'kill-chain-inactive'
        html += f'<div class="kill-chain-step {css_class}">{name}</div>'
    html += '</div>'

    return html


def create_gauge_chart(
    value: float,
    title: str = "Threat Level",
    max_val: float = 1.0
) -> go.Figure:
    """Create a gauge chart for threat level display."""
    if value < 0.3:
        bar_color = COLORS['primary']
    elif value < 0.6:
        bar_color = COLORS['caution']
    elif value < 0.8:
        bar_color = COLORS['warning']
    else:
        bar_color = COLORS['danger']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(size=32, color=bar_color), valueformat='.2%'),
        title=dict(text=title, font=dict(size=14, color=COLORS['text'])),
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor=COLORS['text_muted'],
                       tickfont=dict(size=10, color=COLORS['text_muted'])),
            bar=dict(color=bar_color, thickness=0.7),
            bgcolor=COLORS['card'],
            borderwidth=1,
            bordercolor='rgba(0,255,170,0.2)',
            steps=[
                dict(range=[0, 0.3], color='rgba(0,255,170,0.05)'),
                dict(range=[0.3, 0.6], color='rgba(255,204,0,0.05)'),
                dict(range=[0.6, 0.8], color='rgba(255,153,0,0.08)'),
                dict(range=[0.8, 1.0], color='rgba(255,51,102,0.08)'),
            ],
            threshold=dict(
                line=dict(color=COLORS['danger'], width=2),
                thickness=0.8,
                value=0.8
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=30, r=30, t=50, b=20),
        font=dict(family='Inter'),
    )

    return fig


def create_metric_comparison_table(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    title: str = "Model Comparison"
) -> go.Figure:
    """Create a styled comparison table using Plotly."""
    header_values = ['Model'] + [k.replace('_', ' ').title() for k in metrics.keys()]
    cell_values = [model_names] + [
        [f'{v:.4f}' for v in vals] for vals in metrics.values()
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=header_values,
            fill_color=COLORS['card'],
            font=dict(color=COLORS['primary'], size=12, family='Inter'),
            align='center',
            line_color=COLORS['grid'],
            height=36,
        ),
        cells=dict(
            values=cell_values,
            fill_color=COLORS['background'],
            font=dict(color=COLORS['text'], size=11, family='JetBrains Mono'),
            align='center',
            line_color=COLORS['grid'],
            height=32,
        ),
    ))

    fig = _apply_dark_layout(fig, title, height=max(180, 50 + len(model_names) * 36))

    return fig

def create_state_trajectory_chart(
    predicted_states: np.ndarray,
    feature_names: List[str],
    top_indices: Optional[List[int]] = None,
    title: str = "Predicted State Trajectory"
) -> go.Figure:
    """Plot how selected state features evolve over the forecast horizon."""
    if top_indices is None:
        n = min(4, predicted_states.shape[1])
        top_indices = list(range(n))

    x = list(range(predicted_states.shape[0]))
    marker_colors = [COLORS['secondary'], COLORS['primary'], COLORS['warning'], COLORS['purple']]
    fig = go.Figure()

    for i, idx in enumerate(top_indices[:4]):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        fig.add_trace(go.Scatter(
            x=x, y=predicted_states[:, idx],
            mode='lines+markers',
            name=name,
            line=dict(color=marker_colors[i % len(marker_colors)], width=2),
            marker=dict(size=5),
            hovertemplate='Step %{x}<br>%{name}: %{y:.4f}<extra></extra>',
        ))

    fig = _apply_dark_layout(fig, title, height=380)
    fig.update_xaxes(title='Forecast Step')
    fig.update_yaxes(title='State Value')

    return fig

