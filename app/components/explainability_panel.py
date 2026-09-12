"""
Explainability panel component for the NetSeer dashboard.
"""

import streamlit as st
import numpy as np
from typing import Dict, List, Optional


def render_explainability_panel(
    shap_values: Optional[np.ndarray] = None,
    attention_weights: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    top_features: Optional[List[tuple]] = None,
    stage_name: str = "Unknown",
    stage_idx: int = 0,
):
    """Render the explainability panel with SHAP and attention visualizations."""

    from components.charts import (
        create_feature_importance_bar,
        create_attention_heatmap,
        STAGE_COLORS,
    )

    st.markdown("### 🔍 Prediction Explainability")

    # ─── Top Contributing Features ──────────────────────────────────
    if top_features:
        st.markdown("""
        <div class="glass-card" style="margin-bottom: 16px;">
            <h4 style="color: #00ffaa; margin-bottom: 12px;">🎯 Key Drivers</h4>
        """, unsafe_allow_html=True)

        explanation_items = []
        for i, (name, value) in enumerate(top_features[:5]):
            direction = "↑" if value > 0 else "↓"
            color = "#ff3366" if value > 0 else "#00ffaa"
            explanation_items.append(
                f'<div style="display: flex; justify-content: space-between; '
                f'padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">'
                f'<span style="color: rgba(255,255,255,0.8); font-size: 0.85rem;">'
                f'{i+1}. {name}</span>'
                f'<span style="color: {color}; font-weight: 600; font-size: 0.85rem;">'
                f'{direction} {abs(value):.4f}</span></div>'
            )

        st.markdown("".join(explanation_items) + "</div>", unsafe_allow_html=True)

    # ─── Natural Language Explanation ───────────────────────────────
    if top_features and stage_name:
        stage_color = STAGE_COLORS.get(stage_idx, '#888')
        feature_descriptions = []
        for name, val in top_features[:3]:
            if val > 0:
                feature_descriptions.append(f"elevated **{name}**")
            else:
                feature_descriptions.append(f"suppressed **{name}**")

        desc_text = ", ".join(feature_descriptions)
        st.markdown(f"""
        <div style="background: rgba(0,170,255,0.05); border: 1px solid rgba(0,170,255,0.2);
             border-radius: 10px; padding: 16px; margin: 12px 0;">
            <div style="color: #00aaff; font-weight: 600; margin-bottom: 8px;">💡 Analysis Summary</div>
            <div style="color: rgba(255,255,255,0.75); font-size: 0.9rem; line-height: 1.6;">
                The prediction for <span style="color: {stage_color}; font-weight: 600;">{stage_name}</span>
                is primarily driven by {desc_text}. These patterns are consistent with
                indicators observed during this attack phase in the MITRE ATT&CK framework.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── SHAP Feature Importance Chart ──────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        if top_features and feature_names:
            names = [f[0] for f in top_features]
            vals = [f[1] for f in top_features]
            fig = create_feature_importance_bar(names, vals, top_k=min(15, len(names)),
                                                 title="SHAP Feature Attribution")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if attention_weights is not None and attention_weights.ndim == 2:
            fig = create_attention_heatmap(attention_weights,
                                            title="Temporal Attention Weights")
            st.plotly_chart(fig, use_container_width=True)
        elif attention_weights is not None:
            # Average over heads if 3D
            avg_attn = np.mean(attention_weights, axis=0) if attention_weights.ndim == 3 else attention_weights
            fig = create_attention_heatmap(avg_attn, title="Temporal Attention (Avg. Heads)")
            st.plotly_chart(fig, use_container_width=True)

    # ─── Feature Contribution Breakdown ─────────────────────────────
    if shap_values is not None and feature_names:
        with st.expander("📊 Detailed Feature Breakdown", expanded=False):
            if shap_values.ndim == 3:
                # [1, seq_len, features] -> average over time
                mean_shap = np.mean(np.abs(shap_values[0]), axis=0)
            elif shap_values.ndim == 2:
                mean_shap = np.mean(np.abs(shap_values), axis=0)
            else:
                mean_shap = np.abs(shap_values)

            # Create a table
            import pandas as pd
            n_feats = min(len(feature_names), len(mean_shap))
            df = pd.DataFrame({
                'Feature': feature_names[:n_feats],
                'Mean |SHAP|': mean_shap[:n_feats],
                'Rank': range(1, n_feats + 1),
            }).sort_values('Mean |SHAP|', ascending=False).head(20)
            df['Rank'] = range(1, len(df) + 1)

            st.dataframe(
                df.set_index('Rank'),
                use_container_width=True,
                height=400,
            )


def render_mitre_attack_panel(
    stage_idx: int = 0,
    stage_info: Optional[Dict] = None,
):
    """Render the MITRE ATT&CK information panel."""

    from components.charts import create_kill_chain_html, STAGE_COLORS, STAGE_NAMES

    st.markdown("### 🎯 MITRE ATT&CK Mapping")

    # Kill Chain Progress
    st.markdown(create_kill_chain_html(stage_idx), unsafe_allow_html=True)

    if stage_info:
        stage_color = STAGE_COLORS.get(stage_idx, '#888')

        st.markdown(f"""
        <div class="glass-card" style="margin-top: 12px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <div style="width: 12px; height: 12px; border-radius: 50%;
                     background: {stage_color}; box-shadow: 0 0 10px {stage_color};"></div>
                <h4 style="color: {stage_color}; margin: 0;">
                    {stage_info.get('name', 'Unknown')}
                </h4>
            </div>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-bottom: 12px;">
                {stage_info.get('description', '')}
            </p>
        """, unsafe_allow_html=True)

        # Technique IDs
        techniques = stage_info.get('technique_ids', [])
        if techniques:
            tech_badges = " ".join([
                f'<span style="background: rgba(0,170,255,0.1); border: 1px solid rgba(0,170,255,0.3);'
                f'border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; color: #00aaff;'
                f'font-family: JetBrains Mono, monospace;">{t}</span>'
                for t in techniques
            ])
            st.markdown(f"""
            <div style="margin-bottom: 12px;">
                <span style="color: rgba(255,255,255,0.5); font-size: 0.75rem;
                      text-transform: uppercase; letter-spacing: 1px;">Techniques:</span><br/>
                <div style="margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap;">
                    {tech_badges}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Indicators
        indicators = stage_info.get('indicators', [])
        if indicators:
            indicator_html = "".join([
                f'<li style="color: rgba(255,255,255,0.7); font-size: 0.8rem; margin-bottom: 4px;">{ind}</li>'
                for ind in indicators[:5]
            ])
            st.markdown(f"""
            <div style="margin-bottom: 12px;">
                <span style="color: rgba(255,255,255,0.5); font-size: 0.75rem;
                      text-transform: uppercase; letter-spacing: 1px;">Network Indicators:</span>
                <ul style="margin-top: 6px; padding-left: 20px;">{indicator_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        # Mitigations
        mitigations = stage_info.get('mitigations', [])
        if mitigations:
            mit_html = "".join([
                f'<li style="color: rgba(255,255,255,0.7); font-size: 0.8rem; margin-bottom: 4px;">{mit}</li>'
                for mit in mitigations[:5]
            ])
            st.markdown(f"""
            <div>
                <span style="color: rgba(255,255,255,0.5); font-size: 0.75rem;
                      text-transform: uppercase; letter-spacing: 1px;">Recommended Actions:</span>
                <ul style="margin-top: 6px; padding-left: 20px;">{mit_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)