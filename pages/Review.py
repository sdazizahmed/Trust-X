"""
Page 2 — Radiologist Review.
Select a patient from the triage queue → see full analysis.
"""
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trustx_core import (
    predict, mc_dropout_predict, build_gradcam,
    gradcam_heatmap, overlay_heatmap, triage_score, priority_tier
)

st.set_page_config(page_title="TRUST-X | Review", page_icon="🫁", layout="wide")

# ── Disclaimer ───────────────────────────────────────────────────────────────
st.warning(
    "⚠ **Academic Prototype** — Not validated for clinical use. "
    "Predictions must not be used for diagnosis."
)

# ── Require model loaded ────────────────────────────────────────────────────
if not st.session_state.get("model_loaded", False):
    st.error("Model not loaded. Please return to the **Home** page first.")
    st.stop()

from app import get_model
model, meta, infer_tfm, temperatures = get_model()

# Cache GradCAM setup (hooks are registered once)
@st.cache_resource
def get_gradcam():
    return build_gradcam(model)
gradcam = get_gradcam()

# ── Page ─────────────────────────────────────────────────────────────────────
st.title("🔍 Stage 2 — Radiologist Review")
st.caption("Drill into any patient from the triage queue for full analysis.")

queue = st.session_state.get("triage_queue", {})

# ── Patient selector ─────────────────────────────────────────────────────────
if not queue:
    st.info("📋 The triage queue is empty. Go to the **Triage Queue** page to upload X-rays first.")
    st.stop()

ranked = sorted(queue.items(), key=lambda kv: kv[1]['score'], reverse=True)
options = {f"{pid}  —  score {info['score']:.2f}  |  {info['top_finding']}": pid
           for pid, info in ranked}

selection = st.selectbox(
    "Select patient (ranked by urgency)",
    options=list(options.keys()),
)
pid = options[selection]
info = queue[pid]

# ── Patient header ──────────────────────────────────────────────────────────
tier_label, tier_color = priority_tier(info['score'])
emoji = {'URGENT':'🔴','HIGH':'🟠','MEDIUM':'🟡','ROUTINE':'🟢'}[tier_label]

st.markdown(
    f"""
    <div style="background:{tier_color}; color:white; padding:16px 20px; border-radius:8px; margin-bottom:20px;">
      <div style="font-size:14px; opacity:0.9;">PATIENT</div>
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <span style="font-size:28px; font-weight:bold; font-family:monospace;">{pid}</span>
          <span style="margin-left:20px; font-size:18px;">{emoji} {tier_label}</span>
        </div>
        <div style="text-align:right;">
          <div style="font-size:12px; opacity:0.8;">TRIAGE SCORE</div>
          <div style="font-size:28px; font-weight:bold;">{info['score']:.2f}</div>
        </div>
      </div>
      <div style="font-size:12px; opacity:0.85; margin-top:8px;">{info['filename']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Controls ─────────────────────────────────────────────────────────────────
with st.expander("⚙ Analysis settings", expanded=False):
    threshold   = st.slider("Decision threshold", 0.1, 0.9, 0.3, 0.05)
    top_k       = st.slider("GradCAM — top K predictions", 1, 5, 3)
    n_mc_passes = st.slider("MC Dropout passes (slower, more accurate uncertainty)",
                            10, 50, 20, 5)

# ── Run full analysis ───────────────────────────────────────────────────────
if st.button("🔬 Run Full Analysis", type="primary"):
    img = Image.open(info['path']).convert('RGB')

    # ── 1. Calibrated probabilities ─────────────────────────────────────────
    st.markdown("### 1  ·  Calibrated Predictions")

    probs = predict(model, img, infer_tfm, meta['device'],
                    temperatures=temperatures, calibrate=True)

    # Build results DataFrame
    results = pd.DataFrame({
        'Label': meta['label_cols'],
        'Probability': probs,
        'Prediction': ['POSITIVE' if p >= threshold else 'negative' for p in probs],
    }).sort_values('Probability', ascending=False).reset_index(drop=True)

    col_img, col_chart = st.columns([1, 2])
    with col_img:
        st.image(img, caption=f"{info['filename']}", use_container_width=True)
    with col_chart:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#d73027' if p == 'POSITIVE' else '#91bfdb'
                  for p in results['Prediction']]
        ax.barh(results['Label'], results['Probability'], color=colors)
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.2,
                   label=f'Threshold = {threshold}')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Calibrated probability')
        ax.invert_yaxis()
        ax.legend(loc='lower right', fontsize=9)
        for i, (p, pred) in enumerate(zip(results['Probability'], results['Prediction'])):
            ax.text(min(p + 0.01, 0.97), i, f'{p:.3f}',
                    va='center', fontsize=8,
                    fontweight='bold' if pred == 'POSITIVE' else 'normal',
                    color='darkred' if pred == 'POSITIVE' else 'gray')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Positive findings summary
    pos = results[results['Prediction'] == 'POSITIVE']
    if len(pos) > 0:
        st.success(f"**{len(pos)} positive finding(s) above threshold {threshold}:** "
                   + ", ".join([f"**{r['Label']}** ({r['Probability']:.3f})"
                                for _, r in pos.iterrows()]))
    else:
        st.info(f"No findings above threshold {threshold}.")

    st.markdown("---")

    # ── 2. GradCAM ──────────────────────────────────────────────────────────
    st.markdown("### 2  ·  GradCAM — Where the Model Looked")
    st.caption(f"Showing heatmaps for top-{top_k} predicted labels. "
               f"Red = high attention, blue = low attention.")

    top_idx = probs.argsort()[::-1][:top_k]
    gc_cols = st.columns(top_k)

    with st.spinner("Computing GradCAM heatmaps…"):
        for col, idx in zip(gc_cols, top_idx):
            cam = gradcam_heatmap(gradcam, img, infer_tfm, meta['device'],
                                  int(idx), meta['img_size'])
            overlay = overlay_heatmap(img, cam, meta['img_size'], alpha=0.45)
            with col:
                label = meta['label_cols'][idx]
                prob = probs[idx]
                st.image(overlay, caption=f"**{label}** — prob {prob:.3f}",
                         use_container_width=True)

    st.markdown("---")

    # ── 3. MC Dropout Uncertainty ───────────────────────────────────────────
    st.markdown("### 3  ·  MC Dropout — Uncertainty Estimation")
    st.caption(f"Running {n_mc_passes} stochastic forward passes with dropout enabled. "
               f"Standard deviation across passes indicates model confidence.")

    with st.spinner(f"Running {n_mc_passes} stochastic forward passes…"):
        mean_probs, std_probs, confidence = mc_dropout_predict(
            model, img, infer_tfm, meta['device'],
            temperatures=temperatures, calibrate=True, n_passes=n_mc_passes
        )

    unc_df = pd.DataFrame({
        'Label': meta['label_cols'],
        'Mean Prob': mean_probs,
        'Std': std_probs,
        'Confidence': confidence,
    }).sort_values('Mean Prob', ascending=False).reset_index(drop=True)

    col_plot, col_table = st.columns([2, 1])
    with col_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        color_map = {'HIGH': '#2a9d8f', 'MEDIUM': '#f4a261', 'LOW': '#e63946'}
        bar_colors = [color_map[c] for c in unc_df['Confidence']]
        ax.barh(unc_df['Label'], unc_df['Mean Prob'],
                xerr=unc_df['Std'], color=bar_colors,
                error_kw={'ecolor': 'black', 'capsize': 3, 'elinewidth': 1.2})
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability (mean ± std)')
        ax.invert_yaxis()
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c, label=t) for t, c in color_map.items()]
        ax.legend(handles=handles, loc='lower right', title='Confidence', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_table:
        st.markdown("**Top 5 by mean probability**")
        display_df = unc_df.head(5).copy()
        display_df['Mean'] = display_df['Mean Prob'].apply(lambda v: f"{v:.3f}")
        display_df['Std']  = display_df['Std'].apply(lambda v: f"{v:.3f}")
        st.dataframe(display_df[['Label', 'Mean', 'Std', 'Confidence']],
                     hide_index=True, use_container_width=True)

    # Confidence legend
    st.markdown(
        "**Confidence tiers:** "
        "🟢 **HIGH** (std < 0.05) — model is consistent  ·  "
        "🟡 **MEDIUM** (0.05–0.10)  ·  "
        "🔴 **LOW** (> 0.10) — do not act on probability alone"
    )

    st.markdown("---")
    st.success(f"✅ Full analysis complete for patient **{pid}**.")
