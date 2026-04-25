"""
TRUST-X Streamlit — main entry / landing page.
Run: streamlit run app.py
"""
import streamlit as st
from pathlib import Path

from trustx_core import load_trustx_model, fit_temperatures
import numpy as np

# ── Page config (applied to all pages) ──────────────────────────────────────
st.set_page_config(
    page_title="TRUST-X | Chest X-Ray Triage",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global disclaimer banner — shown on every page ──────────────────────────
def disclaimer_banner():
    st.warning(
        "⚠ **Research Prototype** — TRUST-X is an academic system built on "
        "NLP-labelled data. It is **not validated for clinical use**. "
        "Predictions must not be used for diagnosis or treatment decisions."
    )

# ── Shared model loader (cached across pages & reruns) ──────────────────────
@st.cache_resource(show_spinner="Loading TRUST-X model…")
def get_model():
    """
    Loads checkpoint, calibration arrays, builds model + GradCAM.
    Cached for the session.
    """
    ckpt_path   = Path("TRUSTX_densenet121_v2_320px_AFL.pth")
    val_logits  = np.load("val_logits.npy")
    val_labels  = np.load("val_labels.npy")

    model, meta, infer_tfm = load_trustx_model(str(ckpt_path), device='cpu')
    temperatures = fit_temperatures(val_logits, val_labels, meta['label_cols'])

    return model, meta, infer_tfm, temperatures

# ── Expose loader on session state (so other pages share it) ────────────────
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "triage_queue" not in st.session_state:
    st.session_state.triage_queue = {}   # {patient_id: {...}}

# ── Landing page content ────────────────────────────────────────────────────
disclaimer_banner()

st.title("TRUST-X")
st.subheader("Triage with Uncertainty and Structured Transparency for X-Ray Imaging")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ### What is TRUST-X?

        A research system for chest X-ray triage that combines:
        
        - **Multi-label classification** across 14 thoracic pathologies
        - **GradCAM explainability** — see where the model looks
        - **MC Dropout uncertainty** — know how confident the model is
        - **Severity-weighted triage** — prioritise urgent cases

        ### How to use this app

        **1. Triage Queue** (sidebar) — Upload multiple X-rays. System ranks them by 
        urgency. *Receptionist intake view.*

        **2. Review** (sidebar) — Select any patient from the queue to see 
        classification, GradCAM heatmaps, and uncertainty estimates. 
        *Radiologist reading view.*

        **3. About** (sidebar) — Methodology, limitations, references.
        """
    )

with col2:
    # Load model and show status card
    try:
        model, meta, infer_tfm, temperatures = get_model()
        st.session_state.model_loaded = True

        st.success("✅ Model loaded")
        st.metric("Test AUC", f"{meta['test_auc']:.4f}" if meta['test_auc'] else "—")
        st.metric("Labels", len(meta['label_cols']))
        st.metric("Input size", f"{meta['img_size']}×{meta['img_size']}")
        st.caption(f"Architecture: {meta['architecture']}  |  Device: {meta['device']}")

    except FileNotFoundError as e:
        st.error(
            "❌ Model files not found. Ensure these exist in the app directory:\n"
            "- `TRUSTX_densenet121_v2_320px_AFL.pth`\n"
            "- `val_logits.npy`\n"
            "- `val_labels.npy`"
        )
        st.code(str(e))

st.markdown("---")
st.caption(
    "Built as an academic project. Based on NIH ChestX-ray14 (Wang et al. 2017). "
    "Model: DenseNet-121 trained with Asymmetric Focal Loss (Ben-Baruch et al. 2021)."
)
