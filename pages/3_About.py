"""
Page 3 — About.
Methodology, limitations, references.
"""
import streamlit as st

st.set_page_config(page_title="TRUST-X | About", page_icon="🫁", layout="wide")

st.warning(
    "⚠ **Academic Prototype** — Not validated for clinical use. "
    "Predictions must not be used for diagnosis."
)

st.title("About TRUST-X")

tabs = st.tabs(["Methodology", "Results", "Limitations", "References"])

# ── Methodology ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown(
        """
        ### Data
        - **NIH ChestX-ray14** — 112,120 frontal chest X-rays, 30,805 unique patients
        - Labels extracted from radiology reports via NLP (not direct annotation)
        - Known label noise ~10–20% — a fundamental property of this dataset
        - **Patient-level** train/val/test split (70/15/15) — prevents data leakage, 
          a methodological improvement over the original CheXNet paper's image-level split

        ### Model
        - **DenseNet-121** pre-trained on ImageNet
        - Custom classifier head: `Dropout(0.3) → Linear(1024 → 14)`
        - Input resolution: 320×320px
        - Two-phase training: 5 epochs head-only → 20 epochs full fine-tune (early stop)

        ### Loss Function — the key innovation
        - Originally tried **BCE + pos_weight** — pushed logits to ±15 range, 
          breaking any possibility of post-hoc calibration
        - Tried **temperature scaling** and **Platt scaling** on the broken logits — 
          both failed for the same reason: a single scalar cannot linearise a 
          pathologically skewed distribution
        - **Switched to Asymmetric Focal Loss** (Ben-Baruch et al., ICCV 2021)
            - γ_neg = 4 (heavy focus on hard negatives)
            - γ_pos = 0 (standard BCE signal on positives)
            - No pos_weight needed — imbalance handled dynamically
        - Result: logits stay in ±3 range, calibration becomes possible, and 
          per-label temperature scaling now converges cleanly

        ### Explainability Components
        - **GradCAM** targeting `denseblock4.denselayer16.conv2` — 
          verifies model focuses on clinically correct anatomy
        - **MC Dropout** — N=20 stochastic forward passes with dropout enabled 
          at inference, producing HIGH / MEDIUM / LOW confidence tiers per label
        - **Severity-weighted triage** — score = Σ(probability × clinical severity weight)
        """
    )

# ── Results ─────────────────────────────────────────────────────────────────
with tabs[1]:
    col1, col2, col3 = st.columns(3)
    col1.metric("Validation AUC", "0.8320")
    col2.metric("Test AUC", "0.8311")
    col3.metric("Val/Test gap", "0.0009", help="Small gap = strong generalisation")

    st.markdown("---")
    st.markdown("### Per-Label AUC (test set)")

    st.markdown("**Strong performance** (AUC > 0.85)")
    strong_data = {
        "Pathology": ["Hernia", "Emphysema", "Cardiomegaly", "Edema", "Effusion", "Pneumothorax"],
        "Test AUC": [0.9550, 0.9176, 0.8981, 0.8912, 0.8830, 0.8728],
    }
    st.dataframe(strong_data, hide_index=True, use_container_width=True)

    st.markdown("**Moderate performance** (AUC 0.75–0.85)")
    moderate_data = {
        "Pathology": ["Mass", "Consolidation", "Atelectasis", "Fibrosis", "Pleural Thickening", "Nodule"],
        "Test AUC": [0.8388, 0.8134, 0.8081, 0.7864, 0.7844, 0.7591],
    }
    st.dataframe(moderate_data, hide_index=True, use_container_width=True)

    st.markdown("**Unreliable** (AUC < 0.75)")
    weak_data = {
        "Pathology": ["Pneumonia", "Infiltration"],
        "Test AUC": [0.7395, 0.6874],
        "Reason": [
            "Only ~1,000 positive cases in training — model defaults to negative",
            "Vague clinical term, overlaps with Consolidation/Pneumonia/Edema; ~20% label noise"
        ],
    }
    st.dataframe(weak_data, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Logit Distribution (calibration proof)")
    st.info(
        "After Asymmetric Focal Loss training:\n\n"
        "- Mean logit: **-0.755**  |  Std: **0.647**  |  Range: **-3.68 to +2.14**\n"
        "- No extreme tails → temperature scaling converges cleanly\n"
        "- Per-label temperatures near 1.0 confirm AFL produced well-calibrated outputs"
    )

# ── Limitations ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.error(
        "### ⚠ Important Caveats\n\n"
        "TRUST-X is a **Academic prototype** built on academic data. It must not be "
        "used for clinical diagnosis or treatment decisions."
    )

    st.markdown(
        """
        ### What the AUC actually measures
        - AUC 0.83 means the model correctly **ranks** a sick patient above a healthy 
          one 83% of the time — across thousands of image pairs
        - This is **NOT** the same as "83% per-image accuracy"
        - For any single X-ray, per-label accuracy varies by pathology:
          - Strong labels (Hernia, Emphysema, Cardiomegaly): ~8/10 correct
          - Weak labels (Infiltration, Pneumonia): ~2–4/10 correct

        ### Dataset limitations we inherit
        - **NIH label noise** — labels extracted from radiology reports via NLP. 
          Estimated 10–20% label error rate. Our model reflects this noise.
        - **Rare classes unreliable** — Hernia has only ~150 positive training examples 
          out of 78,000 images. No loss function can manufacture signal that doesn't 
          exist in the data.
        - **No external validation** — not tested on CheXpert, MIMIC-CXR, or any 
          non-NIH distribution.

        ### Appropriate use case
        - **✅ Triage prioritisation** at population scale — rank a queue of X-rays 
          by urgency so radiologists read high-risk cases first
        - **❌ Point-of-care diagnosis** — the probabilities must not be used to 
          confirm or rule out any specific condition for any individual patient
        """
    )

# ── References ──────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown(
        """
        ### Papers

        - **Wang et al.** (2017) — *ChestX-ray8: Hospital-scale Chest X-ray Database 
          and Benchmarks on Weakly-Supervised Classification and Localization of 
          Common Thorax Diseases.* CVPR 2017. _(NIH dataset)_
        
        - **Rajpurkar et al.** (2017) — *CheXNet: Radiologist-Level Pneumonia 
          Detection on Chest X-Rays with Deep Learning.* _(Baseline for comparison)_
        
        - **Huang et al.** (2017) — *Densely Connected Convolutional Networks.* 
          CVPR 2017. _(DenseNet architecture)_

        - **Ben-Baruch et al.** (2021) — *Asymmetric Loss for Multi-Label 
          Classification.* ICCV 2021. _(The AFL that fixed our calibration problem)_

        - **Selvaraju et al.** (2017) — *Grad-CAM: Visual Explanations from Deep 
          Networks via Gradient-based Localization.* ICCV 2017. _(Explainability method)_

        - **Gal & Ghahramani** (2016) — *Dropout as a Bayesian Approximation: 
          Representing Model Uncertainty in Deep Learning.* ICML 2016. 
          _(MC Dropout foundation)_

        ### Related context

        - **Caboolture Hospital Investigation** (2018, Queensland Health) — 
          the real-world radiology backlog incident that motivated this triage 
          framing.
        """
    )

st.markdown("---")
st.caption("TRUST-X — Academic project. Source code & model available for academic use.")
