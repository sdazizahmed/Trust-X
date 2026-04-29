# TRUST-X: Chest X-Ray Triage System

TRUST-X is a clinical decision support tool built on top of a fine-tuned DenseNet-121 model trained on the NIH ChestX-ray14 dataset (~112K images, 14 pathology labels). The goal is to help radiologists prioritize their worklist — not replace their judgment.

The app has three stages that mirror a real clinical workflow:

**1. Triage Queue** — Upload a batch of X-rays. The model scores each image for all 14 pathologies and ranks patients by severity using a weighted triage score, so the most urgent cases surface to the top automatically.

**2. Review** — Drill into any individual patient. This page runs the full analysis: per-label predictions with F1-optimized thresholds, GradCAM heatmaps showing which regions drove each prediction, and MC Dropout uncertainty estimation (20 forward passes) that outputs a HIGH / MEDIUM / LOW confidence rating per finding.

**3. About** — Model card covering architecture choices, training methodology, evaluation results (test AUC 0.8311), known limitations, and references.

---

## Model

The backbone is DenseNet-121 trained in two phases: feature extraction on ImageNet weights, then full fine-tuning with asymmetric focal loss to handle the severe class imbalance in ChestX-ray14 (some pathologies appear in fewer than 0.5% of images). Images are resized to 320px for inference.

Per-label classification thresholds were tuned on a held-out validation set rather than using a fixed 0.5 cutoff — this meaningfully improves recall on rare classes.

The train/val/test split is done at the patient level (no patient appears in more than one split), which avoids the data leakage that affects the original NIH-provided split.

---

## Setup

You'll need three files that aren't in this repo — place them in the root directory alongside `app.py`:

- `TRUSTX_densenet121_v2_320px_AFL.pth` — the trained model checkpoint
- `val_logits.npy` — validation logits used for threshold calibration
- `val_labels.npy` — corresponding validation labels

These are produced by the training notebook.

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploying

The app runs on CPU. DenseNet-121 at 320px takes roughly 1–2 seconds per image for a standard prediction; MC Dropout (20 passes) takes 20–40 seconds and shows a spinner. Streamlit Community Cloud's 1 GB RAM limit is sufficient.

The model is cached after the first load via `@st.cache_resource` — subsequent interactions within a session are fast.

### Getting the checkpoint onto Streamlit Cloud

The `.pth` file is ~30 MB — too large for a plain Git push. Two options:

**Git LFS**
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes TRUSTX_densenet121_v2_320px_AFL.pth
git commit -m "Add model checkpoint"
```

**Hugging Face Hub** (recommended — keeps the repo clean)

Upload the `.pth` to a free HF model repo, then add a download step at the top of `get_model()` in `app.py`:

```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(
    repo_id="your-username/trustx",
    filename="TRUSTX_densenet121_v2_320px_AFL.pth"
)
```

Add `huggingface_hub` to `requirements.txt`. The `.npy` files are small enough to commit directly.

---

## Running a demo

Good test images are the per-pathology samples exported by the `demo_images/` script in the training notebook (one image per pathology from the held-out test set).

1. Open the app — the home page confirms the model loaded and shows the test AUC.
2. Go to **Triage Queue**, upload 5–10 X-rays, and hit "Run Triage" to see the ranked worklist.
3. Pick any patient from **Review**, hit "Run Full Analysis" to see predictions, GradCAM overlays, and uncertainty ratings.
4. Check **About** for the full methodology writeup.
