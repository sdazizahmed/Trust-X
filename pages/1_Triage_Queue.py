"""
Page 1 — Triage Queue (receptionist intake).
Upload multiple X-rays, produce ranked worklist.
"""
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import base64, io
import tempfile


from trustx_core import predict, triage_score, priority_tier

st.set_page_config(page_title="TRUST-X | Triage Queue", page_icon="🫁", layout="wide")

# ── Disclaimer banner ────────────────────────────────────────────────────────
st.warning(
    "⚠ **Academic Prototype** — Not validated for clinical use. "
    "Predictions must not be used for diagnosis."
)

# ── Require model to be loaded via landing page ─────────────────────────────
if not st.session_state.get("model_loaded", False):
    st.error("Model not loaded. Please return to the **Home** page first.")
    st.stop()

# Rebuild cached model handle (same @st.cache_resource call as app.py)
from app import get_model
model, meta, infer_tfm, temperatures = get_model()

# ── Page content ────────────────────────────────────────────────────────────
st.title("📋 Stage 1 — Triage Queue")
st.caption("Receptionist intake: upload multiple X-rays, receive a ranked worklist.")

# ── Upload UI ───────────────────────────────────────────────────────────────
col_up, col_info = st.columns([3, 1])
with col_up:
    uploaded_files = st.file_uploader(
        "Upload chest X-ray images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload 2 or more X-rays to see ranking in action."
    )

with col_info:
    st.info(
        "Scoring formula:\n\n"
        "**Score = Σ (probability × severity)**\n\n"
        "Higher = more urgent."
    )

# ── Processing ──────────────────────────────────────────────────────────────
if uploaded_files:
    if st.button(f"🚀 Run Triage on {len(uploaded_files)} X-rays", type="primary"):
        queue = {}
        progress = st.progress(0, text="Processing…")

        for i, up in enumerate(uploaded_files):
            # Save to temp
            tmp_dir = Path(tempfile.gettempdir()) / "trustx_uploads"
            tmp_dir.mkdir(exist_ok=True, parents=True)
            tmp_path = tmp_dir / up.name
            tmp_path.write_bytes(up.getbuffer())

            # Predict & score
            img = Image.open(tmp_path).convert('RGB')
            probs = predict(model, img, infer_tfm, meta['device'],
                            temperatures=temperatures, calibrate=True)
            score = triage_score(probs)
            top_idx = int(probs.argmax())

            patient_id = f"P{i+1:03d}"

            # Generate base64 thumbnail for inline display in table
            thumb = img.copy()
            thumb.thumbnail((80, 80))
            buf = io.BytesIO()
            thumb.convert('RGB').save(buf, format='JPEG', quality=80)
            thumb_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

            queue[patient_id] = {
                'path'       : str(tmp_path),
                'filename'   : up.name,
                'score'      : score,
                'probs'      : probs,
                'top_finding': meta['label_cols'][top_idx],
                'top_prob'   : float(probs[top_idx]),
                'thumb_b64'  : thumb_b64,
            }
            progress.progress((i + 1) / len(uploaded_files),
                              text=f"Processed {up.name} ({i+1}/{len(uploaded_files)})")

        progress.empty()
        st.session_state.triage_queue = queue
        st.success(f"✅ Processed {len(queue)} patients. See ranked worklist below.")

# ── Display queue ───────────────────────────────────────────────────────────
queue = st.session_state.get("triage_queue", {})

if queue:
    st.markdown("## Ranked Worklist")
    st.caption("Highest priority at top. Note the Patient ID for drill-down on the **Review** page.")

    # Sort by score desc
    ranked = sorted(queue.items(), key=lambda kv: kv[1]['score'], reverse=True)

    # ── Summary metrics ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    scores = [info['score'] for _, info in ranked]
    urgent_count = sum(1 for s in scores if s > 8.0)
    high_count   = sum(1 for s in scores if 4.0 < s <= 8.0)

    m1.metric("Total patients", len(ranked))
    m2.metric("🔴 Urgent", urgent_count)
    m3.metric("🟠 High", high_count)
    m4.metric("Max score", f"{max(scores):.2f}")

    st.markdown("---")

    # ── HTML table with thumbnails + priority badges ────────────────────────
    html = [
        """
        <style>
          .triage-table { border-collapse: collapse; width: 100%; font-family: -apple-system, sans-serif; }
          .triage-table th { background: #1a4731; color: white; padding: 10px 12px; text-align: left; font-size: 13px; }
          .triage-table td { padding: 10px 12px; border-bottom: 1px solid #e0e0e0; font-size: 13px; vertical-align: middle; color: #1a1a1a; }
          .triage-table tr:nth-child(even) { background: #f8faf8; }
          .triage-table tr:hover { background: #e8f5ed; }
          .priority-badge { padding: 4px 10px; border-radius: 12px; color: white; font-weight: bold; font-size: 11px; white-space: nowrap; }
          .score-cell { font-weight: bold; font-size: 15px; color: #1a4731; }
          .thumb { border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
          .patient-id { font-family: monospace; font-weight: bold; color: #2d6a4f; font-size: 14px; }
        </style>
        <table class="triage-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Priority</th>
              <th>Patient</th>
              <th>Preview</th>
              <th>Score</th>
              <th>Top Finding</th>
              <th>Confidence</th>
              <th>Filename</th>
            </tr>
          </thead>
          <tbody>
        """
    ]

    for rank, (pid, info) in enumerate(ranked, start=1):
        tier_label, tier_color = priority_tier(info['score'])
        emoji = {'URGENT': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'ROUTINE': '🟢'}[tier_label]
        html.append(f"""
          <tr>
            <td><b>{rank}</b></td>
            <td><span class="priority-badge" style="background:{tier_color};">
                {emoji} {tier_label}</span></td>
            <td class="patient-id">{pid}</td>
            <td><img class="thumb" src="data:image/jpeg;base64,{info['thumb_b64']}" width="70"/></td>
            <td class="score-cell">{info['score']:.2f}</td>
            <td>{info['top_finding']}</td>
            <td>{info['top_prob']:.3f}</td>
            <td style="color:#666; font-size: 11px;">{info['filename']}</td>
          </tr>
        """)

    html.append("</tbody></table>")
    st.markdown("".join(html), unsafe_allow_html=True)

    st.markdown("---")
    st.info(
        "👉 **Next step:** Go to the **Review** page in the sidebar to drill into "
        "any patient's full analysis (GradCAM + uncertainty)."
    )

    # ── Clear queue button ──────────────────────────────────────────────────
    if st.button("🗑 Clear Queue"):
        st.session_state.triage_queue = {}
        st.rerun()

else:
    st.info("Upload X-rays above and click **Run Triage** to build the queue.")
