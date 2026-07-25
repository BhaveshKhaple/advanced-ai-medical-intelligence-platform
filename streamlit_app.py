"""
Self-contained Streamlit app (runs the model in-process).
Works both locally and on Streamlit Community Cloud - no separate API server needed.
Reuses the self-contained modules and bundled model in hf_space/.
"""
import os
import sys
import uuid
import tempfile

# Make the self-contained modules + bundled model in hf_space/ importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_space"))

import streamlit as st
from inference import predict_image, generate_gradcam
from report import generate_report
from database import save_prediction, get_history

st.set_page_config(page_title="AI Medical Intelligence Platform", page_icon="🫁", layout="wide")

st.title("🫁 Advanced AI Medical Intelligence Platform")
st.caption("Chest X-Ray Analysis · Pneumonia Detection · Explainable AI (Grad-CAM) · LLM Report Generation")

with st.expander("ℹ️ About this project", expanded=False):
    st.markdown("""
    End-to-end deep learning system that:
    - **Detects pneumonia** from chest X-rays using a fine-tuned **EfficientNet-B0** (97.77% val accuracy)
    - **Explains** predictions with **Grad-CAM** heatmaps
    - **Generates** structured radiological reports via **Google Gemini**
    - Persists all predictions in a **SQLite** database

    Built by **Bhavesh Khaple** · [GitHub Repo](https://github.com/BhaveshKhaple/advanced-ai-medical-intelligence-platform)

    _Disclaimer: This is a technical demonstration, not a certified medical device. Not for clinical use._
    """)

tab1, tab2 = st.tabs(["🔬 Analyze X-Ray", "📋 History"])
UPLOAD_DIR = tempfile.gettempdir()

with tab1:
    st.subheader("Upload a Chest X-Ray Image")
    uploaded = st.file_uploader("JPG, PNG, or JPEG", type=["jpg", "jpeg", "png"])

    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Uploaded X-Ray", use_container_width=True)

        if st.button("🚀 Analyze", use_container_width=True, type="primary"):
            with st.spinner("Running inference + Grad-CAM + report generation..."):
                ext = os.path.splitext(uploaded.name)[-1] or ".jpg"
                tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getvalue())

                result = predict_image(tmp_path)
                gradcam_path = generate_gradcam(tmp_path)
                report = generate_report(
                    result["diagnosis"], result["confidence"],
                    result["probabilities"]["NORMAL"],
                    result["probabilities"]["PNEUMONIA"],
                )
                save_prediction(
                    image_filename=uploaded.name,
                    gradcam_path=gradcam_path,
                    diagnosis=result["diagnosis"],
                    confidence=result["confidence"],
                    llm_report=report,
                )

            with col2:
                st.image(gradcam_path, caption="Grad-CAM Heatmap", use_container_width=True)

            diag = result["diagnosis"]
            conf = result["confidence"]
            color = "🔴" if diag == "PNEUMONIA" else "🟢"
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Diagnosis", f"{color} {diag}")
            c2.metric("Confidence", f"{conf}%")
            c3.metric("PNEUMONIA prob", f"{result['probabilities']['PNEUMONIA']}%")
            st.progress(result["probabilities"]["PNEUMONIA"] / 100)
            st.caption(f"NORMAL: {result['probabilities']['NORMAL']}%  |  "
                       f"PNEUMONIA: {result['probabilities']['PNEUMONIA']}%")
            st.markdown("---")
            st.subheader("📄 AI-Generated Medical Report")
            st.markdown(report)

with tab2:
    st.subheader("Prediction History")
    if st.button("🔄 Refresh"):
        st.rerun()
    records = get_history()
    if not records:
        st.info("No predictions yet. Analyze an X-ray in the first tab.")
    for r in records:
        with st.expander(f"#{r.id} · {r.diagnosis} ({r.confidence}%) · {r.image_filename} · {r.created_at}"):
            cols = st.columns(2)
            with cols[0]:
                if r.gradcam_path and os.path.exists(r.gradcam_path):
                    st.image(r.gradcam_path, caption="Grad-CAM", use_container_width=True)
            with cols[1]:
                st.markdown(r.llm_report or "_No report._")
