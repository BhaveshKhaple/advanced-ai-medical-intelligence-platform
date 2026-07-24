import streamlit as st
import requests
import os
from PIL import Image
import io

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Medical Intelligence Platform",
    page_icon="🫁",
    layout="wide",
)

st.title("🫁 Advanced AI Medical Intelligence Platform")
st.caption("Chest X-Ray Analysis · Pneumonia Detection · Explainable AI · LLM Report Generation")

tab1, tab2 = st.tabs(["🔬 Analyze X-Ray", "📋 History"])

# ─── Tab 1: Analysis ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload a Chest X-Ray Image")
    uploaded = st.file_uploader(
        "Supported formats: JPG, PNG, JPEG",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded:
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded, caption="Uploaded X-Ray", use_container_width=True)

        if st.button("🚀 Analyze", use_container_width=True, type="primary"):
            with st.spinner("Running deep learning inference + Grad-CAM + LLM report..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    response = requests.post(f"{API_URL}/predict", files=files, timeout=120)

                    if response.status_code == 200:
                        data = response.json()

                        with col2:
                            # Grad-CAM
                            gradcam_url = f"{API_URL}{data['gradcam_url']}"
                            gcam_resp = requests.get(gradcam_url)
                            if gcam_resp.status_code == 200:
                                gcam_img = Image.open(io.BytesIO(gcam_resp.content))
                                st.image(gcam_img, caption="Grad-CAM Heatmap", use_container_width=True)

                        # Diagnosis card
                        diag = data["diagnosis"]
                        conf = data["confidence"]
                        color = "🔴" if diag == "PNEUMONIA" else "🟢"

                        st.markdown("---")
                        col_d1, col_d2, col_d3 = st.columns(3)
                        with col_d1:
                            st.metric("Diagnosis", f"{color} {diag}")
                        with col_d2:
                            st.metric("Confidence", f"{conf}%")
                        with col_d3:
                            probs = data["probabilities"]
                            st.metric("PNEUMONIA prob", f"{probs['PNEUMONIA']}%")

                        # Probabilities bar
                        st.progress(probs["PNEUMONIA"] / 100)
                        st.caption(f"NORMAL: {probs['NORMAL']}%  |  PNEUMONIA: {probs['PNEUMONIA']}%")

                        # LLM Report
                        st.markdown("---")
                        st.subheader("📄 AI-Generated Medical Report")
                        st.markdown(data["report"])
                        st.caption(f"Prediction ID: {data['id']} | Timestamp: {data['created_at']}")

                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to FastAPI backend. Make sure it is running on port 8000.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# ─── Tab 2: History ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Prediction History")

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/history", timeout=10)
        if resp.status_code == 200:
            records = resp.json()
            if not records:
                st.info("No predictions yet. Upload an X-ray in the Analyze tab.")
            else:
                for rec in records:
                    with st.expander(
                        f"#{rec['id']} · {rec['diagnosis']} ({rec['confidence']}%) · {rec['image_filename']} · {rec['created_at']}"
                    ):
                        cols = st.columns(2)
                        with cols[0]:
                            if rec.get("gradcam_url"):
                                gcam_url = f"{API_URL}{rec['gradcam_url']}"
                                gcam_resp = requests.get(gcam_url)
                                if gcam_resp.status_code == 200:
                                    img = Image.open(io.BytesIO(gcam_resp.content))
                                    st.image(img, caption="Grad-CAM", use_container_width=True)
                        with cols[1]:
                            report_resp = requests.get(f"{API_URL}/report/{rec['id']}")
                            if report_resp.status_code == 200:
                                st.markdown(report_resp.json().get("llm_report", "No report."))
        else:
            st.error("Failed to fetch history.")
    except Exception as e:
        st.error(f"Cannot connect to backend: {e}")
