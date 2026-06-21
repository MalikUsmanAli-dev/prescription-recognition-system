"""
app.py
-------
AI-Assisted Prescription Recognition and Medicine Retrieval System
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from modules import image_processing as dip
from modules import ui_components as ui
from modules.groq_integration import GroqPrescriptionAnalyzer, DEFAULT_MODEL
from modules.pdf_generator import generate_pdf_report

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

load_dotenv()

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="MediScan AI | Prescription Recognition System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_global_css()

# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "uploaded_file_bytes": None,
    "original_image": None,
    "file_metadata": None,
    "pipeline_result": None,
    "analysis_result": None,
    "analysis_input_label": "Original",
    "groq_api_key": os.getenv("GROQ_API_KEY", ""),
    "groq_model": DEFAULT_MODEL,
    "current_page": "Dashboard",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_session():
    for key, value in DEFAULTS.items():
        if key not in ("groq_api_key", "groq_model"):
            st.session_state[key] = value


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="font-size:2rem;">💊</div>
            <div>
                <div style="font-weight:800;font-size:1.05rem;color:#1A2B3C;">MediScan AI</div>
                <div style="font-size:0.72rem;color:#6B7C93;">Prescription Recognition System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    pages = ["Dashboard", "Analytics", "Settings", "About"]
    icons = ["speedometer2", "bar-chart-line", "gear", "info-circle"]

    if HAS_OPTION_MENU:
        selected_page = option_menu(
            menu_title=None,
            options=pages,
            icons=icons,
            default_index=pages.index(st.session_state.current_page),
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": "#0E7C7B", "font-size": "16px"},
                "nav-link": {
                    "font-size": "14px",
                    "font-weight": "600",
                    "text-align": "left",
                    "margin": "3px 0",
                    "border-radius": "10px",
                    "padding": "10px 12px",
                },
                "nav-link-selected": {"background-color": "#0E7C7B", "color": "white"},
            },
        )
    else:
        selected_page = st.radio("Navigation", pages, index=pages.index(st.session_state.current_page))

    st.session_state.current_page = selected_page

    st.markdown("---")
    st.markdown("##### 🔌 System Status")
    api_ready = bool(st.session_state.groq_api_key)
    if api_ready:
        st.markdown("🟢 **Groq API:** Connected")
    else:
        st.markdown("🟡 **Groq API:** Not configured")
        st.caption("Add your key in the Settings page.")

    if st.session_state.pipeline_result:
        st.markdown("🟢 **Image Pipeline:** Processed")
    else:
        st.markdown("⚪ **Image Pipeline:** Idle")

    if st.session_state.analysis_result and st.session_state.analysis_result.success:
        st.markdown("🟢 **AI Analysis:** Complete")
    else:
        st.markdown("⚪ **AI Analysis:** Idle")

    st.markdown("---")
    if st.button("🔄 Start New Prescription", use_container_width=True):
        reset_session()
        st.rerun()

    st.caption("Built for Final Year Project demonstration · v1.0")


# --------------------------------------------------------------------------- #
# DASHBOARD PAGE
# --------------------------------------------------------------------------- #
def render_dashboard():
    ui.render_hero(
        "AI-Assisted Prescription Recognition System",
        "Upload a prescription photo to automatically extract medicines, dosages, "
        "and usage instructions using Digital Image Processing + Groq Llama-4 Vision AI.",
        chips=["OpenCV DIP Pipeline", "Groq Vision AI", "Structured JSON Output", "PDF Reports"],
    )

    # ---------------- Step 1: Upload ---------------- #
    ui.render_section_header("📤", "Step 1 — Upload Prescription Image", "Supported formats: JPG, PNG, JPEG")
    upload_col, preview_col = st.columns([1.1, 1])

    with upload_col:
        st.markdown('<div class="pr-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag & drop or browse a prescription image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            if file_bytes != st.session_state.uploaded_file_bytes:
                # New file uploaded -> reset downstream state
                st.session_state.uploaded_file_bytes = file_bytes
                st.session_state.original_image = Image.open(uploaded_file)
                st.session_state.file_metadata = dip.get_image_metadata(
                    file_bytes, st.session_state.original_image
                )
                st.session_state.pipeline_result = None
                st.session_state.analysis_result = None
                st.success("Prescription image uploaded successfully.")
        elif st.session_state.original_image is None:
            st.info("Upload a prescription image to begin the analysis pipeline.")
        st.markdown("</div>", unsafe_allow_html=True)

    with preview_col:
        if st.session_state.original_image is not None:
            st.markdown('<div class="pr-card"><h4>📋 Image Preview & Metadata</h4>', unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)
            meta = st.session_state.file_metadata
            m1, m2 = st.columns(2)
            m1.metric("Dimensions", meta["Dimensions"])
            m2.metric("File Size", meta["File Size"])
            m3, m4 = st.columns(2)
            m3.metric("Color Mode", meta["Color Mode"])
            m4.metric("Megapixels", meta["Megapixels"])
            st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.original_image is None:
        return  # Nothing further to show until an image exists

    # ---------------- Step 2: DIP Pipeline ---------------- #
    st.markdown("<br>", unsafe_allow_html=True)
    ui.render_section_header(
        "🧪", "Step 2 — Digital Image Processing Pipeline",
        "Grayscale → Median Filter → CLAHE Contrast Enhancement → Sharpening → Adaptive Thresholding"
    )

    run_pipeline_clicked = st.button("▶️ Run Image Processing Pipeline", type="primary")
    if run_pipeline_clicked:
        with st.spinner("Processing prescription image through the DIP pipeline..."):
            time.sleep(0.3)  # tiny UX pause so the spinner is visible even on fast machines
            st.session_state.pipeline_result = dip.run_pipeline(st.session_state.original_image)
        st.success(
            f"Image processing complete in {st.session_state.pipeline_result.total_time_ms:.1f} ms."
        )

    if st.session_state.pipeline_result:
        result = st.session_state.pipeline_result
        captions = {
            "Original": "Raw uploaded prescription",
            "Grayscale": "Single-channel conversion",
            "Noise Reduced": "Median filter denoising",
            "Enhanced": "CLAHE contrast enhancement",
            "Final Processed": "Sharpened + adaptive threshold",
        }
        cols = st.columns(5)
        for col, (stage_name, stage_img) in zip(cols, result.stages.items()):
            with col:
                st.markdown('<div class="pr-card">', unsafe_allow_html=True)
                ui.render_stage_image_card(stage_img, stage_name, captions.get(stage_name, ""))
                st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("⏱️ View per-stage processing time"):
            timing_df = pd.DataFrame(
                {"Stage": list(result.timings_ms.keys()), "Time (ms)": [round(v, 3) for v in result.timings_ms.values()]}
            )
            st.dataframe(timing_df, use_container_width=True, hide_index=True)

    else:
        return  # Don't show AI step until pipeline has run

    # ---------------- Step 3: AI Analysis ---------------- #
    st.markdown("<br>", unsafe_allow_html=True)
    ui.render_section_header("🤖", "Step 3 — AI Prescription Analysis (Groq Vision)",
                              "Choose which processed image version to send to the AI model")

    st.markdown('<div class="pr-card">', unsafe_allow_html=True)
    img_choice = st.radio(
        "Image to analyze",
        options=["Original", "Enhanced", "Final Processed"],
        index=["Original", "Enhanced", "Final Processed"].index(st.session_state.analysis_input_label),
        horizontal=True,
        help="Tip: 'Original' often yields the best AI accuracy for vision models; the "
             "binarized 'Final Processed' image is best for classic OCR engines.",
    )
    st.session_state.analysis_input_label = img_choice

    analyze_clicked = st.button("✨ Analyze with Groq AI", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_clicked:
        if not st.session_state.groq_api_key:
            st.error("⚠️ No Groq API key configured. Please add one in the **Settings** page.")
        else:
            image_to_send = st.session_state.pipeline_result.stages[img_choice]
            try:
                with st.spinner("Groq is reading the prescription... this can take a few seconds."):
                    analyzer = GroqPrescriptionAnalyzer(
                        api_key=st.session_state.groq_api_key,
                        model=st.session_state.groq_model,
                    )
                    st.session_state.analysis_result = analyzer.analyze(image_to_send)
            except Exception as exc:  # noqa: BLE001
                st.error(f"⚠️ Failed to initialize Groq client: {exc}")

    analysis = st.session_state.analysis_result
    if analysis is None:
        return

    if not analysis.success:
        st.error(f"⚠️ Analysis failed: {analysis.error_message}")
        if analysis.raw_response:
            with st.expander("Show raw AI response"):
                st.code(analysis.raw_response)
        return

    st.success(f"✅ AI analysis complete in {analysis.processing_time_s:.2f} seconds.")
    if analysis.prescription_quality.lower() == "poor":
        st.warning("⚠️ The prescription image quality was assessed as **Poor** — results may be unreliable. "
                   "Consider re-uploading a clearer photo.")

    # ---------------- Step 4: Medicine table ---------------- #
    st.markdown("<br>", unsafe_allow_html=True)
    ui.render_section_header("💊", "Step 4 — Extracted Medicines", "Search and filter recognized medicines below")

    if analysis.medicines:
        med_df = pd.DataFrame(
            [
                {
                    "Medicine Name": m.name,
                    "Strength": m.strength or "—",
                    "Frequency": m.frequency or "—",
                    "Confidence (%)": m.confidence,
                }
                for m in analysis.medicines
            ]
        )

        search_col, filter_col = st.columns([2, 1])
        with search_col:
            search_term = st.text_input("🔍 Search medicine name", placeholder="e.g. Panadol, Augmentin...")
        with filter_col:
            min_conf = st.slider("Minimum confidence", 0, 100, 0)

        filtered_df = med_df.copy()
        if search_term:
            filtered_df = filtered_df[filtered_df["Medicine Name"].str.contains(search_term, case=False, na=False)]
        filtered_df = filtered_df[filtered_df["Confidence (%)"] >= min_conf]

        st.markdown('<div class="pr-card">', unsafe_allow_html=True)
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence (%)": st.column_config.ProgressColumn(
                    "Confidence (%)", min_value=0, max_value=100, format="%d%%"
                )
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No medicines were confidently identified in this prescription.")

    if analysis.doctor_notes:
        st.markdown('<div class="pr-card"><h4>📝 Additional Notes</h4>', unsafe_allow_html=True)
        st.write(analysis.doctor_notes)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Step 5: KPIs ---------------- #
    st.markdown("<br>", unsafe_allow_html=True)
    ui.render_section_header("📈", "Step 5 — Analysis Summary")

    quality_metrics = dip.estimate_prescription_quality(
        dip.to_grayscale(dip.pil_to_cv2(st.session_state.original_image))
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui.render_kpi_card("💊", str(len(analysis.medicines)), "Medicines Detected")
    with k2:
        total_time = st.session_state.pipeline_result.total_time_ms / 1000 + analysis.processing_time_s
        ui.render_kpi_card("⏱️", f"{total_time:.2f}s", "Total Processing Time")
    with k3:
        ui.render_kpi_card("🎯", f"{analysis.overall_confidence}%", "Recognition Confidence")
    with k4:
        ui.render_kpi_card("📋", analysis.prescription_quality, "Prescription Quality")

    # ---------------- Step 6: Report generation ---------------- #
    st.markdown("<br>", unsafe_allow_html=True)
    ui.render_section_header("📄", "Step 6 — Download Report", "Export a complete PDF summary of this analysis")

    st.markdown('<div class="pr-card">', unsafe_allow_html=True)
    patient_ref = st.text_input("Reference / Patient ID (optional)", placeholder="e.g. OPD-2026-0142")
    if st.button("📥 Generate PDF Report", type="primary"):
        with st.spinner("Compiling PDF report..."):
            pdf_bytes = generate_pdf_report(
                original_image=st.session_state.original_image,
                processed_image=st.session_state.pipeline_result.stages["Final Processed"],
                medicines=[
                    {
                        "name": m.name,
                        "strength": m.strength,
                        "frequency": m.frequency,
                        "confidence": m.confidence,
                    }
                    for m in analysis.medicines
                ],
                overall_confidence=analysis.overall_confidence,
                prescription_quality=analysis.prescription_quality,
                processing_time_s=total_time,
                doctor_notes=analysis.doctor_notes,
                patient_reference=patient_ref or None,
            )
        st.download_button(
            "⬇️ Download Report (PDF)",
            data=pdf_bytes,
            file_name=f"prescription_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# ANALYTICS PAGE
# --------------------------------------------------------------------------- #
def render_analytics():
    ui.render_hero(
        "Analytics Dashboard",
        "Visual insights into the most recent prescription analysis session.",
        chips=["Confidence Breakdown", "Processing Time", "Quality Assessment"],
    )

    analysis = st.session_state.analysis_result
    pipeline = st.session_state.pipeline_result

    if not analysis or not analysis.success or not pipeline:
        st.info("ℹ️ Run a full analysis from the **Dashboard** page first to populate analytics.")
        return

    k1, k2, k3, k4 = st.columns(4)
    total_time = pipeline.total_time_ms / 1000 + analysis.processing_time_s
    with k1:
        ui.render_kpi_card("💊", str(len(analysis.medicines)), "Medicines Detected")
    with k2:
        ui.render_kpi_card("⏱️", f"{total_time:.2f}s", "Total Processing Time")
    with k3:
        ui.render_kpi_card("🎯", f"{analysis.overall_confidence}%", "Overall Confidence")
    with k4:
        ui.render_kpi_card("📋", analysis.prescription_quality, "Prescription Quality")

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="pr-card"><h4>💊 Per-Medicine Confidence</h4>', unsafe_allow_html=True)
        if analysis.medicines:
            names = [m.name for m in analysis.medicines]
            confs = [m.confidence for m in analysis.medicines]
            colors = ["#1FA97C" if c >= 80 else "#E2A33D" if c >= 50 else "#E2543D" for c in confs]
            fig = go.Figure(go.Bar(x=confs, y=names, orientation="h", marker_color=colors))
            fig.update_layout(
                height=max(260, 40 * len(names)),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Confidence (%)",
                xaxis_range=[0, 100],
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No medicines to chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="pr-card"><h4>🎯 Overall Recognition Confidence</h4>', unsafe_allow_html=True)
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=analysis.overall_confidence,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0E7C7B"},
                    "steps": [
                        {"range": [0, 50], "color": "#FBEAE6"},
                        {"range": [50, 80], "color": "#FCF2DE"},
                        {"range": [80, 100], "color": "#E4F5EE"},
                    ],
                },
            )
        )
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pr-card"><h4>⏱️ Processing Time Breakdown</h4>', unsafe_allow_html=True)
    timing_labels = list(pipeline.timings_ms.keys()) + ["Groq AI Analysis"]
    timing_values = [round(v, 1) for v in pipeline.timings_ms.values()] + [round(analysis.processing_time_s * 1000, 1)]
    fig = go.Figure(go.Bar(x=timing_labels, y=timing_values, marker_color="#2F6FED"))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Time (ms)",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pr-card"><h4>🖼️ Local Image Quality Heuristics</h4>', unsafe_allow_html=True)
    quality_metrics = dip.estimate_prescription_quality(
        dip.to_grayscale(dip.pil_to_cv2(st.session_state.original_image))
    )
    q1, q2, q3 = st.columns(3)
    q1.metric("Sharpness (Laplacian Var.)", quality_metrics["sharpness"])
    q2.metric("Brightness (Mean Pixel)", quality_metrics["brightness"])
    q3.metric("Contrast (Std. Dev.)", quality_metrics["contrast"])
    st.caption(
        "These are computed locally via OpenCV (no AI call) and complement the AI-assessed "
        "prescription quality label above."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# SETTINGS PAGE
# --------------------------------------------------------------------------- #
def render_settings():
    ui.render_hero("Settings", "Configure your Groq API connection and model preferences.", chips=["API Key", "Model"])

    st.markdown('<div class="pr-card"><h4>🔑 Groq API Configuration</h4>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="Paste your Groq API key here (starts with gsk_...)",
        help="Get a free key from https://console.groq.com/keys. "
             "You can also set it permanently via a GROQ_API_KEY environment variable / .env file.",
    )
    model_choice = st.selectbox(
        "Groq Model",
        options=[
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
        ],
        index=0,
        help="'Scout' is fast and cost-effective; 'Maverick' is the larger model and can be more "
             "accurate on messy handwriting at slightly higher latency.",
    )

    if st.button("💾 Save Settings", type="primary"):
        st.session_state.groq_api_key = api_key_input
        st.session_state.groq_model = model_choice
        if api_key_input:
            st.success("✅ Settings saved. You're ready to analyze prescriptions.")
        else:
            st.warning("⚠️ No API key entered — AI analysis will be unavailable until one is provided.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pr-card"><h4>ℹ️ How to get a Groq API Key</h4>', unsafe_allow_html=True)
    st.markdown(
        """
        1. Visit **https://console.groq.com/keys**
        2. Sign in / create a free account
        3. Click **Create API Key**
        4. Copy the key and paste it above (or place it in a `.env` file as `GROQ_API_KEY=...`)
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# ABOUT PAGE
# --------------------------------------------------------------------------- #
def render_about():
    ui.render_hero("About This Project", "Final Year Project — AI-Assisted Prescription Recognition System",
                    chips=["Python", "Streamlit", "OpenCV", "Groq Vision"])

    st.markdown(
        """
        <div class="pr-card">
        <h4>🎯 Project Overview</h4>
        This system automates the error-prone process of manually reading handwritten or printed
        medical prescriptions. It combines classical <b>Digital Image Processing</b> techniques with
        a modern <b>multimodal AI vision model</b> (Groq-hosted Llama-4) to extract medicine names, dosage
        strengths, and usage instructions in a structured, searchable format — reducing dispensing
        errors at pharmacy counters and supporting digital health record-keeping.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="pr-card">
            <h4>🧪 Image Processing Pipeline</h4>
            <ol>
            <li>Grayscale conversion</li>
            <li>Median filter noise reduction</li>
            <li>CLAHE contrast enhancement</li>
            <li>Unsharp-mask sharpening</li>
            <li>Adaptive Gaussian thresholding</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="pr-card">
            <h4>🛠️ Technology Stack</h4>
            <ul>
            <li>Streamlit — UI / dashboard</li>
            <li>OpenCV + NumPy + Pillow — image processing</li>
            <li>Groq (groq SDK) — Llama-4 vision AI, OpenAI-compatible</li>
            <li>Pandas + Plotly — data & analytics</li>
            <li>ReportLab — PDF report generation</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="pr-card">
        <h4>⚠️ Disclaimer</h4>
        This application is built for academic / portfolio demonstration purposes. AI-extracted
        medicine information must always be verified by a licensed pharmacist or physician before
        any clinical or dispensing decision is made.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
PAGE_RENDERERS = {
    "Dashboard": render_dashboard,
    "Analytics": render_analytics,
    "Settings": render_settings,
    "About": render_about,
}

PAGE_RENDERERS[st.session_state.current_page]()
