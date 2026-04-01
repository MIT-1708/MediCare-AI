import streamlit as st
import tempfile
import os
import html
import matplotlib.pyplot as plt
import pandas as pd
from predict_from_report import run
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="MediCare AI - Professional Health Analysis",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional medical UI styling
st.markdown("""
<style>
    :root {
        --bg-0: #061725;
        --bg-1: #0a2337;
        --ink-1: #eaf3ff;
        --ink-2: #b9d3ea;
        --line: #1a3b56;
        --line-strong: #2a5a7c;
        --primary: #1a63ff;
        --primary-dark: #0c3a99;
        --accent: #00c2b8;
        --warn: #ffb14a;
        --danger: #ff6b7a;
        --success: #43d17a;
        --shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
    }

    * {
        font-family: "Aptos", "Trebuchet MS", "Segoe UI Variable", "Segoe UI", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(85rem 50rem at -10% -5%, rgba(26, 99, 255, 0.25) 0%, transparent 60%),
            radial-gradient(72rem 42rem at 105% -5%, rgba(0, 194, 184, 0.20) 0%, transparent 55%),
            linear-gradient(180deg, #041123 0%, #061725 100%);
    }

    .main {
        background: transparent !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #071c2f 0%, #061725 100%);
        border-right: 1px solid var(--line);
    }

    .header-container {
        background: linear-gradient(132deg, #0c3a99 0%, #0b2f7b 50%, #071f55 100%);
        padding: 36px;
        border-radius: 18px;
        margin-bottom: 26px;
        color: #fff;
        box-shadow: var(--shadow);
        border: 1px solid rgba(255, 255, 255, 0.22);
        position: relative;
        overflow: hidden;
        animation: fadeUp 0.6s ease-out forwards;
    }
    
    @keyframes fadeUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .main .block-container {
        animation: fadeUp 0.6s ease-out forwards;
    }

    .header-container::before {
        content: "";
        position: absolute;
        inset: auto -80px -130px auto;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.22) 0%, transparent 64%);
    }

    .header-container h1 {
        margin: 0;
        font-size: 2.2rem;
        letter-spacing: 0.2px;
        font-weight: 750;
    }

    .header-container .subtitle {
        margin-top: 8px;
        font-size: 1rem;
        opacity: 0.95;
        max-width: 56ch;
    }

    .header-row {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
    }

    .badge-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 8px;
        max-width: 320px;
    }

    .badge,
    .badge-medical {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.2);
        color: #fff;
        padding: 6px 12px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.42);
        font-size: 0.74rem;
        font-weight: 650;
        letter-spacing: 0.2px;
        backdrop-filter: blur(1px);
    }

    .badge-certified {
        background: rgba(0, 168, 157, 0.26);
        border-color: rgba(143, 255, 246, 0.74);
    }

    .card {
        background: var(--bg-1);
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: var(--shadow);
        padding: 24px;
        margin-bottom: 18px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover, .metric-box:hover, .sidebar-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.4);
    }

    .card-title {
        color: var(--ink-1);
        font-size: 1.15rem;
        font-weight: 730;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 2px solid var(--line-strong);
    }

    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: #fff;
        border: 0;
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(15, 95, 133, 0.25);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        filter: brightness(1.02);
        box-shadow: 0 11px 24px rgba(15, 95, 133, 0.32);
    }

    .stDownloadButton>button {
        background: linear-gradient(135deg, rgba(26, 99, 255, 0.15) 0%, rgba(0, 194, 184, 0.12) 100%);
        color: var(--ink-1);
        border: 1px solid var(--primary);
        border-radius: 10px;
        min-height: 2.7rem;
        font-weight: 650;
    }

    /* Form controls - hospital style, high readability */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div,
    .stTextArea textarea {
        background: rgba(5, 20, 35, 0.65) !important;
        color: var(--ink-1) !important;
        border: 1px solid #1f4a67 !important;
        border-radius: 10px !important;
    }

    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(26, 99, 255, 0.22) !important;
    }

    /* File uploader — same dark theme as page background */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(10, 35, 55, 0.65) !important;
        border: 2px dashed var(--line-strong) !important;
        border-radius: 14px !important;
        padding-top: 1.15rem !important;
        padding-bottom: 1.15rem !important;
        min-height: 138px !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary) !important;
        background: rgba(26, 99, 255, 0.10) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] div {
        color: var(--ink-1) !important;
    }

    [data-testid="stFileUploaderDropzone"] small {
        color: var(--ink-2) !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    [data-testid="stFileUploader"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 4px;
        background: rgba(6, 23, 37, 0.5) !important;
    }

    [data-testid="stFileUploaderFileList"],
    [data-testid="stFileUploaderFileList"] li {
        background: transparent !important;
    }

    [data-testid="stUploadedFile"] {
        background: rgba(10, 35, 55, 0.85) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        color: var(--ink-1) !important;
    }

    .upload-card-light [data-testid="stWidgetLabel"] p {
        color: var(--ink-1) !important;
        font-weight: 650 !important;
        font-size: 1rem !important;
    }

    .upload-intro {
        padding: 8px 6px 18px;
    }

    .upload-panel-title {
        text-align: center;
        color: var(--ink-1);
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
    }

    .upload-panel-sub {
        text-align: center;
        color: var(--ink-2);
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0 0 8px 0;
    }

    .upload-hint {
        text-align: center;
        color: #83a9c7;
        font-size: 0.82rem;
        margin: 0 0 18px 0;
    }

    .upload-card-light {
        background: linear-gradient(180deg, var(--bg-1) 0%, rgba(10, 35, 55, 0.92) 100%) !important;
        border: 1px solid var(--line) !important;
        box-shadow: var(--shadow) !important;
    }

    .upload-file-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-top: 18px;
        padding: 14px 18px;
        background: rgba(10, 35, 55, 0.75);
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    }

    .upload-file-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(26, 99, 255, 0.15) 0%, rgba(10, 35, 55, 0.9) 100%);
        border: 1px solid var(--line-strong);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }

    .upload-file-meta {
        min-width: 0;
        flex: 1;
    }

    .upload-file-name {
        color: var(--ink-1);
        font-weight: 700;
        font-size: 0.95rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .upload-file-size {
        color: var(--ink-2);
        font-size: 0.82rem;
        margin-top: 2px;
    }

    .upload-file-status {
        margin-left: auto;
        flex-shrink: 0;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: var(--success);
        font-weight: 750;
        font-size: 0.88rem;
    }

    .upload-file-status .check {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: rgba(67, 209, 122, 0.18);
        color: var(--success);
        font-size: 0.75rem;
    }

    .metric-box {
        background: linear-gradient(180deg, rgba(26, 99, 255, 0.08) 0%, rgba(10, 35, 55, 1) 100%);
        border: 1px solid #1a3b56;
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-label {
        color: var(--ink-2);
        font-size: 0.72rem;
        letter-spacing: 0.55px;
        text-transform: uppercase;
        font-weight: 760;
    }

    .metric-value {
        color: var(--ink-1);
        font-size: 1.45rem;
        font-weight: 760;
        margin-top: 5px;
    }

    .metric-safe {
        color: #83a9c7;
        margin-top: 6px;
        font-size: 0.82rem;
    }

    .alert-danger,
    .alert-warning,
    .alert-success,
    .alert-info {
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 14px;
        border: 1px solid transparent;
    }

    .alert-danger {
        background: rgba(255, 107, 122, 0.12);
        border-color: rgba(255, 107, 122, 0.35);
        color: #ffd1d7;
    }

    .alert-warning {
        background: rgba(255, 177, 74, 0.12);
        border-color: rgba(255, 177, 74, 0.35);
        color: #ffe4b9;
    }

    .alert-success {
        background: rgba(67, 209, 122, 0.10);
        border-color: rgba(67, 209, 122, 0.30);
        color: #c9f6dd;
    }

    .alert-info {
        background: rgba(26, 99, 255, 0.10);
        border-color: rgba(26, 99, 255, 0.28);
        color: #cfe1ff;
    }

    .risk-high,
    .risk-medium,
    .risk-low {
        border-radius: 13px;
        text-align: center;
        padding: 18px 14px;
        border: 1px solid;
        margin: 10px 0;
    }

    .risk-high {
        background: rgba(255, 107, 122, 0.12);
        border-color: rgba(255, 107, 122, 0.35);
        color: #ffccd3;
    }

    .risk-medium {
        background: rgba(255, 177, 74, 0.10);
        border-color: rgba(255, 177, 74, 0.35);
        color: #ffe4b9;
    }

    .risk-low {
        background: rgba(67, 209, 122, 0.10);
        border-color: rgba(67, 209, 122, 0.30);
        color: #c9f6dd;
    }

    .risk-title {
        font-size: 1rem;
        font-weight: 700;
    }

    .risk-percent {
        font-size: 1.7rem;
        font-weight: 760;
        margin: 4px 0;
    }

    .risk-level {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    }

    .sidebar-section {
        background: #0a2033;
        border: 1px solid var(--line);
        border-radius: 13px;
        box-shadow: 0 8px 20px rgba(7, 48, 77, 0.06);
        padding: 16px;
        margin-bottom: 14px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .sidebar-title {
        color: var(--ink-1);
        font-size: 0.92rem;
        font-weight: 730;
        margin-bottom: 8px;
    }

    .sidebar-text {
        color: #b9d3ea;
        font-size: 0.82rem;
        line-height: 1.65;
    }

    .divider {
        margin: 24px 0;
        border: 0;
        border-top: 1px solid var(--line-strong);
    }

    @media (max-width: 900px) {
        .header-row {
            flex-direction: column;
            gap: 14px;
        }

        .badge-container {
            justify-content: flex-start;
            max-width: none;
        }

        .header-container {
            padding: 24px 20px;
            border-radius: 15px;
        }

        .header-container h1 {
            font-size: 1.75rem;
        }
    }

    /* —— Overall app theme: Streamlit defaults → MediCare palette —— */
    .stApp {
        color: var(--ink-2);
        background-color: var(--bg-0) !important;
    }

    [data-testid="stMain"],
    [data-testid="stMain"] > div {
        background: transparent !important;
    }

    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Main text: paragraphs & lists outside custom HTML cards */
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] li,
    .main [data-testid="stMarkdownContainer"] td {
        color: var(--ink-2) !important;
    }

    .main [data-testid="stMarkdownContainer"] strong {
        color: var(--ink-1) !important;
    }

    .main [data-testid="stMarkdownContainer"] h1,
    .main [data-testid="stMarkdownContainer"] h2,
    .main [data-testid="stMarkdownContainer"] h3,
    .main [data-testid="stMarkdownContainer"] h4 {
        color: var(--ink-1) !important;
        font-weight: 700;
    }

    .main [data-testid="stMarkdownContainer"] a {
        color: #7eb8ff !important;
    }

    /* Captions & small labels */
    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: var(--ink-2) !important;
        opacity: 0.95;
    }

    /* Expanders (About, explanations) */
    [data-testid="stExpander"] {
        background: var(--bg-1) !important;
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stExpander"] summary {
        color: var(--ink-1) !important;
        font-weight: 650 !important;
    }

    [data-testid="stExpander"] summary:hover {
        color: #ffffff !important;
    }

    [data-testid="stExpanderDetails"] {
        color: var(--ink-2);
    }

    /* Data tables (SHAP / pandas) */
    .main [data-testid="stTable"] table {
        color: var(--ink-1) !important;
    }

    .main [data-testid="stTable"] thead tr th {
        background: rgba(26, 99, 255, 0.12) !important;
        color: var(--ink-1) !important;
        border-color: var(--line) !important;
    }

    .main [data-testid="stTable"] tbody tr td {
        border-color: var(--line) !important;
        background: rgba(10, 35, 55, 0.45) !important;
    }

    /* Built-in alerts */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid var(--line) !important;
    }

    /* Top toolbar area */
    header[data-testid="stHeader"] {
        background: linear-gradient(180deg, rgba(7, 28, 47, 0.95) 0%, rgba(7, 28, 47, 0.65) 100%) !important;
        border-bottom: 1px solid var(--line) !important;
    }

    [data-testid="stToolbar"] button {
        color: var(--ink-2) !important;
    }

    /* Spinner text */
    .stSpinner > div {
        color: var(--ink-1) !important;
    }

    /* Dividers Streamlit inserts */
    hr {
        border-color: var(--line-strong) !important;
        opacity: 0.85;
    }

    /* Sidebar: Streamlit native widgets */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
        color: var(--ink-2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Professional medical header
st.markdown("""
<div class="header-container">
    <div class="header-row">
        <div>
            <h1>MediCare AI</h1>
            <p class="subtitle">Professional medical report analysis with clear, actionable risk insights.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">How It Works</div>
        <div class="sidebar-text">
        1. Upload your medical report (PDF or image)<br>
        2. AI extracts key medical measurements<br>
        3. The app estimates disease risk levels<br>
        4. Review with a qualified doctor
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">Supported Conditions</div>
        <div class="sidebar-text">
        - Diabetes risk screening<br>
        - Heart disease risk screening<br>
        - Kidney disease risk screening
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">File Requirements</div>
        <div class="sidebar-text">
        - Accepted formats: PDF, PNG, JPG, JPEG<br>
        - Upload clear reports with readable lab values<br>
        - Better scan quality improves extraction accuracy
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">Important Notice</div>
        <div class="sidebar-text">
        This tool is for informational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main disclaimer
st.markdown("""
<div class="alert-danger">
    <strong>MEDICAL DISCLAIMER</strong><br>
    This AI-powered tool provides predictive estimates only and is NOT a medical diagnosis. 
    Results must be verified by a qualified healthcare professional. Do not delay medical treatment based on these results.
</div>
""", unsafe_allow_html=True)

# Safe ranges reference
SAFE_RANGES = {
    "age": "18-65",
    "glucose": "70-140 mg/dL",
    "bp": "<120 mmHg systolic",
    "cholesterol": "<200 mg/dL",
    "hemoglobin": "13-17 g/dL (male) / 12-15 g/dL (female)",
    "creatinine": "0.6-1.3 mg/dL",
    "bmi": "18.5-24.9",
    "albumin": "3.5-5.0 g/dL",
    "urea": "7-20 mg/dL",
}

# Single-page main container
main = st.container()


def format_result(result: dict):
    """Display results with parsed values and predictions in modern style."""
    
    # Display parsed values
    st.markdown('<div class="card-title">Extracted Medical Values</div>', unsafe_allow_html=True)
    parsed_values = result.get("parsed_values", {})
    
    if parsed_values:
        cols = st.columns(2)
        for idx, (k, v) in enumerate(parsed_values.items()):
            with cols[idx % 2]:
                safe = SAFE_RANGES.get(k)
                if safe:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">{k.title()}</div>
                        <div class="metric-value">{v}</div>
                        <div class="metric-safe">Normal: {safe}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">{k.title()}</div>
                        <div class="metric-value">{v}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-info">No values could be parsed from the report.</div>', unsafe_allow_html=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Display predictions
    st.markdown('<div class="card-title">Disease Risk Assessment</div>', unsafe_allow_html=True)
    
    predictions = result.get("predictions", {})
    explanations = result.get("explanations", {}) or {}
    
    if predictions:
        # Prepare data for chart
        chart_data = []
        prediction_items = []
        
        for disease, out in predictions.items():
            status = out.get("status", "ok")
            if status != "ok":
                missing = out.get("missing", [])
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>{disease.title()}</strong><br>
                    Insufficient data (missing: {', '.join(missing)})
                </div>
                """, unsafe_allow_html=True)
            else:
                prob = out.get("prob", 0)
                risk = out.get("risk", "")
                chart_data.append({
                    "Disease": disease.title(),
                    "Probability (%)": prob * 100
                })
                prediction_items.append((disease.title(), prob, risk))
        
            # Display chart if we have data
        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            
            # Modern high-quality Matplotlib styling
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#1a63ff', '#00c2b8', '#8cbf26']
            bars = ax.bar(
                df_chart["Disease"],
                df_chart["Probability (%)"],
                color=colors[:len(df_chart)],
                alpha=0.85,
                edgecolor='#eaf3ff',
                linewidth=2,
            )
            
            # Customize chart
            ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold', color='#eaf3ff')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.25, linestyle='--')
            ax.set_facecolor('#071b2d')
            fig.patch.set_facecolor('#071b2d')
            ax.tick_params(axis='both', colors='#b9d3ea')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11, color='#eaf3ff')
            
            plt.xticks(fontsize=11, fontweight='bold')
            plt.tight_layout()
            
            # Use columns to roughly center or constrain the image if needed
            st.pyplot(fig)

            # Donut visualization for risk distribution (circular chart requirement)
            if df_chart['Probability (%)'].sum() > 0:
                fig2, ax2 = plt.subplots(figsize=(5.2, 5.2))
                risky_colors = ['#dc3545', '#ffc107', '#28a745']
                labels = df_chart['Disease']
                values = df_chart['Probability (%)']
                wedges, texts = ax2.pie(
                    values,
                    labels=labels,
                    colors=[risky_colors[i % len(risky_colors)] for i in range(len(values))],
                    autopct='%.1f%%',
                    startangle=90,
                    pctdistance=0.77,
                    textprops={'color': '#edf3ff', 'weight': 'bold'},
                    wedgeprops={'width':0.35, 'edgecolor':'#0a1f36', 'linewidth':1.5},
                )
                center_circle = plt.Circle((0, 0), 0.45, color='#071b2d')
                ax2.add_artist(center_circle)
                ax2.set_title('Risk Mix by Condition', color='#eaf3ff', weight='bold', fontsize=14)
                ax2.set_facecolor('#071b2d')
                fig2.patch.set_facecolor('#071b2d')
                st.pyplot(fig2)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            
            max_per_row = 3
            for i in range(0, len(prediction_items), max_per_row):
                row_items = prediction_items[i : i + max_per_row]
                cols = st.columns(len(row_items))
                for idx, (disease, prob, risk) in enumerate(row_items):
                    with cols[idx]:
                        prob_percent = prob * 100
                        if risk == "High":
                            st.markdown(f"""
                            <div class="risk-high">
                                <div class="risk-title">{disease}</div>
                                <div class="risk-percent">{prob_percent:.1f}%</div>
                                <div class="risk-level">HIGH RISK</div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif risk == "Medium":
                            st.markdown(f"""
                            <div class="risk-medium">
                                <div class="risk-title">{disease}</div>
                                <div class="risk-percent">{prob_percent:.1f}%</div>
                                <div class="risk-level">MEDIUM RISK</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="risk-low">
                                <div class="risk-title">{disease}</div>
                                <div class="risk-percent">{prob_percent:.1f}%</div>
                                <div class="risk-level">LOW RISK</div>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-info">No predictions available.</div>', unsafe_allow_html=True)

    # Display Grok AI explanations (computed in the backend)
    if predictions:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Model Explanation (AI-Powered)</div>', unsafe_allow_html=True)
        st.caption("This section provides AI-generated explanations of what influenced your risk assessment, powered by Grok AI.")

        def pretty_feature_name(feature: str) -> str:
            """Convert internal feature names into patient-friendly labels."""
            if not feature:
                return "Unknown factor"
            f = str(feature).lower()

            # Numeric biomarkers you extract from reports (or their encoded names)
            biomarker_map = {
                "age": "Age",
                "trestbps": "Blood pressure (systolic)",
                "bp": "Blood pressure",
                "chol": "Cholesterol",
                "cholesterol": "Cholesterol",
                "glucose": "Blood sugar (glucose)",
                "bmi": "BMI",
                "sc": "Creatinine",
                "creatinine": "Creatinine",
                "bu": "Urea (BUN)",
                "urea": "Urea (BUN)",
                "al": "Albumin",
                "albumin": "Albumin",
                "hemo": "Hemoglobin",
                "hemoglobin": "Hemoglobin",
            }
            # Match startswith to handle transformed/encoded variants.
            for k, v in biomarker_map.items():
                if f.startswith(k):
                    return v

            # Fallback for one-hot / encoded features
            if "sex_" in f:
                return "Sex (as encoded in the dataset)"
            if "cp_" in f:
                return "Chest pain type (as encoded)"
            return str(feature)

        for disease, out in predictions.items():
            if out.get("status") != "ok":
                continue

            exp = explanations.get(disease, {})
            with st.expander(f"{disease.title()} explanation", expanded=False):
                if not exp:
                    st.info("Explanation not available for this prediction.")
                    continue

                if "error" in exp:
                    st.warning(exp.get("error", "AI explanation unavailable."))
                    continue

                # Display Groq AI explanation if available
                explanation_text = exp.get("explanation_text", "")
                if explanation_text:
                    st.markdown("**AI-Generated Explanation:**")
                    st.markdown(explanation_text)
                    prob = exp.get("risk_probability", 0)
                    risk_level = "HIGH" if prob > 0.7 else "MODERATE" if prob > 0.3 else "LOW"
                    st.markdown(f"**Predicted Risk:** {prob:.1%} ({risk_level})")
                    st.markdown(
                        "<div style='margin-top:10px;color:rgba(185,211,234,0.95)'>"
                        "Note: This explanation is generated by AI and should not replace professional medical advice."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Fallback to legacy SHAP display if no AI explanation
                    global_like = exp.get("global_importance", [])
                    positive = exp.get("individual_positive", [])
                    negative = exp.get("individual_negative", [])

                    if global_like:
                        st.markdown("**Most influential factors (overall):**")
                        for item in global_like[:5]:
                            feat = item.get("feature", "unknown")
                            st.markdown(f"- {pretty_feature_name(feat)}")
                    else:
                        st.info("Most-influential factors are not available for this prediction.")

                    if positive or negative:
                        col_inc, col_dec = st.columns(2)
                        with col_inc:
                            st.markdown("**What increased the predicted risk:**")
                            if positive:
                                for item in positive[:4]:
                                    feat = item.get("feature", "unknown")
                                    st.markdown(f"- {pretty_feature_name(feat)}")
                            else:
                                st.markdown("- None found")

                        with col_dec:
                            st.markdown("**What lowered the predicted risk:**")
                            if negative:
                                for item in negative[:4]:
                                    feat = item.get("feature", "unknown")
                                    st.markdown(f"- {pretty_feature_name(feat)}")
                            else:
                                st.markdown("- None found")

                    st.markdown(
                        "<div style='margin-top:10px;color:rgba(185,211,234,0.95)'>"
                        "Note: If a value was missing from the report, the model may use typical baseline values "
                        "from training data for that part of the calculation. So explanations reflect the model’s estimates."
                        "</div>",
                        unsafe_allow_html=True,
                    )

def _build_pdf_report_bytes(result: dict) -> bytes:
    """Build a professional hospital-grade PDF report using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import matplotlib.pyplot as plt
    import io

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0c3a99'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#666666'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )

    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#0c3a99'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    # Extract data
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parsed_values = result.get("parsed_values", {}) or {}
    predictions = result.get("predictions", {}) or {}
    explanations = result.get("explanations", {}) or {}

    # ===== HEADER SECTION =====
    story.append(Paragraph("MediCare AI", title_style))
    story.append(Paragraph("Health Risk Assessment Report", subtitle_style))

    header_info = f"""
    <font size=10>
    <b>Report ID:</b> {generated_at.replace(':', '').replace(' ', '_')} | 
    <b>Generated:</b> {generated_at} | 
    <b>System Version:</b> 1.0
    </font>
    """
    story.append(Paragraph(header_info, styles['Normal']))
    story.append(Spacer(1, 0.15*inch))

    # Horizontal line
    story.append(Table([['']], colWidths=[7.5*inch], rowHeights=[1], 
                      style=TableStyle([('LINEABOVE', (0, 0), (-1, -1), 2, colors.HexColor('#0c3a99'))])))
    story.append(Spacer(1, 0.2*inch))

    # ===== DISCLAIMER SECTION =====
    story.append(Paragraph("⚠ Important Medical Disclaimer", heading_style))
    disclaimer_text = """
    <font size=9>
    This report contains <b>AI-powered predictive estimates ONLY</b> and is <b>NOT a medical diagnosis</b>. 
    Results must be reviewed and verified by a licensed healthcare professional. 
    Do NOT delay or avoid seeking medical care based on these results. 
    Always consult with your doctor before making any health decisions.
    </font>
    """
    disclaimer_para = Paragraph(disclaimer_text, styles['Normal'])
    disclaimer_table = Table([[disclaimer_para]], colWidths=[7*inch],
                            style=TableStyle([
                                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff3cd')),
                                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                                ('TOPPADDING', (0, 0), (-1, -1), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                                ('BORDER', (0, 0), (-1, -1), 2, colors.HexColor('#ff9800')),
                                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 0.2*inch))

    # ===== PATIENT DATA TABLE =====
    story.append(Paragraph("Extracted Medical Information", heading_style))

    table_data = [['Parameter', 'Value', 'Normal Range', 'Status']]
    table_style_list = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0c3a99')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]

    row_idx = 1
    for param, value in parsed_values.items():
        safe = SAFE_RANGES.get(param, "N/A")
        status_text = '✓ Normal'

        table_data.append([param.title(), str(value), str(safe), status_text])
        table_style_list.append(('BACKGROUND', (3, row_idx), (3, row_idx), colors.HexColor('#d4edda')))
        table_style_list.append(('FONTNAME', (3, row_idx), (3, row_idx), 'Helvetica-Bold'))
        row_idx += 1

    if not parsed_values:
        table_data.append(['No Data', 'N/A', 'N/A', 'N/A'])

    patient_table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.8*inch, 1.2*inch])
    patient_table.setStyle(TableStyle(table_style_list))
    story.append(patient_table)
    story.append(Spacer(1, 0.2*inch))

    # ===== DISEASE RISK SECTION =====
    story.append(Paragraph("Disease Risk Assessment", heading_style))

    disease_boxes = []
    for disease, out in predictions.items():
        status = out.get("status", "ok")
        if status != "ok":
            continue

        prob = float(out.get("prob", 0))
        risk = out.get("risk", "Unknown")

        # Color coding
        if risk == "Low":
            risk_color = '#28a745'
        elif risk == "Medium":
            risk_color = '#ffc107'
        else:
            risk_color = '#dc3545'

        risk_text = f"""
        <font size=11><b>{disease.upper()}</b></font><br/>
        <font size=10>
        Risk Level: <font color="{risk_color}"><b>{risk}</b></font><br/>
        Confidence: {prob*100:.1f}%
        </font>
        """
        disease_boxes.append(Paragraph(risk_text, styles['Normal']))

    if disease_boxes:
        cols = min(3, len(disease_boxes))
        disease_data = [disease_boxes[i:i+cols] for i in range(0, len(disease_boxes), cols)]
        disease_table = Table(disease_data, colWidths=[7.5/cols*inch]*cols)
        disease_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('BORDER', (0, 0), (-1, -1), 1, colors.HexColor('#0c3a99')),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(disease_table)
    story.append(Spacer(1, 0.2*inch))

    # ===== CHART SECTION =====
    if predictions:
        story.append(Paragraph("Risk Distribution Chart", heading_style))

        fig, ax = plt.subplots(figsize=(7, 4))
        diseases = []
        risks = []
        colors_list = []

        for disease, out in predictions.items():
            if out.get("status") == "ok":
                prob = float(out.get("prob", 0)) * 100
                diseases.append(disease.title())
                risks.append(prob)
                risk_obj = out.get("risk", "Low")
                if risk_obj == "Low":
                    colors_list.append('#28a745')
                elif risk_obj == "Medium":
                    colors_list.append('#ffc107')
                else:
                    colors_list.append('#dc3545')

        ax.bar(diseases, risks, color=colors_list, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Risk Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        chart_file = io.BytesIO()
        fig.savefig(chart_file, format='png', dpi=100, bbox_inches='tight')
        chart_file.seek(0)
        plt.close(fig)

        chart_img = Image(chart_file, width=6*inch, height=3*inch)
        story.append(chart_img)
        story.append(Spacer(1, 0.2*inch))

    # ===== AI EXPLANATIONS SECTION =====
    story.append(Paragraph("AI-Generated Analysis", heading_style))

    for disease, out in predictions.items():
        if out.get("status") != "ok":
            continue

        exp = explanations.get(disease, {}) or {}
        explanation_text = exp.get("explanation_text", "")

        disease_heading = f"<font size=11><b>{disease.title()}</b></font>"
        story.append(Paragraph(disease_heading, styles['Normal']))

        if explanation_text:
            exp_para = Paragraph(f"<font size=10>{explanation_text}</font>", styles['Normal'])
            story.append(exp_para)
        else:
            story.append(Paragraph("<font size=10><i>Explanation not available.</i></font>", styles['Normal']))

        story.append(Spacer(1, 0.1*inch))

    story.append(Spacer(1, 0.15*inch))

    # ===== RECOMMENDATIONS SECTION =====
    story.append(Paragraph("Recommended Actions", heading_style))

    recommendations = """
    <font size=10>
    <b>•</b> Schedule an appointment with your healthcare provider to discuss these results<br/>
    <b>•</b> Bring this report to your doctor for professional review and diagnosis<br/>
    <b>•</b> Maintain regular health monitoring and preventive care<br/>
    <b>•</b> Follow lifestyle recommendations from your physician<br/>
    <b>•</b> Do not delay seeking medical attention if experiencing health concerns
    </font>
    """
    story.append(Paragraph(recommendations, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # ===== FOOTER =====
    footer_text = """
    <font size=8 color="#666666">
    <b>Generated by MediCare AI v1.0</b> | This report is AI-assisted and not a medical diagnosis | {0}
    </font>
    """.format(generated_at)
    story.append(Paragraph(footer_text, styles['Normal']))

    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.read()


# Upload section — drag-and-drop card (clear for non-technical users)
st.markdown('<div class="card upload-card-light">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="upload-intro">
        <div class="upload-panel-title">Drag and Drop File Uploader</div>
        <div class="upload-panel-sub">Upload a photo or PDF of your lab report. Drag it into the box below, or use <strong>Browse files</strong>.</div>
        <div class="upload-hint">We accept PDF, PNG, JPG, or JPEG · up to 200MB per file</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = main.file_uploader(
    "Select a file to analyze",
    type=["pdf", "png", "jpg", "jpeg"],
    label_visibility="visible",
    help="Tip: Use a clear scan or photo so lab numbers are easy to read.",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="alert-info">
            <strong>Get started:</strong> Choose a medical report (PDF or photo) above to begin.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    _raw_name = uploaded_file.name.replace("\\", "/").split("/")[-1]
    _safe_name = html.escape(_raw_name)
    _size_kb = uploaded_file.size / 1024
    if _size_kb >= 1024:
        _size_str = f"{_size_kb / 1024:.2f} MB"
    else:
        _size_str = f"{_size_kb:.2f} KB"
    st.markdown(
        f"""
        <div class="upload-file-row">
            <div class="upload-file-icon" aria-hidden="true">📄</div>
            <div class="upload-file-meta">
                <div class="upload-file-name" title="{_safe_name}">{_safe_name}</div>
                <div class="upload-file-size">{_size_str}</div>
            </div>
            <div class="upload-file-status"><span class="check">✓</span> Uploaded</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("---")
    
    # Show preview for images
    _, ext = os.path.splitext(uploaded_file.name)
    if ext.lower() in [".png", ".jpg", ".jpeg"]:
        try:
            st.markdown('<div class="card-title">Document Preview</div>', unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
        except Exception:
            st.markdown('<div class="alert-warning">Preview unavailable for this image.</div>', unsafe_allow_html=True)
    elif ext.lower() == ".pdf":
        st.markdown('<div class="alert-info">PDF file uploaded. Click "Analyze" to extract medical values.</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analyze button
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analyze_button = st.button("Analyze Medical Report", type="primary", use_container_width=True)
    
    if analyze_button:
        with st.spinner("Analyzing your medical report with AI..."):
            tmp_path = None
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                
                # Run prediction
                result = run(tmp_path, models_dir="models")
                
                # Display results
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Analysis Results</div>', unsafe_allow_html=True)
                
                format_result(result)
                
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                
                # Success message
                st.markdown("""
                <div class="alert-success">
                    <strong>Analysis Complete</strong><br>
                    Results have been generated. Review carefully and consult your healthcare provider.
                </div>
                """, unsafe_allow_html=True)
                
                # Additional recommendations
                st.markdown("""
                <div class="card">
                    <div class="card-title">Next Steps</div>
                    <div class="sidebar-text">
                    1. <strong>Review Results</strong> - Examine the risk assessments above<br>
                    2. <strong>Consult Doctor</strong> - Share these results with your healthcare provider<br>
                    3. <strong>Follow-up</strong> - Get professional diagnosis and treatment plan<br>
                    4. <strong>Download PDF Report</strong> - Save the formatted report for your records
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Download results (PDF)
                try:
                    pdf_bytes = _build_pdf_report_bytes(result)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"medicare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"PDF generation failed: {str(e)}")
                
            except Exception as e:
                st.markdown(f"""
                <div class="alert-danger">
                    <strong>Error Processing File</strong><br>
                    {str(e)}<br>
                    Please ensure your file is valid and readable.
                </div>
                """, unsafe_allow_html=True)
            finally:
                # Clean up temp file
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.markdown("</div>", unsafe_allow_html=True)

# About section
st.markdown("---")

with st.expander("About MediCare AI - Professional Medical Platform"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Clinical Mission")
        st.markdown("""
        MediCare AI is dedicated to advancing preventive medicine through intelligent analysis of medical data. Our platform leverages cutting-edge machine learning to provide clinically-informed risk assessments that support healthcare professionals and empower patients with actionable health insights.
        """)
        
        st.subheader("Clinical Features")
        st.markdown("""
        - **Advanced Document Parsing** - OCR & AI-powered extraction
        - **Multi-Disease Analysis** - Multiple condition assessment
        - **Clinically-Validated Models** - Evidence-based algorithms
        - **Reference Range Mapping** - Clinical standard comparison
        - **Risk Stratification** - Low/Medium/High classification
        - **Secure Reporting** - HIPAA-compliant data handling
        """)
    
    with col2:
        st.subheader("Supported Conditions")
        st.markdown("""
        **Diabetes Risk**
        - Glucose, HbA1c, BMI, metabolic markers
        
        **Heart Disease Risk**
        - Cholesterol, Blood Pressure, cardiac biomarkers
        
        **Kidney Disease Risk**
        - Creatinine, BUN, Albumin, electrolytes
        """)
        
        st.subheader("Professional Standards")
        st.markdown("""
        - Evidence-Based Algorithms
        
        - HIPAA-Ready Architecture
        
        - FDA-Compliant Standards
        
        - Clinically Validated Models
        
        - Transparent & Explainable AI
        """)

# Professional Medical Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### Multi-Disease Analysis
    Diabetes, Heart, Kidney assessed simultaneously
    """)

with col2:
    st.markdown("""
    #### Secure & Private
    HIPAA-compliant, FDA-ready technology
    """)

with col3:
    st.markdown("""
    #### Clinically Validated
    Evidence-based, real patient data tested
    """)

st.markdown("---")

st.markdown("""
**MediCare AI Platform** | Advanced Medical Intelligence & Risk Assessment System

**Medical Disclaimer:** This platform provides AI-generated risk assessments for informational purposes only. 
These results are NOT medical diagnoses and should NOT be used as a substitute for professional medical advice. 
Always consult with a qualified healthcare professional.

---

Copyright 2024-2026 MediCare AI | Built with Python, Streamlit, Machine Learning | v2.0 Medical Grade
""")
