import os
import tempfile
import html
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predict_from_report import predict_from_parsed, run

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"

# Make sure Streamlit secret-managed API keys work in deployed env
# Avoid StreamlitSecretNotFoundError when no secrets.toml is present
def _get_secret(key):
    try:
        return st.secrets.get(key) if hasattr(st, "secrets") and hasattr(st.secrets, "get") else None
    except Exception:
        return None

if "GROK_API_KEY" not in os.environ:
    grok_secret = _get_secret("GROK_API_KEY")
    if grok_secret:
        os.environ["GROK_API_KEY"] = grok_secret

if "GROQ_API_KEY" not in os.environ:
    groq_secret = _get_secret("GROQ_API_KEY")
    if groq_secret:
        os.environ["GROQ_API_KEY"] = groq_secret

# Page configuration
st.set_page_config(
    page_title="MediCare AI - Professional Health Analysis",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional medical UI styling
def load_css(css_path: Path):
    """Load app-level CSS from the assets directory."""
    if not css_path.exists():
        st.error(f"Missing stylesheet: {css_path}")
        return

    css = css_path.read_text(encoding="utf-8-sig")
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)


load_css(ASSETS_DIR / "styles.css")

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

    with st.expander("Connect Grok/Groq API key", expanded=False):
        grok_k = st.text_input("GROK_API_KEY", type="password", value=st.session_state.get("grok_api_key", ""))
        groq_k = st.text_input("GROQ_API_KEY", type="password", value=st.session_state.get("groq_api_key", ""))
        if st.button("Save API keys"):
            if grok_k:
                st.session_state["grok_api_key"] = grok_k
                os.environ["GROK_API_KEY"] = grok_k
                st.success("GROK_API_KEY stored in session")
            if groq_k:
                st.session_state["groq_api_key"] = groq_k
                os.environ["GROQ_API_KEY"] = groq_k
                st.success("GROQ_API_KEY stored in session")

# Main disclaimer
st.markdown("""
<div class="alert-danger">
    <strong>MEDICAL DISCLAIMER</strong><br>
    This AI-powered tool provides predictive estimates only and is NOT a medical diagnosis. 
    Results must be verified by a qualified healthcare professional. Do not delay medical treatment based on these results.
</div>
""", unsafe_allow_html=True)

MODEL_DIR = "models"

# Clinical reference ranges shown beside extracted or manually entered values.
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
                wedges, texts, autotexts = ax2.pie(
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
                        "from training data for that part of the calculation. So explanations reflect the modelâ€™s estimates."
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
    story.append(Paragraph("âš  Important Medical Disclaimer", heading_style))
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
        status_text = 'âœ“ Normal'

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
    <b>â€¢</b> Schedule an appointment with your healthcare provider to discuss these results<br/>
    <b>â€¢</b> Bring this report to your doctor for professional review and diagnosis<br/>
    <b>â€¢</b> Maintain regular health monitoring and preventive care<br/>
    <b>â€¢</b> Follow lifestyle recommendations from your physician<br/>
    <b>â€¢</b> Do not delay seeking medical attention if experiencing health concerns
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


MANUAL_VALUE_FIELDS = [
    ("age", "Age", "years", "45"),
    ("glucose", "Glucose", "mg/dL", "110"),
    ("bp", "Blood Pressure (Systolic)", "mmHg", "120"),
    ("cholesterol", "Cholesterol", "mg/dL", "180"),
    ("bmi", "BMI", "kg/m2", "24.5"),
    ("creatinine", "Creatinine", "mg/dL", "1.0"),
    ("urea", "Urea", "mg/dL", "15"),
    ("albumin", "Albumin", "g/dL", "4.0"),
    ("hemoglobin", "Hemoglobin", "g/dL", "13.5"),
]

MANUAL_FIELD_LOOKUP = {field[0]: field for field in MANUAL_VALUE_FIELDS}

MANUAL_PROBLEM_OPTIONS = [
    {
        "key": "diabetes",
        "title": "Diabetes",
        "summary": "Glucose, BMI, and age",
        "fields": ["glucose", "bmi", "age"],
    },
    {
        "key": "heart",
        "title": "Heart Disease",
        "summary": "Age, blood pressure, and cholesterol",
        "fields": ["age", "bp", "cholesterol"],
    },
    {
        "key": "ckd",
        "title": "Kidney Disease",
        "summary": "Creatinine, urea, albumin, and hemoglobin",
        "fields": ["creatinine", "urea", "albumin", "hemoglobin"],
    },
]

MANUAL_PROBLEM_LOOKUP = {problem["key"]: problem for problem in MANUAL_PROBLEM_OPTIONS}


def _sync_session_api_keys_to_env():
    """Keep saved sidebar API keys available for prediction explanations."""
    if st.session_state.get("grok_api_key"):
        os.environ["GROK_API_KEY"] = st.session_state["grok_api_key"]

    if st.session_state.get("groq_api_key"):
        os.environ["GROQ_API_KEY"] = st.session_state["groq_api_key"]


def _parse_manual_values(raw_values: dict) -> tuple[dict, list[str]]:
    """Convert form strings into numeric model inputs and collect validation errors."""
    parsed = {}
    errors = []

    for key, raw in raw_values.items():
        cleaned = str(raw or "").strip()
        if not cleaned:
            continue
        try:
            parsed[key] = float(cleaned)
        except ValueError:
            label = MANUAL_FIELD_LOOKUP.get(key, (key, key))[1]
            errors.append(f"{label} must be a number.")

    return parsed, errors


def _show_completed_analysis(result: dict, download_key: str):
    """Render shared results, next steps, and PDF download for both input paths."""
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Analysis Results</div>', unsafe_allow_html=True)

    format_result(result)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="alert-success">
        <strong>Analysis Complete</strong><br>
        Results have been generated. Review carefully and consult your healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card-title">Next Steps</div>
    <div class="sidebar-text">
    1. <strong>Review Results</strong> - Examine the risk assessments above<br>
    2. <strong>Consult Doctor</strong> - Share these results with your healthcare provider<br>
    3. <strong>Follow-up</strong> - Get professional diagnosis and treatment plan<br>
    4. <strong>Download PDF Report</strong> - Save the formatted report for your records
    </div>
    """, unsafe_allow_html=True)

    try:
        pdf_bytes = _build_pdf_report_bytes(result)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"medicare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=download_key,
        )
    except Exception as e:
        st.warning(f"PDF generation failed: {str(e)}")


# Manual value workflow
st.markdown('<div class="card-title">Enter Medical Values Manually</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="alert-info">
        Use this option when you already know your lab values or do not have a report file to upload.
    </div>
    """,
    unsafe_allow_html=True,
)

if "manual_problem_key" not in st.session_state:
    st.session_state["manual_problem_key"] = None

problem_cols = st.columns(3)
for idx, problem in enumerate(MANUAL_PROBLEM_OPTIONS):
    is_selected = st.session_state["manual_problem_key"] == problem["key"]
    selected_label = "Open" if is_selected else "Select"
    with problem_cols[idx]:
        st.markdown(
            f"""
            <div class="metric-box" style="min-height:118px;border-left-color:{'#43d17a' if is_selected else '#1a63ff'};">
                <div class="metric-label">{html.escape(selected_label)}</div>
                <div class="metric-value" style="font-size:1.15rem;">{html.escape(problem["title"])}</div>
                <div class="metric-safe">{html.escape(problem["summary"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(problem["title"], key=f"select_manual_{problem['key']}", use_container_width=True):
            if is_selected:
                st.session_state["manual_problem_key"] = None
            else:
                st.session_state["manual_problem_key"] = problem["key"]
            st.rerun()

selected_problem = None
manual_inputs = {}
manual_analyze_button = False

if st.session_state["manual_problem_key"]:
    selected_problem = MANUAL_PROBLEM_LOOKUP[st.session_state["manual_problem_key"]]
    st.markdown(
        f'<div class="card-title">{html.escape(selected_problem["title"])} Manual Input</div>',
        unsafe_allow_html=True,
    )

    with st.form(f"manual_values_form_{selected_problem['key']}"):
        selected_fields = selected_problem["fields"]
        for row_start in range(0, len(selected_fields), 3):
            current_fields = selected_fields[row_start:row_start + 3]
            cols = st.columns(min(3, len(current_fields)))
            for idx, field_key in enumerate(current_fields):
                key, label, unit, placeholder = MANUAL_FIELD_LOOKUP[field_key]
                with cols[idx]:
                    manual_inputs[key] = st.text_input(
                        f"{label} ({unit})",
                        value="",
                        placeholder=placeholder,
                        key=f"manual_{selected_problem['key']}_{key}",
                    )

        manual_analyze_button = st.form_submit_button(
            f"Analyze {selected_problem['title']}",
            type="primary",
            use_container_width=True,
        )

if manual_analyze_button:
    manual_values, manual_errors = _parse_manual_values(manual_inputs)
    missing_manual_fields = [
        field_key for field_key in selected_problem["fields"] if field_key not in manual_values
    ]
    if manual_errors:
        for error in manual_errors:
            st.error(error)
    elif missing_manual_fields:
        missing_labels = [
            MANUAL_FIELD_LOOKUP[field_key][1] for field_key in missing_manual_fields
        ]
        st.warning(f"Please enter: {', '.join(missing_labels)}.")
    else:
        with st.spinner("Analyzing manually entered medical values..."):
            try:
                _sync_session_api_keys_to_env()
                prediction_payload = predict_from_parsed(manual_values, models_dir=MODEL_DIR)
                prediction_key = selected_problem["key"]
                manual_result = {
                    "parsed_values": manual_values,
                    "predictions": {
                        prediction_key: prediction_payload["predictions"].get(prediction_key, {})
                    },
                    "explanations": {
                        prediction_key: prediction_payload["explanations"].get(prediction_key, {})
                    },
                }
                _show_completed_analysis(manual_result, download_key="manual_pdf_download")
            except Exception as e:
                st.markdown(f"""
                <div class="alert-danger">
                    <strong>Error Processing Manual Values</strong><br>
                    {str(e)}
                </div>
                """, unsafe_allow_html=True)


# Report upload workflow
st.markdown('<div class="card upload-card-light">', unsafe_allow_html=True)

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
    raw_file_name = uploaded_file.name.replace("\\", "/").split("/")[-1]
    safe_file_name = html.escape(raw_file_name)
    file_size_kb = uploaded_file.size / 1024
    if file_size_kb >= 1024:
        file_size_text = f"{file_size_kb / 1024:.2f} MB"
    else:
        file_size_text = f"{file_size_kb:.2f} KB"
    st.markdown(
        f"""
        <div class="upload-file-row">
            <div class="upload-file-icon" aria-hidden="true">File</div>
            <div class="upload-file-meta">
                <div class="upload-file-name" title="{safe_file_name}">{safe_file_name}</div>
                <div class="upload-file-size">{file_size_text}</div>
            </div>
            <div class="upload-file-status"><span class="check">OK</span> Uploaded</div>
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
    
    analyze_col, _ = st.columns([2, 1])
    
    with analyze_col:
        analyze_button = st.button("Analyze Medical Report", type="primary", use_container_width=True)
    
    if analyze_button:
        with st.spinner("Analyzing your medical report with AI..."):
            tmp_path = None
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                
                _sync_session_api_keys_to_env()

                # Run prediction
                result = run(tmp_path, models_dir=MODEL_DIR)
                _show_completed_analysis(result, download_key="upload_pdf_download")
                
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
