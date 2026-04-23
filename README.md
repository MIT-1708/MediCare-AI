# MediCare AI

MediCare AI is a Streamlit-based health risk screening application for diabetes, heart disease, and chronic kidney disease. It can analyze uploaded medical reports or manually entered lab values, then shows model-based risk estimates, AI explanations, charts, and a downloadable PDF report.

> This project is for educational and informational risk screening only. It is not a medical diagnosis and must not replace advice from a qualified healthcare professional.

## Features

- Manual value entry for specific health problems:
  - Diabetes: glucose, BMI, age
  - Heart disease: age, systolic blood pressure, cholesterol
  - Kidney disease: creatinine, urea, albumin, hemoglobin
- Report upload analysis for PDF, PNG, JPG, and JPEG files
- PDF text extraction with `pdfplumber`
- Image OCR with `pytesseract`
- Multi-disease ML prediction pipeline
- Risk bands: Low, Medium, High
- AI-generated explanations using Grok/Groq-compatible OpenAI client settings
- Downloadable PDF summary report
- Streamlit UI with hospital-style dark theme

## Supported Conditions

| Condition | Main values used |
| --- | --- |
| Diabetes | Glucose, BMI, Age, Blood Pressure when available |
| Heart Disease | Age, Systolic Blood Pressure, Cholesterol |
| Chronic Kidney Disease | Creatinine, Urea, Albumin, Hemoglobin, Age/BP when available |

## Project Structure

```text
Care/
|-- assets/
|   |-- styles.css                 # Streamlit UI styling
|-- data/                         # Training datasets
|-- eda_outputs/                  # EDA charts and summary JSON files
|-- models/                       # Saved model artifacts and metrics
|-- exapmles report/              # Example medical reports
|-- keras_model.py                # Lightweight Keras-compatible inference wrapper
|-- predict_from_report.py        # OCR/PDF parsing, value extraction, prediction logic
|-- streamlit_app.py              # Main Streamlit application
|-- train_models.py               # Model training pipeline
|-- requirements.txt              # Runtime dependencies
|-- packages.txt                  # Streamlit Cloud system packages
|-- start_project.bat             # Windows launcher
|-- README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run The App

Recommended command:

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Or use the included Windows launcher:

```powershell
.\start_project.bat
```

Open the app:

```text
http://localhost:8501
```

## How To Use

### Manual Value Analysis

1. Open the app.
2. Go to `Enter Medical Values Manually`.
3. Click one health problem: `Diabetes`, `Heart Disease`, or `Kidney Disease`.
4. Enter the requested values.
5. Click the analyze button for that condition.

Clicking the same health-problem button again hides its manual input form.

### Uploaded Report Analysis

1. Upload a medical report as PDF, PNG, JPG, or JPEG.
2. Preview the document when available.
3. Click `Analyze Medical Report`.
4. Review extracted values, risk results, explanations, and the PDF download.

## Optional API Keys

AI explanations use a Grok/Groq-compatible OpenAI client. You can add keys in the app sidebar or set environment variables:

```powershell
$env:GROK_API_KEY = "your_key_here"
$env:GROQ_API_KEY = "your_key_here"
```

Optional overrides:

```powershell
$env:API_BASE_URL = "https://api.groq.com/openai/v1"
$env:GROK_MODEL = "llama-3.3-70b-versatile"
```

If no API key is provided, predictions can still run, but AI explanations may show as unavailable.

## OCR Setup For Image Reports

PDF reports work through `pdfplumber`. Image reports need Tesseract OCR.

On Windows, install Tesseract from:

```text
https://github.com/UB-Mannheim/tesseract/wiki
```

Default path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If needed, set the path manually:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Command Line Prediction

Run prediction directly from a report file:

```powershell
.venv\Scripts\python.exe predict_from_report.py --file path\to\report.pdf
```

Save JSON output:

```powershell
.venv\Scripts\python.exe predict_from_report.py --file path\to\report.jpg --out result.json
```

## Training Models

Run the training pipeline:

```powershell
.venv\Scripts\python.exe train_models.py
```

Optional GPU modes:

```powershell
.venv\Scripts\python.exe train_models.py --gpu auto
.venv\Scripts\python.exe train_models.py --gpu always
.venv\Scripts\python.exe train_models.py --gpu never
```

Training outputs include:

- `models/heart_model.pkl`
- `models/diabetes_model.pkl`
- `models/kidney_model.pkl`
- `models/scalers.pkl`
- `models/dataset_metadata.json`
- disease metrics JSON files
- EDA charts in `eda_outputs/`

## Deployment Notes

For Streamlit Cloud:

- Install Python dependencies from `requirements.txt`.
- Use `packages.txt` to install `tesseract-ocr` for image OCR.
- Prefer Python 3.11 if the deployment platform allows choosing a version.
- Add `GROQ_API_KEY` or `GROK_API_KEY` in Streamlit secrets if AI explanations are needed.

## Important Limitations

- Model output is only a risk estimate, not a diagnosis.
- OCR quality depends heavily on report clarity.
- Manually entered values must use the units shown in the form.
- Missing values can prevent a condition-specific prediction.
- AI explanations are generated text and should be reviewed critically.

## Medical Disclaimer

MediCare AI provides AI-assisted screening information only. It does not diagnose, treat, or prevent disease. Always consult a qualified healthcare provider before making health decisions.
