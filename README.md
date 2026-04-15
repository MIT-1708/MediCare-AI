# MediCare AI

MediCare AI is a compact healthcare risk prediction app that extracts clinical values from structured datasets and uploaded reports to predict heart disease, diabetes, and chronic kidney disease.

## What it does
- extracts values from PDF/image reports using OCR and PDF parsing
- predicts disease risk using classical ML and a Keras neural network
- displays results in a Streamlit interface
- supports explainable output for model transparency

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.
3. Run:
   ```bash
   streamlit run streamlit_app.py
   ```

## Diseases covered
- Heart Disease
- Diabetes
- Chronic Kidney Disease

## Models used
- Heart disease: Keras neural network
- Diabetes: voting ensemble classifier
- CKD: XGBoost

## Key files
- `streamlit_app.py` — user interface
- `predict_from_report.py` — report parsing and feature extraction
- `train_models.py` — training and model selection
- `keras_model.py` — heart disease neural network

## Project structure
```text
Care/
|-- data/
|-- eda_outputs/
|-- models/
|-- .venv/
|-- keras_model.py
|-- packages.txt
|-- predict_from_report.py
|-- streamlit_app.py
```

```

For local model training, install the training extras too:

```powershell
pip install -r requirements-train.txt
```

### 3. Install Tesseract OCR on Windows

Download:
- `https://github.com/UB-Mannheim/tesseract/wiki`

Default install path:
- `C:\Program Files\Tesseract-OCR\tesseract.exe`

If needed, set it manually:

```powershell
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Training

Run model training:

```powershell
.venv\Scripts\python.exe train_models.py
```

Optional GPU flags:

```powershell
.venv\Scripts\python.exe train_models.py --gpu auto
.venv\Scripts\python.exe train_models.py --gpu always
.venv\Scripts\python.exe train_models.py --gpu never
```

Training outputs:
- `models/heart_model.pkl`
- `models/diabetes_model.pkl`
- `models/kidney_model.pkl`
- `models/scalers.pkl`
- `models/dataset_metadata.json`
- per-disease metrics JSON files
- EDA charts and summaries in `eda_outputs/`

## Streamlit Cloud Deployment

If deployment fails during dependency installation, the most common cause is that the app is trying to install training-only packages that are not needed in production.

This repository is now split into:
- `requirements.txt`
  Runtime dependencies for the deployed app
- `requirements-train.txt`
  Extra packages used only for training and experimentation
- `packages.txt`
  Linux system packages needed by Streamlit Cloud for OCR

Deployment notes:
- Streamlit Cloud should install `requirements.txt`
- `packages.txt` adds `tesseract-ocr` so image OCR can work in deployment
- TensorFlow is kept in `requirements-train.txt` because the deployed app now uses the saved heart model weights for inference without needing the full TensorFlow package
- if your app settings allow choosing Python, prefer Python `3.11`
- after pushing these changes, redeploy or reboot the app

## Run the Streamlit App

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open in browser:

```text
http://localhost:8501
```

## Run Prediction from the Command Line

```powershell
.venv\Scripts\python.exe predict_from_report.py --file path\to\report.pdf
```

Example with output file:

```powershell
.venv\Scripts\python.exe predict_from_report.py --file path\to\report.jpg --out result.json
```

## App Output

The app can show:
- extracted report values
- predicted probability for each disease
- risk band: Low, Medium, or High
- explanation blocks for classical models

Risk bands:
- `0.00 - 0.30` -> `Low`
- `0.30 - 0.60` -> `Medium`
- `0.60 - 1.00` -> `High`

## Current Limitations

- Heart predictions use a Keras neural network, so SHAP explanations for heart are not currently available
- TensorFlow is required for training the heart model, but not for deployed inference
- TensorFlow GPU support is limited on native Windows for newer TensorFlow versions
- deployment environments may still need a restart after dependency changes
- prediction quality depends on report clarity and correct OCR extraction
- this system is for risk estimation, not medical diagnosis

## Important Medical Disclaimer

This application is an AI-assisted risk prediction tool for educational and informational use only.

It does not provide a medical diagnosis.

Always consult a qualified healthcare professional before making any medical decision based on the results.
