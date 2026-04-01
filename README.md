# MediCare AI

MediCare AI is a multi-disease risk prediction project for structured health data and uploaded medical reports.

The system can:
- read PDF and image-based reports
- extract key health values such as age, glucose, blood pressure, cholesterol, creatinine, albumin, and hemoglobin
- estimate risk for heart disease, diabetes, and chronic kidney disease
- show model outputs in a Streamlit web app
- provide AI-powered explanations using Grok (xAI) for model interpretability

## Project Overview

This project combines:
- data preprocessing with `pandas` and `scikit-learn`
- OCR and PDF parsing with `pytesseract` and `pdfplumber`
- classical machine learning with Logistic Regression, Random Forest, XGBoost, and Voting Ensemble
- deep learning for the heart disease model with `TensorFlow` / `Keras`
- a user-friendly frontend built with `Streamlit`

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys (IMPORTANT - Keep Secrets Safe):**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Get your Groq API key from [Groq Console](https://console.groq.com/keys)
   - Edit the `.env` file with your actual API key:
     ```
     GROQ_API_KEY=your_actual_groq_api_key_here
     ```
   - **⚠️ NEVER commit `.env` to GitHub** - it contains sensitive credentials
   - The `.gitignore` file is configured to prevent `.env` from being uploaded

3. **Run the Application:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Diseases Covered

- Heart Disease
- Diabetes
- Chronic Kidney Disease (CKD)

## Current Model Setup

The project currently uses different model strategies for different diseases:

- Heart disease:
  Uses a Keras deep learning neural network only
- Diabetes:
  Uses automatic model selection from classical ML candidates
- CKD:
  Uses automatic model selection from classical ML candidates

Model selection is healthcare-oriented:
- priority 1: higher recall
- priority 2: higher ROC-AUC

This means the project prefers models that miss fewer positive-risk cases.

## Final Selected Models

Based on the latest training run:

| Disease | Final Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Heart Disease | Keras Neural Network | 0.7869 | 0.7500 | 0.9091 | 0.8219 | 0.8983 |
| Diabetes | Ensemble Voting Classifier | 0.7532 | 0.6333 | 0.7037 | 0.6667 | 0.8302 |
| CKD | XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Metric source files:
- `models/heart_metrics.json`
- `models/diabetes_metrics.json`
- `models/kidney_metrics.json`

## Model Comparison Summary

### Heart Disease

Only the neural network is currently used for heart disease.

Selected model:
- Keras Neural Network

Scores:
- Accuracy: `0.7869`
- Precision: `0.7500`
- Recall: `0.9091`
- F1 Score: `0.8219`
- ROC-AUC: `0.8983`

### Diabetes

Models evaluated:
- Logistic Regression
- Random Forest
- XGBoost
- Ensemble Voting
- Keras Neural Network

Selected model:
- Ensemble Voting Classifier

Selected scores:
- Accuracy: `0.7532`
- Precision: `0.6333`
- Recall: `0.7037`
- F1 Score: `0.6667`
- ROC-AUC: `0.8302`

### Chronic Kidney Disease

Models evaluated:
- Logistic Regression
- Random Forest
- XGBoost
- Ensemble Voting
- Keras Neural Network

Selected model:
- XGBoost

Selected scores:
- Accuracy: `1.0000`
- Precision: `1.0000`
- Recall: `1.0000`
- F1 Score: `1.0000`
- ROC-AUC: `1.0000`

## Deep Learning Details

The heart disease pipeline uses a custom Keras binary classifier defined in `keras_model.py`.

The neural network is a tabular binary classification model with:
- dense hidden layers
- ReLU activations
- dropout regularization
- sigmoid output for binary risk prediction
- Adam optimizer
- early stopping during training

Why deep learning is used only for heart:
- you requested heart to use a neural network only
- the project was updated so the heart artifact is always trained from the Keras model
- diabetes and CKD still performed better with the classical model-selection flow

## Features Used

### Heart Disease Feature Mapping

- `age -> age`
- `cholesterol -> chol`
- `bp -> trestbps`

### Diabetes Feature Mapping

- `glucose -> Glucose`
- `bmi -> BMI`
- `age -> Age`
- `bp -> BloodPressure`

### CKD Feature Mapping

- `creatinine -> sc`
- `albumin -> al`
- `hemoglobin -> hemo`
- `age -> age`
- `bp -> bp`
- `urea -> bu`

## Required Inputs Per Disease

Predictions are skipped when the minimum required values are missing.

- Heart disease:
  `age`, `bp`, `cholesterol`
- Diabetes:
  `glucose`, `bmi`, `age`
- CKD:
  `creatinine`, `urea`, `albumin`, `hemoglobin`

## Project Structure

```text
Care/
|-- data/
|-- eda_outputs/
|-- models/
|-- .venv/
|-- keras_model.py
|-- packages.txt
|-- predict_from_report.py
|-- requirements-train.txt
|-- streamlit_app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
```

## Setup

### 1. Create and activate the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
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
