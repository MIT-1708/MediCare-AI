import argparse
import json
import os
import re
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd


def extract_text_from_pdf(file_path: str) -> str:
    import pdfplumber

    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def extract_text_from_image(file_path: str) -> str:
    import os
    import pytesseract
    from PIL import Image
    from pytesseract import TesseractNotFoundError

    # Allow explicit override by environment variable, useful for containers/CI
    tesseract_cmd = os.environ.get(
        "TESSERACT_CMD",
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
    )
    if tesseract_cmd and os.path.exists(tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    img = Image.open(file_path)
    try:
        return pytesseract.image_to_string(img)
    except TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract OCR binary not found. "
            "Install tesseract and ensure it is in PATH, or set the environment variable "
            "TESSERACT_CMD to the full path of tesseract.exe. "
            "See README for instructions."
        ) from e


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(file_path)
    raise ValueError("Unsupported file format. Use PDF, PNG, JPG, or JPEG.")


def to_float(raw: str):
    try:
        return float(raw)
    except Exception:
        return None


def parse_medical_values(text: str) -> Dict[str, float]:
    normalized = re.sub(r"[ \t]+", " ", text.lower())

    patterns = {
        "age": [
            r"\bage\b\s*[:\-]?\s*(\d{1,3}(?:\.\d+)?)",
        ],
        "glucose": [
            r"\b(?:glucose|blood glucose|rbs|fbs)\b\s*[:\-]?\s*(\d{1,4}(?:\.\d+)?)",
        ],
        "bp": [
            r"\b(?:blood pressure|bp)\b\s*[:\-]?\s*(\d{2,3})(?:\s*/\s*\d{2,3})?",
        ],
        "cholesterol": [
            r"\b(?:cholesterol|total cholesterol)\b\s*[:\-]?\s*(\d{1,4}(?:\.\d+)?)",
        ],
        "hemoglobin": [
            r"\b(?:hemoglobin|haemoglobin|hb|hemo)\b\s*[:\-]?\s*(\d{1,3}(?:\.\d+)?)",
        ],
        "creatinine": [
            r"\b(?:creatinine|serum creatinine|sc)\b\s*[:\-]?\s*(\d{1,3}(?:\.\d+)?)",
        ],
        "bmi": [
            r"\bbmi\b\s*[:\-]?\s*(\d{1,3}(?:\.\d+)?)",
        ],
        "albumin": [
            r"\b(?:albumin|al)\b\s*[:\-]?\s*(\d{1,3}(?:\.\d+)?)",
        ],
        "urea": [
            r"\b(?:urea|blood urea|bu)\b\s*[:\-]?\s*(\d{1,4}(?:\.\d+)?)",
        ],
    }

    parsed = {}
    for field, regs in patterns.items():
        val = None
        for rgx in regs:
            m = re.search(rgx, normalized, flags=re.IGNORECASE)
            if m:
                val = to_float(m.group(1))
                if val is not None:
                    break
        if val is not None:
            parsed[field] = val
    return parsed


def risk_band(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    if prob < 0.60:
        return "Medium"
    return "High"


def check_required_features(parsed_values: dict, required_features: list) -> Tuple[bool, list]:
    """Validate that all *required_features* are present in *parsed_values*.

    Returns a tuple ``(available, missing_list)`` where ``available`` is True if
    every feature in *required_features* exists in *parsed_values* (and is not
    ``None``). ``missing_list`` contains features that are absent.
    """
    missing = [f for f in required_features if f not in parsed_values]
    return (len(missing) == 0, missing)


def load_models(models_dir: str = "models"):
    bundles = {
        "heart": joblib.load(os.path.join(models_dir, "heart_model.pkl")),
        "diabetes": joblib.load(os.path.join(models_dir, "diabetes_model.pkl")),
        "ckd": joblib.load(os.path.join(models_dir, "kidney_model.pkl")),
    }
    scalers = joblib.load(os.path.join(models_dir, "scalers.pkl"))
    return bundles, scalers


def _build_raw_feature_row(
    disease: str, parsed: Dict[str, float], summary: Dict[str, object]
) -> pd.DataFrame:
    cols = summary["feature_columns"]
    means = summary["means"]
    row = {}

    if disease == "heart":
        mapping = {"age": "age", "cholesterol": "chol", "bp": "trestbps"}
    elif disease == "diabetes":
        mapping = {"glucose": "Glucose", "bmi": "BMI", "age": "Age", "bp": "BloodPressure"}
    else:
        mapping = {"creatinine": "sc", "albumin": "al", "hemoglobin": "hemo", "age": "age", "bp": "bp", "urea": "bu"}

    for c in cols:
        if c in means:
            row[c] = float(means[c])
        else:
            row[c] = np.nan

    for src, dst in mapping.items():
        if src in parsed and dst in cols:
            row[dst] = float(parsed[src])

    # If a non-numeric feature has no parser signal, leave as NaN (mode-imputed in preprocessor)
    return pd.DataFrame([row], columns=cols)


def _grok_explain(bundle: Dict[str, object], X_row_transformed, parsed_values: Dict[str, float], disease: str, top_k: int = 10):
    try:
        import openai
        import os
        from dotenv import load_dotenv

        # Load environment variables
        load_dotenv()

        # Support xAI / Grok (xAI) and Groq providers via separate env vars
        api_key = os.getenv("API_KEY")
        api_key = os.getenv("API_KEY")
        api_key = grok_key or groq_key
        if not api_key:
            return {"error": "Grok/Groq API key is missing. Set GROK_API_KEY or GROQ_API_KEY in .env."}

        # provider base-url fallback list; user can override with env
        base_url = os.getenv("API_BASE_URL")
        if not base_url:
            if groq_key:
                base_url = "https://api.groq.com/openai/v1"
            else:
                base_url = "https://api.x.ai/v1"

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        model = bundle["model"]
        model_name = model.__class__.__name__.lower()
        feat_names = bundle.get("feature_names_transformed", [])

        # Get prediction probability
        prob = float(model.predict_proba(X_row_transformed)[0, 1])

        # Prepare context for Grok
        disease_names = {
            "heart": "heart disease",
            "diabetes": "diabetes",
            "ckd": "chronic kidney disease"
        }

        disease_full_name = disease_names.get(disease, disease)

        # Create feature importance context
        feature_context = ""
        if parsed_values:
            feature_context = "Available medical values: " + ", ".join([f"{k}: {v}" for k, v in parsed_values.items()])

        # Create prompt for Grok - simple, short, easy to understand
        prompt = f"""Explain in 2-3 simple sentences why the model predicts {prob:.1%} risk for {disease_full_name}.
Medical values: {feature_context if feature_context else 'Not provided'}
Keep it very simple - like talking to someone without medical training."""

        # Get explanation from Grok/Groq; try known model IDs with fallback
        last_exception = None
        # Groq models: llama-3.3, llama-3.1, compound; xAI models: grok-*
        model_candidates = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "groq/compound", "grok-1", "grok-mini"]

        for grok_model in model_candidates:
            try:
                response = client.chat.completions.create(
                    model=grok_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant. Explain health risks in simple language."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                used_model = grok_model
                break
            except Exception as inner_exc:
                last_exception = inner_exc
        else:
            return {"error": f"Grok explanation unavailable: {last_exception}"}

        if not response or not hasattr(response, "choices"):
            return {"error": "Grok explanation failed (no response)."}

        explanation_text = response.choices[0].message.content.strip()

        return {
            "explanation_text": explanation_text,
            "risk_probability": prob,
            "model_type": model_name,
            "disease": disease_full_name,
            "grok_model": used_model,
            "global_importance": [],
            "individual_positive": [],
            "individual_negative": [],
        }

    except Exception as exc:
        return {"error": f"Grok explanation unavailable: {exc}"}


def predict_from_parsed(parsed: Dict[str, float], models_dir: str = "models") -> Dict[str, object]:
    bundles, _scalers = load_models(models_dir=models_dir)
    output = {}
    explanations = {}

    # required features defined against the keys returned by parse_medical_values
    required_map = {
        "heart": ["age", "bp", "cholesterol"],
        "diabetes": ["glucose", "bmi", "age"],
        "ckd": ["creatinine", "urea", "albumin", "hemoglobin"],
    }

    for disease, bundle in bundles.items():
        required = required_map.get(disease, [])
        available, missing = check_required_features(parsed, required)
        if not available:
            # do not build features or run the model
            output[disease] = {"status": "insufficient_data", "missing": missing}
            explanations[disease] = {}
            continue

        summary = bundle["summary"]
        preprocessor = bundle["preprocessor"]
        model = bundle["model"]

        raw_row = _build_raw_feature_row(disease, parsed, summary)
        x = preprocessor.transform(raw_row)
        prob = float(model.predict_proba(x)[0, 1])

        output[disease] = {"status": "ok", "prob": round(prob, 4), "risk": risk_band(prob)}
        explanations[disease] = _grok_explain(bundle, x, parsed, disease)

    return {"predictions": output, "explanations": explanations}


def run(file_path: str, models_dir: str = "models") -> Dict[str, object]:
    text = extract_text(file_path)
    parsed = parse_medical_values(text)
    pred = predict_from_parsed(parsed, models_dir=models_dir)
    result = {
        "parsed_values": parsed,
        "predictions": pred["predictions"],
        "explanations": pred["explanations"],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict multi-disease risks from uploaded health report.")
    parser.add_argument("--file", required=True, help="Path to report file (.pdf/.png/.jpg/.jpeg)")
    parser.add_argument("--models-dir", default="models", help="Directory containing saved model artifacts.")
    parser.add_argument("--out", default="", help="Optional output JSON file path.")
    args = parser.parse_args()

    result = run(args.file, models_dir=args.models_dir)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)


if __name__ == "__main__":
    main()
