import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download
import joblib

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE      = 224
THRESHOLD     = 0.50
TEXT_WEIGHT   = 0.45
IMAGE_WEIGHT  = 0.55

HF_REPO = "darkthanos/osteoporosis-models"

XGB_FEATURES = [
    "Age", "Family History", "Race/Ethnicity", "Calcium Intake",
    "Physical Activity", "Smoking", "Medical Conditions",
    "Prior Fractures", "Age_x_FamilyHistory"
]

def load_xgb_model():
    """Download and load XGBoost model and label encoders from HuggingFace."""
    print("Downloading XGBoost model from HuggingFace...")
    xgb_path     = hf_hub_download(repo_id=HF_REPO, filename="xgb_model.joblib")
    encoder_path = hf_hub_download(repo_id=HF_REPO, filename="label_encoders.joblib")

    model    = joblib.load(xgb_path)
    encoders = joblib.load(encoder_path)
    print("XGBoost model loaded ✓")
    return model, encoders


def load_cnn_model():
    """Download and load EfficientNetB0 weights from HuggingFace."""
    print("Downloading CNN model from HuggingFace...")
    cnn_path = hf_hub_download(repo_id=HF_REPO, filename="efficientnetb0_osteoporosis.pth")

    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, 1),
    )
    model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    print("CNN model loaded ✓")
    return model

def predict_from_text(patient_data: dict, xgb_model, encoders) -> dict:
    """
    Run XGBoost inference on structured patient text/clinical data.

    patient_data: dict with keys matching XGB_FEATURES
    Example:
    {
        "Age": 65,
        "Family History": "Yes",
        "Race/Ethnicity": "Asian",
        "Calcium Intake": "Low",
        "Physical Activity": "Sedentary",
        "Smoking": "No",
        "Medical Conditions": "Rheumatoid Arthritis",
        "Prior Fractures": "Yes",
    }
    """
    df = pd.DataFrame([patient_data])

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
            )

    if "Age" in df.columns and "Family History" in df.columns:
        df["Age_x_FamilyHistory"] = df["Age"] * df["Family History"]

    df = df.reindex(columns=XGB_FEATURES, fill_value=0)

    prob_positive = float(xgb_model.predict_proba(df)[0][1])
    prob_negative = 1.0 - prob_positive

    return {
        "branch":           "text",
        "prob_positive":    prob_positive,
        "prob_negative":    prob_negative,
        "pred":             int(prob_positive >= THRESHOLD),
    }


def predict_from_image(image_path: str, cnn_model) -> dict:
    """
    Run EfficientNetB0 inference on a knee X-ray image.

    image_path: path to .jpg / .png X-ray file
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = cnn_model(tensor)
        prob_positive = float(torch.sigmoid(logit).cpu().item())

    prob_negative = 1.0 - prob_positive

    return {
        "branch":           "image",
        "prob_positive":    prob_positive,
        "prob_negative":    prob_negative,
        "pred":             int(prob_positive >= THRESHOLD),
    }

def fuse(text_result: dict, image_result: dict) -> dict:
    """
    Weighted average fusion of text and image branch scores.
    text_weight=0.45, image_weight=0.55
    """
    fused_prob = (
        TEXT_WEIGHT  * text_result["prob_positive"] +
        IMAGE_WEIGHT * image_result["prob_positive"]
    )
    return {
        "branch":        "fusion",
        "prob_positive": fused_prob,
        "prob_negative": 1.0 - fused_prob,
        "pred":          int(fused_prob >= THRESHOLD),
        "text_prob":     text_result["prob_positive"],
        "image_prob":    image_result["prob_positive"],
        "text_weight":   TEXT_WEIGHT,
        "image_weight":  IMAGE_WEIGHT,
    }

def get_confidence(prob: float) -> str:
    if prob >= 0.85 or prob <= 0.15:
        return "High"
    elif prob >= 0.70 or prob <= 0.30:
        return "Medium"
    else:
        return "Low"

def generate_report(result: dict, patient_data: dict = None,
                    image_path: str = None) -> str:
    """
    Generate a full diagnostic report from branch result.
    Works for single-branch or fused results.
    """
    prob     = result["prob_positive"]
    pred     = result["pred"]
    conf     = get_confidence(prob)
    branch   = result["branch"]
    diagnosis = "OSTEOPOROSIS DETECTED" if pred == 1 else "NO OSTEOPOROSIS DETECTED"

    lines = []
    lines.append("=" * 60)
    lines.append("   OSTEOPOROSIS DIAGNOSTIC REPORT")
    lines.append("=" * 60)

    lines.append("\n[INPUT]")
    if branch == "fusion":
        lines.append("  Modalities used : Image (X-ray) + Clinical text data")
    elif branch == "image":
        lines.append("  Modalities used : Image (X-ray) only")
        if image_path:
            lines.append(f"  Image file      : {os.path.basename(image_path)}")
    elif branch == "text":
        lines.append("  Modalities used : Clinical text data only")
        if patient_data:
            lines.append("  Patient data    :")
            for k, v in patient_data.items():
                if k != "Age_x_FamilyHistory":
                    lines.append(f"    {k:<25}: {v}")

    lines.append("\n[DIAGNOSIS]")
    lines.append(f"  Result          : {diagnosis}")
    lines.append(f"  Confidence      : {conf}")
    lines.append(f"  Risk probability: {prob * 100:.1f}%")

    lines.append("\n[BRANCH SCORES]")
    if branch == "fusion":
        lines.append(f"  Text branch     : {result['text_prob']*100:.1f}%  "
                     f"(weight: {result['text_weight']})")
        lines.append(f"  Image branch    : {result['image_prob']*100:.1f}%  "
                     f"(weight: {result['image_weight']})")
        lines.append(f"  Fused score     : {prob * 100:.1f}%")

        text_pred  = int(result["text_prob"]  >= THRESHOLD)
        image_pred = int(result["image_prob"] >= THRESHOLD)
        if text_pred == image_pred:
            lines.append("  Branch agreement: ✓ Both branches agree")
        else:
            lines.append("  Branch agreement: ✗ Branches disagree — "
                         "recommend clinical review")
    elif branch == "image":
        lines.append(f"  Image branch    : {prob * 100:.1f}%")
        lines.append("  Text branch     : Not provided")
    elif branch == "text":
        lines.append(f"  Text branch     : {prob * 100:.1f}%")
        lines.append("  Image branch    : Not provided")

    lines.append("\n[CLINICAL NOTE]")
    if conf == "Low":
        lines.append("  Low confidence — additional testing strongly recommended.")
    elif conf == "Medium":
        lines.append("  Moderate confidence — consider corroborating with additional data.")
    else:
        lines.append("  High confidence result.")

    if branch != "fusion":
        lines.append(f"  Note: Only {branch} modality was provided. "
                     "Providing both X-ray and clinical data enables "
                     "fusion analysis for improved accuracy.")

    lines.append("\n[DISCLAIMER]")
    lines.append("  This report is AI-generated and intended to assist")
    lines.append("  clinical decision-making only. Final diagnosis must")
    lines.append("  be confirmed by a qualified medical professional.")
    lines.append("=" * 60)

    return "\n".join(lines)

def diagnose(patient_data: dict = None, image_path: str = None,
             xgb_model=None, encoders=None, cnn_model=None) -> str:
    """
    Main routing function. Detects what inputs are available
    and routes to the appropriate branch or fusion.
    """
    has_text  = patient_data is not None and len(patient_data) > 0
    has_image = image_path is not None and os.path.exists(image_path)

    if not has_text and not has_image:
        return "ERROR: No input provided. Please supply clinical data, an X-ray image, or both."

    if has_text and has_image:
        print("Both modalities detected → running fusion...")
        text_result  = predict_from_text(patient_data, xgb_model, encoders)
        image_result = predict_from_image(image_path, cnn_model)
        result       = fuse(text_result, image_result)

    elif has_image:
        print("Image modality detected → running CNN branch...")
        result = predict_from_image(image_path, cnn_model)

    elif has_text:
        print("Text modality detected → running XGBoost branch...")
        result = predict_from_text(patient_data, xgb_model, encoders)

    report = generate_report(result, patient_data=patient_data,
                             image_path=image_path)
    return report

def save_xgb_artifacts(best_xgb_model, label_encoders):
    joblib.dump(best_xgb_model, "xgb_model.joblib")
    joblib.dump(label_encoders, "label_encoders.joblib")
    print("Saved → xgb_model.joblib")
    print("Saved → label_encoders.joblib")