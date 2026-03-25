import os
import io
import uuid
import logging
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

import uvicorn

from fusion import (
    load_xgb_model,
    load_cnn_model,
    predict_from_text,
    predict_from_image,
    fuse,
    generate_report,
)
from llm import (
    load_embeddings,
    build_or_load_index,
    full_pipeline_report,
    generate_explanation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class AppState:
    xgb_model   = None
    encoders    = None
    cnn_model   = None
    vectorstore = None
    embeddings  = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models…")
    state.xgb_model, state.encoders = load_xgb_model()
    state.cnn_model                  = load_cnn_model()
    state.embeddings                 = load_embeddings()
    state.vectorstore                = build_or_load_index(state.embeddings)
    logger.info("All models ready ✓")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Osteoporosis Detection API",
    description=(
        "Multi-modal osteoporosis detection using EfficientNetB0 (X-ray) + "
        "XGBoost (clinical data) with weighted fusion and LangChain RAG explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    Age:               int            = Field(..., ge=1,  le=120, example=70)
    Family_History:    str            = Field(..., example="Yes")
    Race_Ethnicity:    str            = Field(..., example="Asian")
    Calcium_Intake:    str            = Field(..., example="Low")
    Physical_Activity: str            = Field(..., example="Sedentary")
    Smoking:           str            = Field(..., example="No")
    Medical_Conditions: str           = Field(..., example="Rheumatoid Arthritis")
    Prior_Fractures:   str            = Field(..., example="Yes")

    def to_fusion_dict(self) -> dict:
        """Map Pydantic fields → keys expected by XGBoost branch."""
        return {
            "Age":               self.Age,
            "Family History":    self.Family_History,
            "Race/Ethnicity":    self.Race_Ethnicity,
            "Calcium Intake":    self.Calcium_Intake,
            "Physical Activity": self.Physical_Activity,
            "Smoking":           self.Smoking,
            "Medical Conditions": self.Medical_Conditions,
            "Prior Fractures":   self.Prior_Fractures,
        }


class DiagnosisResponse(BaseModel):
    request_id:      str
    branch:          str
    diagnosis:       str
    pred:            int
    prob_positive:   float
    confidence:      str
    report:          str
    explanation:     Optional[str] = None


class HealthResponse(BaseModel):
    status:  str
    models:  dict

def _confidence(prob: float) -> str:
    if prob >= 0.85 or prob <= 0.15:
        return "High"
    elif prob >= 0.70 or prob <= 0.30:
        return "Medium"
    return "Low"


async def _save_upload(file: UploadFile) -> str:
    """Save an uploaded image to a temp file; return path."""
    suffix = os.path.splitext(file.filename)[-1] or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()
    return tmp.name


def _build_response(result: dict, patient_data: dict = None,
                    image_path: str = None,
                    include_explanation: bool = True) -> DiagnosisResponse:
    report = generate_report(result, patient_data=patient_data, image_path=image_path)

    explanation = None
    if include_explanation:
        try:
            explanation = generate_explanation(
                result, state.vectorstore,
                patient_data=patient_data,
                image_path=image_path,
            )
        except Exception as e:
            logger.warning("LLM explanation failed: %s", e)
            explanation = f"LLM unavailable: {e}"

    return DiagnosisResponse(
        request_id    = str(uuid.uuid4()),
        branch        = result["branch"],
        diagnosis     = "Osteoporosis Detected" if result["pred"] == 1 else "No Osteoporosis Detected",
        pred          = result["pred"],
        prob_positive = round(result["prob_positive"], 4),
        confidence    = _confidence(result["prob_positive"]),
        report        = report,
        explanation   = explanation,
    )

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health():
    """Check that all models are loaded."""
    return HealthResponse(
        status="ok",
        models={
            "xgboost":    state.xgb_model   is not None,
            "cnn":        state.cnn_model    is not None,
            "vectorstore": state.vectorstore is not None,
        },
    )

@app.post(
    "/predict/text",
    response_model=DiagnosisResponse,
    tags=["Prediction"],
    summary="Predict from clinical data only (XGBoost branch)",
)
def predict_text(
    patient: PatientData,
    explain: bool = True,
):
    """
    Run XGBoost inference on structured clinical data.
    Set `explain=false` to skip the LLM explanation (faster).
    """
    logger.info("predict/text  age=%s", patient.Age)
    pd_dict = patient.to_fusion_dict()
    result  = predict_from_text(pd_dict, state.xgb_model, state.encoders)
    return _build_response(result, patient_data=pd_dict, include_explanation=explain)

@app.post(
    "/predict/image",
    response_model=DiagnosisResponse,
    tags=["Prediction"],
    summary="Predict from X-ray image only (CNN branch)",
)
async def predict_image(
    file:    UploadFile = File(..., description="X-ray image (.jpg / .png)"),
    explain: bool       = True,
):
    """
    Run EfficientNetB0 inference on an uploaded knee X-ray image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image.")

    logger.info("predict/image  file=%s", file.filename)
    img_path = await _save_upload(file)
    try:
        result = predict_from_image(img_path, state.cnn_model)
        return _build_response(result, image_path=img_path, include_explanation=explain)
    finally:
        os.unlink(img_path)

@app.post(
    "/predict/fusion",
    response_model=DiagnosisResponse,
    tags=["Prediction"],
    summary="Predict using both X-ray image + clinical data (fusion)",
)
async def predict_fusion(
    file:              UploadFile = File(..., description="X-ray image (.jpg / .png)"),
    # Clinical fields passed as form data so the endpoint accepts multipart
    Age:               int  = Form(...),
    Family_History:    str  = Form(...),
    Race_Ethnicity:    str  = Form(...),
    Calcium_Intake:    str  = Form(...),
    Physical_Activity: str  = Form(...),
    Smoking:           str  = Form(...),
    Medical_Conditions: str = Form(...),
    Prior_Fractures:   str  = Form(...),
    explain:           bool = Form(True),
):
    """
    Weighted fusion of CNN (image, weight=0.55) and XGBoost (text, weight=0.45).
    Accepts multipart/form-data with both the image file and clinical fields.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image.")

    patient_dict = {
        "Age":               Age,
        "Family History":    Family_History,
        "Race/Ethnicity":    Race_Ethnicity,
        "Calcium Intake":    Calcium_Intake,
        "Physical Activity": Physical_Activity,
        "Smoking":           Smoking,
        "Medical Conditions": Medical_Conditions,
        "Prior Fractures":   Prior_Fractures,
    }

    logger.info("predict/fusion  age=%s  file=%s", Age, file.filename)
    img_path = await _save_upload(file)
    try:
        text_result  = predict_from_text(patient_dict, state.xgb_model, state.encoders)
        image_result = predict_from_image(img_path, state.cnn_model)
        fused        = fuse(text_result, image_result)
        return _build_response(fused, patient_data=patient_dict,
                               image_path=img_path, include_explanation=explain)
    finally:
        os.unlink(img_path)


# ── 4. EXPLAIN ONLY (re-explain an existing result) ──────────────────────────
class ExplainRequest(BaseModel):
    fusion_result: dict  = Field(..., example={
        "branch": "text", "prob_positive": 0.997,
        "prob_negative": 0.003, "pred": 1
    })
    patient_data:  Optional[dict] = None

@app.post(
    "/explain",
    tags=["Explanation"],
    summary="Generate LLM explanation for an existing fusion result",
)
def explain(req: ExplainRequest):
    """
    Re-run the LangChain + FAISS RAG explanation for a previously
    computed fusion result without re-running the ML models.
    """
    try:
        text = generate_explanation(
            req.fusion_result,
            state.vectorstore,
            patient_data=req.patient_data,
        )
        return {"explanation": text}
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
