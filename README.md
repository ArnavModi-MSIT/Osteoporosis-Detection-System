# Multi-Modal Osteoporosis Detection System

A production-oriented machine learning system for osteoporosis risk prediction using multi-modal data fusion. This project integrates X-ray image analysis (CNN) and clinical risk modeling (XGBoost), enhanced with LLM-based clinical explanations via Retrieval-Augmented Generation (RAG).

---

## Overview

Osteoporosis diagnosis in real-world clinical settings relies on both imaging data (X-rays, bone density scans) and patient-specific risk factors (age, medical history, lifestyle). This system simulates a realistic diagnostic pipeline by:

- Ingesting heterogeneous inputs — radiographic images and structured clinical data
- Performing modality-specific inference on each input stream
- Combining predictions through a learned fusion layer
- Generating clinically grounded, interpretable explanations using a retrieval-augmented language model

---

## Key Features

### Multi-Modal Learning

The system combines two complementary modeling approaches:

- **CNN (EfficientNet-B0)** for radiographic image analysis
- **XGBoost** for structured clinical risk factor modeling

Each modality is trained and evaluated independently before being integrated into the fusion pipeline.

---

### Adaptive Input Routing

The system automatically detects which input modalities are available and routes accordingly:

| Input Type            | Inference Pipeline |
|-----------------------|--------------------|
| Image only            | CNN                |
| Clinical data only    | XGBoost            |
| Both                  | Weighted fusion    |

This design ensures graceful degradation in partial-data scenarios — a common occurrence in real-world clinical environments.

---

### Fusion Layer

Predictions from each modality are combined using a configurable weighted averaging scheme. Weights can be tuned based on modality reliability or clinical context, allowing the system to prioritize one source of evidence over another when appropriate.

---

## Architecture Summary

```
Input
 ├── X-ray Image ──────────► CNN (EfficientNet-B0) ──────────┐
 │                                                            ├──► Fusion Layer ──► Risk Score + Explanation
 └── Clinical Data ────────► XGBoost Classifier ─────────────┘
                                                              │
                                                   RAG-augmented LLM
                                               (Clinical Explanation Generator)
```

---

## Technology Stack

| Component              | Technology            |
|------------------------|-----------------------|
| Image model            | EfficientNet-B0 (CNN) |
| Structured data model  | XGBoost               |
| Fusion strategy        | Weighted averaging    |
| Explanation generation | LLM + RAG             |

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Pipeline

```bash
python run_pipeline.py --image path/to/xray.png --clinical path/to/patient_data.csv
```

For image-only input:

```bash
python run_pipeline.py --image path/to/xray.png
```

For clinical data only:

```bash
python run_pipeline.py --clinical path/to/patient_data.csv
```

---

## Project Structure

```
.
├── models/
│   ├── cnn/              # EfficientNet-B0 training and inference
│   ├── xgboost/          # XGBoost training and inference
│   └── fusion/           # Weighted fusion layer
├── rag/                  # Retrieval-Augmented Generation pipeline
├── data/                 # Data preprocessing and loaders
├── scripts/              # Utility scripts
├── run_pipeline.py       # Main entry point
└── requirements.txt
```

---

## Clinical Disclaimer

This system is intended for research and educational purposes only. It is not validated for clinical use and should not be used as a substitute for professional medical diagnosis or advice.

---

## License

[Add license here]
