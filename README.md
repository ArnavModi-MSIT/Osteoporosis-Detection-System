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

- **CNN (EfficientNet-B0)** for radiographic image analysis — model weights hosted on Hugging Face
- **XGBoost** for structured clinical risk factor modeling

Each modality is trained and evaluated independently before being integrated into the fusion pipeline.

---

### Adaptive Input Routing

The system automatically detects which input modalities are available and routes accordingly:

| Input Type             | Inference Pipeline |
|------------------------|--------------------|
| Image only             | CNN                |
| Clinical data only     | XGBoost            |
| Both                   | Weighted fusion    |

This design ensures graceful degradation in partial-data scenarios — a common occurrence in real-world clinical environments.

---

### Fusion Layer

Predictions from each modality are combined using a configurable weighted averaging scheme (`fusion.py`). Weights can be tuned based on modality reliability or clinical context.

---

### RAG-Based Clinical Explanations

The `llm.py` module uses a FAISS vector index (`faiss_index/`) backed by a curated clinical knowledge base (`knowledge_base/`) to retrieve relevant context. This context is passed to an LLM to generate grounded, human-readable risk explanations alongside each prediction.

---

## Architecture Summary

```
Input
 ├── X-ray Image ──────────► cnn.py (EfficientNet-B0) ───────┐
 │                                                            ├──► fusion.py ──► Risk Score
 └── Clinical Data ────────► xg_boost.py (XGBoost) ──────────┘                      │
                                                                                      ▼
                                                                   llm.py + FAISS (RAG)
                                                               (Clinical Explanation Generator)
```

---

## Technology Stack

| Component              | Technology                  |
|------------------------|-----------------------------|
| Image model            | EfficientNet-B0 (CNN)       |
| Model weights          | Hugging Face                |
| Structured data model  | XGBoost                     |
| Fusion strategy        | Weighted averaging          |
| Vector search          | FAISS                       |
| Explanation generation | LLM + RAG (knowledge_base/) |
| Containerization       | Docker / docker-compose     |

---

## Project Structure

This is what you get after cloning the repository. Model weights, indexes, and data files are not included — see the [Hugging Face section](#hugging-face-model-repository) below.

```
osteoporosis/
├── cnn.py                # EfficientNet-B0 inference pipeline
├── xg_boost.py           # XGBoost inference pipeline
├── fusion.py             # Weighted fusion of modality predictions
├── llm.py                # RAG-based clinical explanation generator
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-service container orchestration
├── README.md
├── .dockerignore
└── .gitignore
```

> The following are downloaded from Hugging Face before running (see setup steps below):
> `efficientnetb0_osteoporosis.pth`, `xg_model.joblib`, `label_encoders.joblib`, `faiss_index/`, `knowledge_base/`

---

## Hugging Face Model Repository

All model weights and supporting assets are hosted at:  
**[darkthanos/osteoporosis-models](https://huggingface.co/darkthanos/osteoporosis-models)**

| File                              | Description                                  |
|-----------------------------------|----------------------------------------------|
| `efficientnetb0_osteoporosis.pth` | EfficientNet-B0 CNN weights                  |
| `xg_model.joblib`                 | Trained XGBoost model                        |
| `label_encoders.joblib`           | Fitted label encoders for clinical features  |
| `faiss_index/`                    | FAISS vector index for RAG retrieval         |
| `knowledge_base/`                 | Clinical knowledge base documents            |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ArnavModi-MSIT/osteoporosis.git
cd osteoporosis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model Weights and Assets

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="darkthanos/osteoporosis-models",
    local_dir="."
)
```

Or download individual files:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="darkthanos/osteoporosis-models", filename="efficientnetb0_osteoporosis.pth", local_dir=".")
hf_hub_download(repo_id="darkthanos/osteoporosis-models", filename="xg_model.joblib", local_dir=".")
hf_hub_download(repo_id="darkthanos/osteoporosis-models", filename="label_encoders.joblib", local_dir=".")
```

### 4. Run the Pipeline

```bash
python main.py --image path/to/xray.png --clinical path/to/patient_data.csv
```

For image-only input:

```bash
python main.py --image path/to/xray.png
```

For clinical data only:

```bash
python main.py --clinical path/to/patient_data.csv
```

### Running with Docker

```bash
docker-compose up --build
```

---

## Clinical Disclaimer

This system is intended for research and educational purposes only. It is not validated for clinical use and should not be used as a substitute for professional medical diagnosis or advice.
