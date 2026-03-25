# 🦴 Multi-Modal Osteoporosis Detection System

A production-oriented machine learning system for osteoporosis risk prediction using **multi-modal data fusion**.  
This project integrates **X-ray image analysis (CNN)** and **clinical risk modeling (XGBoost)**, enhanced with **LLM-based clinical explanations using Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Overview

Osteoporosis diagnosis in real-world clinical settings relies on both **imaging data** (X-rays, scans) and **patient-specific risk factors** (age, medical history, lifestyle).

This system simulates a **real-world diagnostic pipeline** by:

- Processing heterogeneous inputs (image + structured data)
- Performing modality-specific inference
- Combining predictions via a fusion layer
- Generating clinically grounded explanations using an LLM

---

## ✨ Key Features

### 🧠 Multi-Modal Learning
- **CNN (EfficientNet-B0)** for X-ray image analysis  
- **XGBoost** for structured clinical data  
- Handles **image-only**, **text-only**, and **combined inputs**

---

### 🔀 Adaptive Input Routing
Automatically detects available input type:

| Input Type | Pipeline |
|----------|---------|
| Image only | CNN |
| Clinical data only | XGBoost |
| Both | Weighted fusion |

---

### ⚖️ Fusion Layer
- Combines predictions from both modalities  
- Uses weighted averaging:
