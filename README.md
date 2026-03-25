# 🦴 Multi-Modal Osteoporosis Detection System

A production-oriented machine learning system for osteoporosis risk prediction using multi-modal data fusion. The system integrates X-ray image analysis (CNN) and clinical data modeling (XGBoost), enhanced with LLM-based clinical explanations using RAG (FAISS + Ollama).

---

## 🚀 Features

- Dual-model architecture (CNN + XGBoost)
- Adaptive input routing (image / text / fusion)
- Weighted fusion for improved prediction
- Explainable AI using RAG + LLM
- Dockerized deployment

---

## 🏗️ Architecture

User Input → CNN / XGBoost → Fusion → Prediction → FAISS → LLM → Explanation

---

## ⚙️ Setup

git clone <your-repo>
cd <your-repo>
docker compose up --build

---

## 📡 Endpoints

- /predict/text
- /predict/image
- /predict/fusion
- /explain

---

## ⚠️ Notes

Models and FAISS index are downloaded from Hugging Face at runtime.

---

## 👨‍💻 Author

Arnav Modi
