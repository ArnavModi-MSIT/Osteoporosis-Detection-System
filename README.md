Multi-Modal Osteoporosis Detection System
A production-oriented machine learning system for osteoporosis risk prediction using multi-modal data fusion. This project integrates X-ray image analysis (CNN) and clinical risk modeling (XGBoost), enhanced with LLM-based clinical explanations via Retrieval-Augmented Generation (RAG).

Overview
Osteoporosis diagnosis in real-world clinical settings relies on both imaging data (X-rays, bone density scans) and patient-specific risk factors (age, medical history, lifestyle). This system simulates a realistic diagnostic pipeline by:

Ingesting heterogeneous inputs — radiographic images and structured clinical data
Performing modality-specific inference on each input stream
Combining predictions through a learned fusion layer
Generating clinically grounded, interpretable explanations using a retrieval-augmented language model


Key Features
Multi-Modal Learning
The system combines two complementary modeling approaches:

CNN (EfficientNet-B0) for radiographic image analysis
XGBoost for structured clinical risk factor modeling

Each modality is trained and evaluated independently before being integrated into the fusion pipeline.

Adaptive Input Routing
The system automatically detects which input modalities are available and routes accordingly:
Input TypeInference PipelineImage onlyCNNClinical data onlyXGBoostBothWeighted fusion
This design ensures the system degrades gracefully in partial-data scenarios — a common occurrence in clinical environments.

Fusion Layer
Predictions from each modality are combined using a configurable weighted averaging scheme...
