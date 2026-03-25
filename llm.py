import os
import shutil
import sys
import requests
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from huggingface_hub import snapshot_download

OLLAMA_URL       = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
PRIMARY_MODEL    = "mistral"
FALLBACK_MODEL   = "mistral"
THRESHOLD        = 0.50
KB_FOLDER        = "knowledge_base"
FAISS_INDEX_PATH = "faiss_index"
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_CHUNKS     = 5

# Hugging Face repo
HF_REPO = "darkthanos/osteoporosis-models"

FALLBACK_KB = [
    "Osteoporosis is a systemic skeletal disease characterized by low bone mass and "
    "microarchitectural deterioration of bone tissue, leading to enhanced bone fragility "
    "and susceptibility to fracture. The WHO defines osteoporosis as a bone mineral density "
    "T-score of -2.5 or below at the femoral neck or lumbar spine.",
    "Age is the strongest independent risk factor for osteoporosis. Bone mass peaks around "
    "age 30 and declines progressively thereafter. After age 50, women lose bone mass at "
    "1-2% per year; men at 0.5-1% per year. By age 70, nearly 30% of women meet criteria "
    "for osteoporosis.",
    "Family history of osteoporosis or fragility fracture is a significant independent risk "
    "factor. First-degree relatives of osteoporotic patients have a 2-3x increased risk. "
    "Genetic factors account for approximately 60-80% of peak bone mass variance.",
    "A prior fragility fracture is one of the strongest predictors of future fracture risk, "
    "increasing risk by 86% regardless of BMD status. The FRAX tool incorporates prior "
    "fracture history as a major risk factor in 10-year fracture probability estimation.",
    "Adequate calcium and vitamin D intake are essential for bone health. Recommended daily "
    "calcium intake is 1000-1200mg for adults over 50. Vitamin D deficiency impairs calcium "
    "absorption and increases PTH-mediated bone resorption.",
    "Weight-bearing and resistance exercise are proven interventions for bone health. "
    "Sedentary lifestyle is associated with 1.5-2x increased fracture risk. Regular exercise "
    "increases bone mineral density by 1-3% and reduces fall risk by 23%.",
    "Rheumatoid arthritis is associated with a 1.5-2x increased risk of osteoporosis due to "
    "systemic inflammation, corticosteroid use, and reduced physical activity.",
    "Conventional knee radiography can suggest osteoporosis through decreased cortical "
    "thickness and trabecular pattern changes. However, plain radiographs can only detect "
    "osteoporosis after 30-40% bone loss has occurred. DXA remains the gold standard.",
    "First-line pharmacological treatment includes bisphosphonates (alendronate, zoledronic "
    "acid) which reduce vertebral fracture risk by 40-70%. Denosumab is an alternative for "
    "patients who cannot tolerate bisphosphonates.",
    "WHO and USPSTF recommend DXA screening for all women aged 65+ and younger postmenopausal "
    "women with risk factors. IOF recommends screening men aged 70+ or 50+ with risk factors.",
    "Discordance between imaging and clinical risk assessment occurs in approximately 20-30% "
    "of cases. Clinical risk factor assessment and imaging findings should be integrated by "
    "a physician for final diagnosis.",
]


def load_embeddings():
    print("Loading embedding model (first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Embedding model ready.\n")
    return embeddings


def download_from_hf():
    """
    Download faiss_index and knowledge_base folders from HuggingFace
    into the local working directory if not already present.
    """
    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    kb_exists  = os.path.exists(KB_FOLDER) and len(os.listdir(KB_FOLDER)) > 0

    if os.path.exists(index_file) and kb_exists:
        print("FAISS index and knowledge base already cached locally.")
        return

    print("Downloading faiss_index and knowledge_base from HuggingFace...")
    local_dir = snapshot_download(
        repo_id=HF_REPO,
        allow_patterns=["faiss_index/*", "knowledge_base/*"],
    )

    hf_faiss = os.path.join(local_dir, "faiss_index")
    if os.path.exists(hf_faiss) and not os.path.exists(FAISS_INDEX_PATH):
        shutil.copytree(hf_faiss, FAISS_INDEX_PATH)
        print(f"faiss_index copied to '{FAISS_INDEX_PATH}' ✓")

    hf_kb = os.path.join(local_dir, "knowledge_base")
    if os.path.exists(hf_kb) and not os.path.exists(KB_FOLDER):
        shutil.copytree(hf_kb, KB_FOLDER)
        print(f"knowledge_base copied to '{KB_FOLDER}' ✓")


def load_pdfs_from_folder(folder: str) -> list:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created '{folder}/' — drop your medical PDFs there.")
        return []

    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in '{folder}/' — using fallback knowledge base.")
        return []

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )

    all_docs = []
    for fname in pdf_files:
        path = os.path.join(folder, fname)
        try:
            loader = PyPDFLoader(path)
            chunks = splitter.split_documents(loader.load())
            all_docs.extend(chunks)
            print(f"  '{fname}' → {len(chunks)} chunks")
        except Exception as e:
            print(f"  WARNING: Could not load '{fname}' — {e}")

    print(f"Total chunks: {len(all_docs)}\n")
    return all_docs


def build_or_load_index(embeddings, kb_folder=KB_FOLDER, index_path=FAISS_INDEX_PATH):
    download_from_hf()

    index_file = os.path.join(index_path, "index.faiss")
    if os.path.exists(index_path) and os.path.exists(index_file):
        print(f"Loading existing FAISS index from '{index_path}'...")
        try:
            vectorstore = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            print("FAISS index loaded ✓\n")
            return vectorstore
        except Exception as e:
            print(f"WARNING: Failed to load FAISS index ({e}) — rebuilding...")

    docs = load_pdfs_from_folder(kb_folder)
    if docs:
        print("Building FAISS index from PDFs...")
        vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        print("Building FAISS index from fallback knowledge base...")
        fallback_docs = [
            Document(page_content=text, metadata={"source": "fallback_kb"})
            for text in FALLBACK_KB
        ]
        vectorstore = FAISS.from_documents(fallback_docs, embeddings)

    vectorstore.save_local(index_path)
    print(f"FAISS index saved to '{index_path}'\n")
    return vectorstore


def rebuild_index(embeddings):
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
        print(f"Removed old index at '{FAISS_INDEX_PATH}'")
    return build_or_load_index(embeddings)


def retrieve_context(query: str, vectorstore, k=TOP_K_CHUNKS) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "knowledge base")
        source = os.path.basename(source) if "/" in source or "\\" in source else source
        chunks.append(f"[{i}] (Source: {source})\n{doc.page_content.strip()}")
    return "\n\n".join(chunks)


def build_retrieval_query(fusion_result: dict, patient_data: dict = None) -> str:
    parts = ["osteoporosis bone density diagnosis"]

    if patient_data:
        if patient_data.get("Age"):
            parts.append(f"age {patient_data['Age']} bone loss risk")
        if str(patient_data.get("Family History", "")).lower() == "yes":
            parts.append("family history osteoporosis genetic risk")
        if str(patient_data.get("Prior Fractures", "")).lower() == "yes":
            parts.append("prior fracture fragility fracture risk")
        if str(patient_data.get("Physical Activity", "")).lower() == "sedentary":
            parts.append("sedentary lifestyle bone density exercise")
        if str(patient_data.get("Calcium Intake", "")).lower() == "low":
            parts.append("calcium deficiency bone health")
        mc = patient_data.get("Medical Conditions", "")
        if mc:
            parts.append(f"{mc} osteoporosis risk")

    if fusion_result.get("branch") in ["image", "fusion"]:
        parts.append("knee xray radiograph osteoporosis imaging findings")

    if fusion_result.get("branch") == "fusion":
        text_pred  = int(fusion_result.get("text_prob",  0) >= THRESHOLD)
        image_pred = int(fusion_result.get("image_prob", 0) >= THRESHOLD)
        if text_pred != image_pred:
            parts.append("discordance imaging clinical risk assessment")

    return " ".join(parts)


def build_prompt(fusion_result: dict, patient_data: dict,
                 image_path: str, retrieved_context: str) -> str:

    branch    = fusion_result.get("branch", "unknown")
    prob      = fusion_result.get("prob_positive", 0)
    pred      = fusion_result.get("pred", 0)
    diagnosis = "Osteoporosis Detected" if pred == 1 else "No Osteoporosis Detected"

    input_section = ""
    if branch == "fusion":
        text_pred  = int(fusion_result.get("text_prob",  0) >= THRESHOLD)
        image_pred = int(fusion_result.get("image_prob", 0) >= THRESHOLD)
        agreement  = "AGREE" if text_pred == image_pred else "DISAGREE"
        input_section = (
            f"Modality          : Both X-ray image and clinical data (fusion)\n"
            f"Text branch score : {fusion_result['text_prob']*100:.1f}%\n"
            f"Image branch score: {fusion_result['image_prob']*100:.1f}%\n"
            f"Fused risk score  : {prob*100:.1f}%\n"
            f"Branch agreement  : {agreement}\n"
        )
    elif branch == "image":
        fname = os.path.basename(image_path) if image_path else "unknown"
        input_section = (
            f"Modality         : X-ray image only\n"
            f"Image file       : {fname}\n"
            f"Image risk score : {prob*100:.1f}%\n"
        )
    elif branch == "text":
        input_section = f"Modality         : Clinical data only\nRisk score       : {prob*100:.1f}%\n"
        if patient_data:
            input_section += "Patient profile  :\n"
            for k, v in patient_data.items():
                if k != "Age_x_FamilyHistory":
                    input_section += f"  {k}: {v}\n"

    prompt = f"""You are a clinical AI assistant specializing in bone health and osteoporosis.
Explain the following AI diagnostic result to a physician clearly and concisely.
Use ONLY the provided medical knowledge to ground your explanation.
Do NOT make final clinical decisions.

RETRIEVED MEDICAL KNOWLEDGE:
{retrieved_context}

AI DIAGNOSTIC RESULT:
Diagnosis : {diagnosis}
{input_section}
Provide a structured response covering:
1. Clinical interpretation of this result
2. Key risk factors that contributed (if clinical data available)
3. Evidence-based context from the medical knowledge above
4. If branches disagreed, what that discordance likely means
5. Recommended next steps for the physician
6. Limitations of this AI assessment

Be concise and professional. Write for a physician audience.
Do not repeat the prompt or medical knowledge — just provide the explanation."""

    return prompt


def call_ollama(prompt: str, model: str = None) -> str:
    if model is None:
        model = PRIMARY_MODEL

    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
            "top_p":       0.9,
            "stop":        ["USER:", "<|im_end|>"],
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        text = response.json().get("response", "").strip()

        cutoffs = ["ASSISTANT:"]
        for cutoff in cutoffs:
            if cutoff in text:
                text = text[text.find(cutoff) + len(cutoff):].strip()
                break

        return text

    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama not running. Start with: ollama serve"
    except requests.exceptions.HTTPError as e:
        if "404" in str(e) and model == PRIMARY_MODEL:
            print(f"'{model}' not found — falling back to {FALLBACK_MODEL}...")
            return call_ollama(prompt, model=FALLBACK_MODEL)
        return f"ERROR: {e}"
    except requests.exceptions.Timeout:
        return "ERROR: Ollama timed out. Try reducing num_predict or use a smaller model."
    except Exception as e:
        return f"ERROR: {e}"


def generate_explanation(fusion_result: dict, vectorstore,
                         patient_data: dict = None,
                         image_path: str = None) -> str:
    query   = build_retrieval_query(fusion_result, patient_data)
    context = retrieve_context(query, vectorstore)
    prompt  = build_prompt(fusion_result, patient_data, image_path, context)
    return call_ollama(prompt)


def full_pipeline_report(fusion_result: dict, fusion_report: str,
                         vectorstore,
                         patient_data: dict = None,
                         image_path: str = None) -> str:
    print("Retrieving from FAISS + generating explanation via Ollama...")
    explanation = generate_explanation(
        fusion_result, vectorstore, patient_data, image_path
    )
    lines = [
        fusion_report,
        "\n" + "=" * 60,
        "   LLM CLINICAL EXPLANATION  (LangChain + FAISS RAG)",
        "=" * 60,
        explanation,
        "=" * 60,
    ]
    return "\n".join(lines)