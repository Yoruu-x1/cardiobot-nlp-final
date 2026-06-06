import re
from pathlib import Path

import streamlit as st
import torch
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = BASE_DIR / "models"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = MODEL_DIR / "qwen2_5_1_5b_cardio_lora"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 3
DEFAULT_MAX_NEW_TOKENS = 300
MIN_SCORE = 0.25

SYSTEM_PROMPT = (
    "You are CardioBot, a cardiovascular health education assistant. "
    "Answer only using the provided context. "
    "If the answer is not available in the context, say that the information is not available in the knowledge base. "
    "Do not provide diagnosis, prescriptions, or emergency medical decisions. "
    "For emergency symptoms, advise the user to seek immediate medical help."
)


# =========================
# PAGE SETUP
# =========================

st.set_page_config(
    page_title="CardioBot",
    page_icon="❤️",
    layout="centered"
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        color: #d62828;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .answer-card {
        background-color: #f8f9fb;
        padding: 1.25rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        line-height: 1.6;
        font-size: 1rem;
    }

    .note {
        font-size: 0.85rem;
        color: #777;
        margin-top: 0.75rem;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">❤️ CardioBot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A cardiovascular health education chatbot</div>',
    unsafe_allow_html=True
)

st.info(
    "CardioBot provides educational information only. "
    "It cannot diagnose conditions, prescribe medication, or replace a healthcare professional."
)


# =========================
# DOCUMENT PROCESSING
# =========================

def clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_raw_documents():
    raw_files = sorted(RAW_DIR.glob("*.txt"))

    documents = []

    for file_path in raw_files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        documents.append({
            "source": file_path.name,
            "text": clean_text(text)
        })

    return documents


def chunk_text(documents, chunk_size=180):
    chunks = []

    for doc in documents:
        paragraphs = [p.strip() for p in doc["text"].split("\n") if p.strip()]

        for para in paragraphs:
            if len(para.split()) < 20:
                continue

            sentences = re.split(r'(?<=[.!?])\s+', para)

            current_chunk = []
            current_length = 0

            for sentence in sentences:
                words = sentence.split()

                if current_length + len(words) <= chunk_size:
                    current_chunk.append(sentence)
                    current_length += len(words)
                else:
                    chunk_str = " ".join(current_chunk).strip()

                    if len(chunk_str.split()) >= 30:
                        chunks.append({
                            "source": doc["source"],
                            "text": chunk_str
                        })

                    current_chunk = current_chunk[-1:]
                    current_length = sum(len(s.split()) for s in current_chunk)

                    current_chunk.append(sentence)
                    current_length += len(words)

            if current_chunk:
                chunk_str = " ".join(current_chunk).strip()

                if len(chunk_str.split()) >= 30:
                    chunks.append({
                        "source": doc["source"],
                        "text": chunk_str
                    })

    unique_chunks = []
    seen = set()

    for chunk in chunks:
        if chunk["text"] not in seen:
            unique_chunks.append(chunk)
            seen.add(chunk["text"])

    return unique_chunks


# =========================
# SAFETY GUARD
# =========================

def safety_check(question):
    q = question.lower()

    prescription_keywords = [
        "prescribe",
        "medicine should i take",
        "what medicine",
        "best medicine",
        "dosage",
        "dose",
        "stop taking",
        "should i stop",
        "can i stop",
        "medication"
    ]

    diagnosis_keywords = [
        "diagnose",
        "do i have",
        "am i having",
        "whether i have",
        "confirm if i have"
    ]

    emergency_keywords = [
        "severe chest pain",
        "sudden chest pain",
        "can't breathe",
        "cannot breathe",
        "fainting",
        "face drooping",
        "trouble speaking",
        "stroke symptoms",
        "heart attack symptoms"
    ]

    if any(keyword in q for keyword in emergency_keywords):
        return (
            "This may be an emergency symptom. I cannot diagnose your condition, "
            "but you should seek immediate medical help or contact local emergency services right away."
        )

    if any(keyword in q for keyword in prescription_keywords):
        return (
            "I cannot prescribe, recommend, change, or stop medication. "
            "Please consult a licensed healthcare professional for medication advice, especially for chest pain or heart-related symptoms."
        )

    if any(keyword in q for keyword in diagnosis_keywords):
        return (
            "I cannot diagnose whether you have a specific condition. "
            "I can explain general cardiovascular information, but diagnosis requires evaluation by a healthcare professional."
        )

    return None


# =========================
# FAISS VECTOR RETRIEVER
# =========================

@st.cache_resource
def build_vector_retriever():
    documents = load_raw_documents()
    chunks = chunk_text(documents)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    chunk_texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    embedding_dim = chunk_embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(chunk_embeddings)

    return chunks, embedding_model, faiss_index


def retrieve_faiss_context(question, chunks, embedding_model, faiss_index, top_k=3):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = faiss_index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"],
            "score": float(score)
        })

    return results


# =========================
# MODEL LOADING
# =========================

@st.cache_resource
def load_lora_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None
    )

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    if device == "cpu":
        model.to("cpu")

    return model, tokenizer


# =========================
# RAG GENERATION
# =========================

def build_rag_prompt(question, retrieved_contexts, tokenizer):
    context_text = ""

    for i, item in enumerate(retrieved_contexts, start=1):
        context_text += f"[Context {i} | Source: {item['source']}]\n"
        context_text += item["text"] + "\n\n"

    user_prompt = (
        f"Context:\n{context_text}\n"
        f"Question: {question}\n\n"
        f"Answer clearly and completely based only on the context. "
        f"If the question asks for a process, pathway, or flow, explain the full sequence step by step from beginning to end. "
        f"Do not skip important steps if they are present in the context."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def generate_rag_answer(
    question,
    model,
    tokenizer,
    chunks,
    embedding_model,
    faiss_index,
    top_k=3,
    max_new_tokens=300,
    min_score=0.25
):
    safety_response = safety_check(question)

    if safety_response is not None:
        return safety_response

    retrieved = retrieve_faiss_context(
        question=question,
        chunks=chunks,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        top_k=top_k
    )

    if retrieved[0]["score"] < min_score:
        return "The information is not available in the cardiovascular knowledge base."

    prompt = build_rag_prompt(question, retrieved, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.15,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return answer


# =========================
# LOAD RESOURCES
# =========================

with st.spinner("Preparing CardioBot..."):
    chunks, embedding_model, faiss_index = build_vector_retriever()
    model, tokenizer = load_lora_model()


# =========================
# UI
# =========================

example_questions = [
    "How does blood flow through the heart and body?",
    "What test can check if my heart rhythm is irregular? ECG or Holter Monitor? What are the differences?",
    "Can high cholesterol cause heart problems even if I feel healthy?",
    "What are the warning signs of a stroke?",
    "Can you prescribe medicine for my chest pain?"
]

selected_example = st.selectbox(
    "Choose an example question:",
    [""] + example_questions
)

user_question = st.text_area(
    "Ask your question:",
    value=selected_example,
    height=120,
    placeholder="Example: What are the warning signs of a stroke?"
)

ask_button = st.button(
    "Ask CardioBot",
    type="primary",
    use_container_width=True
)

if ask_button:
    if not user_question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("CardioBot is thinking..."):
            answer = generate_rag_answer(
                question=user_question,
                model=model,
                tokenizer=tokenizer,
                chunks=chunks,
                embedding_model=embedding_model,
                faiss_index=faiss_index,
                top_k=DEFAULT_TOP_K,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                min_score=MIN_SCORE
            )

        st.markdown("### Answer")
        st.markdown(
            f"""
            <div class="answer-card">
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="note">This response is generated from a cardiovascular knowledge base and is intended for educational use only.</div>',
            unsafe_allow_html=True
        )

st.markdown(
    '<div class="footer">CardioBot | Your Mini Heart Doctor</div>',
    unsafe_allow_html=True
)