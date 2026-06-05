# CardioBot: Cardiovascular Domain Chatbot

CardioBot is an NLP project that develops a cardiovascular health education chatbot using baseline retrieval, transfer learning, and retrieval-augmented generation.

## Project Overview

This project compares three approaches:

1. **TF-IDF Baseline**
   - Uses TF-IDF vectorization and cosine similarity to retrieve relevant cardiovascular document chunks.
   - Produces extractive answers from retrieved text.

2. **Qwen LoRA Fine-Tuning**
   - Uses Qwen2.5-1.5B-Instruct as the pretrained model.
   - Applies LoRA fine-tuning on cardiovascular Q&A pairs.

3. **RAG + Qwen LoRA**
   - Uses TF-IDF retrieval to retrieve top-k context from raw documents.
   - Uses the fine-tuned Qwen LoRA model to generate grounded answers.

## Dataset

The dataset consists of 29 curated cardiovascular text documents.  
From these documents, 160 Q&A pairs were created and split into:

- Train: 101
- Validation: 25
- Test: 34

## Evaluation Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Answer Similarity |
|---|---:|---:|---:|---:|---:|
| TF-IDF Baseline | 0.2972 | 0.1193 | 0.2180 | 0.0408 | 0.2897 |
| Qwen LoRA | 0.4621 | 0.2320 | 0.3793 | 0.1302 | 0.3624 |
| RAG + Qwen LoRA | 0.5312 | 0.3086 | 0.4509 | 0.1654 | 0.4489 |

The RAG + Qwen LoRA system achieved the best overall result.

## Local App

The final chatbot is implemented using Streamlit.

Run the app locally:

```bash
streamlit run src/app.py