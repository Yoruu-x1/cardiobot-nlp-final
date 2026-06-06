# CardioBot: Cardiovascular Domain Chatbot

CardioBot is an NLP final project that develops a cardiovascular health education chatbot using baseline retrieval, transfer learning, and Retrieval-Augmented Generation (RAG).

The project compares traditional retrieval, LoRA fine-tuning, TF-IDF-based RAG, and FAISS vector database-based RAG.

## Project Overview

This project compares four approaches:

1. **TF-IDF Baseline**
   - Uses TF-IDF vectorization and cosine similarity.
   - Retrieves relevant cardiovascular document chunks.
   - Produces extractive answers from retrieved text.

2. **Qwen LoRA**
   - Uses `Qwen/Qwen2.5-1.5B-Instruct` as the pretrained model.
   - Applies LoRA fine-tuning on cardiovascular Q&A pairs.
   - Generates answers without retrieving external document context during inference.

3. **TF-IDF RAG + Qwen LoRA**
   - Uses TF-IDF retrieval to retrieve top-k contexts from raw documents.
   - Uses the fine-tuned Qwen LoRA model to generate grounded answers.

4. **FAISS RAG + Qwen LoRA**
   - Uses sentence embeddings and FAISS vector search for semantic retrieval.
   - Uses the fine-tuned Qwen LoRA model as the generator.
   - Selected as the final system based on overall evaluation.

## Dataset

The dataset consists of 29 curated cardiovascular text documents.

From these documents, 160 Q&A pairs were created and split into:

- Train: 101
- Validation: 25
- Test: 34

The Q&A dataset was created using an AI-assisted and human-reviewed process. Each Q&A pair was grounded in the collected cardiovascular documents.

## Topics Covered

The dataset covers multiple cardiovascular topics, including:

- Angiography
- Angioplasty and Stent
- Arrhythmia
- Atherosclerosis
- Blood Pressure Measurement
- Blood Flow
- Cardiac Ablation
- Cardiac Rehabilitation
- Cardiomyopathy
- Cholesterol
- Congenital Heart Disease
- Coronary Artery Disease
- Echocardiogram
- Electrocardiogram
- Heart Valve Disease
- Heart Disease
- Holter Monitor
- Pacemaker
- Peripheral Artery Disease
- Stress Test
- Stroke
- Circulation
- Heart Anatomy and Function
- Heart Disease Risk Factors
- Heart Failure
- Hypertension
- Prevention
- Treatment for Heart Disease
- Warning Signs of Heart Attack

## Folder Structure

```text
CardioBot_NLP_Final/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_baseline_tfidf.ipynb
│   ├── 03_lora_finetuning.ipynb
│   ├── 04_rag_chatbot.ipynb
│   ├── 04b_vector_db_rag_experiment.ipynb
│   └── 05_evaluation_results.ipynb
├── results/
├── src/
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore
