# PSR-Precedents-and-Statutes-Retrieval-
# PSR: Precedents and Statutes Retrieval System

### A Production-Grade Hybrid Neural Search & Cross-Encoder Reranking Architecture for Legal Tech

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-ff4043?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![HuggingFace](https://img.shields.io/badge/AI_Hosting-HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![GitHub Pages](https://img.shields.io/badge/Frontend-GitHub_Pages-222222?style=for-the-badge&logo=github&logoColor=white)](https://pages.github.com/)

---

## 🚀 Live Links
* **Interactive Frontend:** [Your GitHub Pages Link Here]
* **Production API Gateway:** [Your Hugging Face Space Direct Link Here]/docs
* **Public Dataset Card:** [Your Hugging Face Dataset or Kaggle Link Here]

---

## 💡 The Problem & The Solution
Traditional legal information systems rely heavily on rigid keyword matching (BM25), failing to capture structural semantic context, synonym variations, or the deep contextual nuances inherent in legal precedents and complex statutory provisions. 

**PSR** resolves this by introducing a highly robust, fault-tolerant **two-stage retrieval pipeline**:
1. **Stage 1 (Hybrid Semantic Retrieval):** Utilizes `BAAI/bge-m3` to simultaneously compute dense vector embeddings for deep semantic meaning and sparse vector activations for exact keyword token matching.
2. **Stage 2 (Neural Reranking):** Leverages a high-performance Cross-Encoder model (`BAAI/bge-reranker-base`) to evaluate the joint attention between the query and the top candidates, producing highly accurate, relevance-sorted legal contexts.

---

## 🏗️ System Architecture

```text
[ User Query ] 
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│              Stage 1: Hybrid Extraction                 │
│  - Dense Vector Representation (1024-dim Cosine Sim)   │
│  - Sparse Vector Token Frequency Weights                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               Qdrant Cluster Filtering                  │
│  - Dual Prefetch Vector Strategy                        │
│  - Instant Candidate Selection Pool (Top K)            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Stage 2: Cross-Encoder Reranking           │
│  - Joint Self-Attention Matrix Evaluation               │
│  - Non-linear logit mapping: f(q, d) -> Score           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
[ Reranked, Sorted Relevance Output Card Array ]
