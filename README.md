# PSR: Precedents and Statutes Retrieval System

### A Production-Grade Hybrid Neural Search & Cross-Encoder Reranking Architecture for Legal Tech

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-ff4043?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![HuggingFace](https://img.shields.io/badge/AI_Hosting-HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![GitHub Pages](https://img.shields.io/badge/Frontend-GitHub_Pages-222222?style=for-the-badge&logo=github&logoColor=white)](https://pages.github.com/)

---

## 🚀 Live Links
* **Interactive Frontend:** [https://subramani2005.github.io/PSR-Precedents-and-Statutes-Retrieval/](https://subramani2005.github.io/PSR-Precedents-and-Statutes-Retrieval/)

---

## 💡 The Problem & The Solution
Traditional legal information systems rely heavily on rigid keyword matching (BM25), failing to capture structural semantic context, synonym variations, or the deep contextual nuances inherent in legal precedents and complex statutory provisions. 

**PSR** resolves this by introducing a highly robust, fault-tolerant **three-stage neural retrieval pipeline**:

* **Stage 0 (LLM Query Expansion):** Intercepts colloquial user language (e.g., *"husband beating wife daily"*) and leverages a locally hosted `Qwen2.5-0.5B-Instruct` model to autonomously expand the query with formal legal terminology and statutory synonyms (e.g., *"Domestic Violence Act, Section 498A IPC, physical cruelty"*).
* **Stage 1 (Hybrid Semantic Retrieval):** Utilizes `BAAI/bge-m3` to simultaneously compute dense vector embeddings (1024-dimensional continuous vector space) for abstract semantic meaning and sparse vector token weights for exact match precision on specific legal acts or sections.
* **Stage 2 (Neural Reranking):** Leverages a high-performance Cross-Encoder model (`BAAI/bge-reranker-base`) to evaluate the joint attention matrix between the original user query and the combined candidate pools fetched from Qdrant, sorting the output by absolute contextual relevance.

---

## 🏗️ System Architecture

```text
       [ User Query ] (e.g., "husband beats wife daily")
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│        Stage 0: Local LLM Legal Query Expansion         │
│  - Zero-shot inference via local Qwen2.5-0.5B model     │
│  - Lexical transformation: Colloquial -> Legal Lexicon   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        [ Expanded Query String ]
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Stage 1: Hybrid Extraction                 │
│  - Dense Vector Representation (1024-dim Cosine Sim)    │
│  - Sparse Vector Token Frequency Weights                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               Qdrant Cluster Filtering                  │
│  - Parallel Dual Prefetch Vector Strategy               │
│  - Reciprocal Rank Fusion (RRF) Integration             │
│  - Instant Candidate Selection Pool (Top K candidates)  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│             Stage 2: Cross-Encoder Reranking            │
│  - Joint Self-Attention Matrix Evaluation               │
│  - Non-linear logit mapping: f(q, d) -> Score           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
[ Reranked, Relevance-Sorted Legal Context Card Array ]
