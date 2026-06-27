# Ecommerce Customer Support — RAG with Evaluation Harness & CI Quality Gates

> A production-style RAG system for ecommerce customer support, with a built-in
> evaluation harness (retrieval metrics + LLM-as-judge scoring) and a CI quality
> gate that blocks bad deployments automatically.

## Problem

LLM-powered support chatbots hallucinate policy details, miss relevant context,
or give incomplete answers — leading to financial loss, compliance risk, and
escalations. This project builds a RAG pipeline where every answer is **faithful,
complete, correct, and auditable**, and adds an automated harness that scores
and gates every change before it reaches production.

## Architecture

    Customer Query
       │
       ▼
    RAG Pipeline
       ├──► Vector DB (KB Articles)   — ChromaDB
       ├──► Vector DB (Tickets)       — ChromaDB  
       └──► SQL DB (Product SKU)      — SQLite
       │
       ▼
    LLM Generation (Ollama llama3)
       │
       ▼
    Evaluation Harness ──► CI Quality Gate


## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (free, local) |
| Vector Store | ChromaDB (local, persistent) |
| Structured Data | SQLite (SKU, warranty, returnable flags) |
| LLM | Ollama llama3 (local, no API cost) |
| Orchestration | LangChain + LangChain-Core |
| Evaluation | Custom retrieval metrics + LLM-as-Judge |
| Logging | SQLite `eval_runs` table |

## Evaluation Harness

| Harness | What it does | File |
|---|---|---|
| Retrieval Evaluation | Precision@K, Recall@K, F1@K, MRR, NDCG@K | `notebooks/eval_retrieval.py` |
| Generation Evaluation | Faithfulness, Answer Relevancy, Context Recall (LLM-as-Judge) | `notebooks/eval_ragas.py` |
| CI Quality Gate | Pass/Fail vs thresholds, blocks deployment, logs to SQLite | `notebooks/quality_gate.py` |
| Observability | Structured logging to `eval_runs` table with run history | `notebooks/quality_gate.py` |

## Results

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Precision@5 | 0.35 | ≥ 0.30 | ✅ PASS |
| Recall@5 | **0.91** | ≥ 0.70 | ✅ PASS |
| F1@5 | 0.50 | ≥ 0.40 | ✅ PASS |
| MRR | **0.97** | ≥ 0.70 | ✅ PASS |
| NDCG@5 | **0.87** | ≥ 0.70 | ✅ PASS |
| Faithfulness | **1.00** | ≥ 0.70 | ✅ PASS |
| Answer Relevancy | **1.00** | ≥ 0.70 | ✅ PASS |
| Context Recall | 0.70 | ≥ 0.60 | ✅ PASS |

> All 8 metrics passed CI quality gate — deployment approved ✅

## Data Sources

| Source | Format | Store | Purpose |
|---|---|---|---|
| KB Articles (6 docs) | Markdown | ChromaDB | Policy rules — return, warranty, account, shipping |
| Support Tickets (20) | JSON | ChromaDB | Past resolved cases — how issues were handled |
| Product Catalog (20 SKUs) | SQLite | SQL query | Exact facts — warranty_months, returnable, price |
| Golden Dataset (20 Q&A) | JSON | Eval only | Ground truth for evaluation harness |

## Project Structure
ecommerce-rag-eval/

├── data/

│   ├── kb_articles/        # 6 policy markdown files

│   ├── tickets.json        # 20 historical support tickets

│   └── products.db         # SQLite — 20 SKUs

├── notebooks/

│   ├── ingest.py           # chunking + embedding + ChromaDB

│   ├── rag_chain.py        # retrieval + generation pipeline

│   ├── eval_retrieval.py   # Precision@K, Recall@K, MRR, NDCG

│   ├── eval_ragas.py       # LLM-as-judge scoring

│   └── quality_gate.py     # CI gate + SQLite logging

├── eval/

│   └── golden_dataset.json # 20 manually written Q&A pairs

├── .github/

└── requirements.txt


## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama (separate terminal)
ollama serve
ollama pull llama3

# 3. Ingest data into ChromaDB (run once)
python notebooks/ingest.py

# 4. Run RAG chatbot
python notebooks/rag_chain.py

# 5. Run evaluation harness
python notebooks/eval_retrieval.py
python notebooks/eval_ragas.py

# 6. Check CI quality gate
python notebooks/quality_gate.py
```

## Key Design Decisions

**Why three data sources?**
Unstructured policy text → semantic search (ChromaDB).
Structured product facts → exact SQL lookup (SQLite).
Mixing both gives accurate, grounded answers.

**Why LLM-as-Judge instead of RAGAS library?**
RAGAS uses LLM-as-Judge internally anyway.
Building it from scratch shows deeper understanding
and removes external dependency issues.

**Why CI Quality Gate?**
A RAG system without evaluation is a liability.
Every KB change could silently degrade quality.
The gate ensures regressions are caught before production.

## Author

Pranay — Data Scientist / GenAI Engineer(Fresher)
