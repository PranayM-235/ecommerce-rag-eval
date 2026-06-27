"""
eval_retrieval.py - Retrieval Metrics
Calculates: Precision@K, Recall@K, F1@K, MRR, NDCG@K
"""

import json
import math
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
CHROMA_DIR     = BASE_DIR / "data" / "chroma_db"
GOLDEN_DATASET = BASE_DIR / "eval" / "golden_dataset.json"

# ── Config ─────────────────────────────────────────────────────
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
KB_COLLECTION      = "kb_articles_prod"
TICKETS_COLLECTION = "support_tickets_prod"
TOP_K              = 5


def load_golden_dataset():
    with open(GOLDEN_DATASET, encoding="utf-8") as f:
        return json.load(f)


def get_vectorstores():
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBEDDING_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )
    kb_store = Chroma(
        collection_name    = KB_COLLECTION,
        embedding_function = embeddings,
        persist_directory  = str(CHROMA_DIR),
    )
    ticket_store = Chroma(
        collection_name    = TICKETS_COLLECTION,
        embedding_function = embeddings,
        persist_directory  = str(CHROMA_DIR),
    )
    return kb_store, ticket_store


def retrieve_docs(question, kb_store, ticket_store, k=TOP_K):
    """
    Query se top-K docs retrieve karo dono stores se.
    Returns: list of retrieved doc_ids (order matters for MRR/NDCG)
    """
    kb_results     = kb_store.similarity_search_with_score(question, k=k)
    ticket_results = ticket_store.similarity_search_with_score(question, k=k)

    # Dono results combine karo score ke basis pe sort karke
    all_results = []
    for doc, score in kb_results:
        all_results.append((doc.metadata.get("doc_id", ""), score))
    for doc, score in ticket_results:
        all_results.append((doc.metadata.get("doc_id", ""), score))

    # Score ascending sort (lower = more similar in chromadb)
    all_results.sort(key=lambda x: x[1])

    # Top-K unique doc_ids lo
    seen     = set()
    top_docs = []
    for doc_id, score in all_results:
        if doc_id not in seen:
            seen.add(doc_id)
            top_docs.append(doc_id)
        if len(top_docs) == k:
            break

    return top_docs


# ── Metric Functions ───────────────────────────────────────────

def precision_at_k(retrieved, relevant, k):
    """
    Top-K retrieved mein se kitne relevant hain.
    Formula: relevant docs in top K / K
    """
    retrieved_k = retrieved[:k]
    hits        = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved, relevant, k):
    """
    Total relevant docs mein se kitne top-K mein mile.
    Formula: relevant docs in top K / total relevant docs
    """
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits        = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(relevant)


def f1_at_k(retrieved, relevant, k):
    """
    Precision aur Recall ka harmonic mean.
    Formula: 2 * (P * R) / (P + R)
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def mrr(retrieved, relevant):
    """
    Pehla relevant doc kitni jaldi mila.
    Formula: 1 / rank of first relevant doc
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    """
    Relevant docs kitne upar hain list mein - position matters.
    Formula: DCG / IDCG
    """
    retrieved_k = retrieved[:k]

    # DCG - actual order
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)  # log2(rank+1), rank starts at 1

    # IDCG - ideal order (sare relevant docs pehle hote)
    ideal_hits = min(len(relevant), k)
    idcg       = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ── Main Evaluation Function ───────────────────────────────────

def calculate_retrieval_metrics(k=TOP_K):
    """
    Golden dataset pe sare retrieval metrics calculate karo.
    Returns: dict with all metric scores + per-query results
    """
    print("=" * 60)
    print("  RETRIEVAL EVALUATION HARNESS")
    print(f"  K = {k}")
    print("=" * 60)

    dataset                  = load_golden_dataset()
    kb_store, ticket_store   = get_vectorstores()

    all_precision = []
    all_recall    = []
    all_f1        = []
    all_mrr       = []
    all_ndcg      = []
    per_query     = []

    for item in dataset:
        question     = item["question"]
        relevant     = set(item["relevant_doc_ids"])
        query_id     = item["query_id"]
        category     = item["category"]

        # Retrieve
        retrieved = retrieve_docs(question, kb_store, ticket_store, k=k)

        # Calculate metrics
        p    = precision_at_k(retrieved, relevant, k)
        r    = recall_at_k(retrieved, relevant, k)
        f1   = f1_at_k(retrieved, relevant, k)
        mrr_ = mrr(retrieved, relevant)
        ndcg = ndcg_at_k(retrieved, relevant, k)

        all_precision.append(p)
        all_recall.append(r)
        all_f1.append(f1)
        all_mrr.append(mrr_)
        all_ndcg.append(ndcg)

        per_query.append({
            "query_id"  : query_id,
            "category"  : category,
            "question"  : question[:60] + "...",
            "retrieved" : retrieved,
            "relevant"  : list(relevant),
            "precision" : round(p, 4),
            "recall"    : round(r, 4),
            "f1"        : round(f1, 4),
            "mrr"       : round(mrr_, 4),
            "ndcg"      : round(ndcg, 4),
        })

        # Per query print
        status = "✅" if p >= 0.3 and r >= 0.4 else "❌"
        print(f"{status} {query_id} | P@{k}={p:.2f} R@{k}={r:.2f} "
              f"F1={f1:.2f} MRR={mrr_:.2f} NDCG={ndcg:.2f} | {category}")

    # Average scores
    avg_scores = {
        f"precision@{k}" : round(sum(all_precision) / len(all_precision), 4),
        f"recall@{k}"    : round(sum(all_recall)    / len(all_recall),    4),
        f"f1@{k}"        : round(sum(all_f1)        / len(all_f1),        4),
        "mrr"            : round(sum(all_mrr)        / len(all_mrr),       4),
        f"ndcg@{k}"      : round(sum(all_ndcg)       / len(all_ndcg),      4),
    }

    print("\n" + "=" * 60)
    print("  AVERAGE RETRIEVAL SCORES")
    print("=" * 60)
    for metric, score in avg_scores.items():
        bar    = "█" * int(score * 20)
        status = "✅" if score >= 0.5 else "❌"
        print(f"  {status} {metric:<15} : {score:.4f}  {bar}")

    return {"avg_scores": avg_scores, "per_query": per_query}


if __name__ == "__main__":
    calculate_retrieval_metrics(k=TOP_K)