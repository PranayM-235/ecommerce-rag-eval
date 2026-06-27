"""
eval_ragas.py - Manual RAGAS-style Scoring
No external RAGAS library needed — pure Python
Metrics: Faithfulness, Answer Relevancy, Context Recall
"""

import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
CHROMA_DIR     = BASE_DIR / "data" / "chroma_db"
GOLDEN_DATASET = BASE_DIR / "eval" / "golden_dataset.json"

# ── Config ─────────────────────────────────────────────────────
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
KB_COLLECTION      = "kb_articles_prod"
TICKETS_COLLECTION = "support_tickets_prod"
OLLAMA_MODEL       = "llama3"
TOP_K              = 3

# ── Prompt for RAG answer ──────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["kb_context", "ticket_context", "question"],
    template="""You are a helpful ecommerce customer support agent.
Answer using ONLY the context below. Do not make up anything.

--- KNOWLEDGE BASE ---
{kb_context}

--- HISTORICAL TICKETS ---
{ticket_context}

--- CUSTOMER QUESTION ---
{question}

--- YOUR ANSWER ---"""
)

# ── Scoring Prompts (LLM as Judge) ────────────────────────────
FAITHFULNESS_PROMPT = PromptTemplate(
    input_variables=["context", "answer"],
    template="""You are an evaluator. Check if the answer is fully supported by the context.
Score 1.0 if every claim in the answer is supported by context.
Score 0.5 if some claims are supported.
Score 0.0 if the answer contains information not in the context.
Reply with ONLY a number: 0.0, 0.5, or 1.0

Context: {context}
Answer: {answer}
Score:"""
)

RELEVANCY_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""You are an evaluator. Check if the answer directly addresses the question.
Score 1.0 if the answer fully addresses the question.
Score 0.5 if the answer partially addresses the question.
Score 0.0 if the answer does not address the question.
Reply with ONLY a number: 0.0, 0.5, or 1.0

Question: {question}
Answer: {answer}
Score:"""
)

CONTEXT_RECALL_PROMPT = PromptTemplate(
    input_variables=["ground_truth", "context"],
    template="""You are an evaluator. Check if the context contains the information needed to produce the ground truth answer.
Score 1.0 if context contains all information needed.
Score 0.5 if context contains some information needed.
Score 0.0 if context is missing key information.
Reply with ONLY a number: 0.0, 0.5, or 1.0

Ground Truth Answer: {ground_truth}
Retrieved Context: {context}
Score:"""
)


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


def get_rag_answer(question, kb_store, ticket_store, llm):
    """RAG answer + contexts return karo."""
    kb_results     = kb_store.similarity_search(question, k=TOP_K)
    ticket_results = ticket_store.similarity_search(question, k=TOP_K)

    kb_context     = "\n\n".join([d.page_content for d in kb_results])
    ticket_context = "\n\n".join([d.page_content for d in ticket_results])
    all_context    = kb_context + "\n\n" + ticket_context

    answer = llm.invoke(RAG_PROMPT.format(
        kb_context     = kb_context or "None",
        ticket_context = ticket_context or "None",
        question       = question,
    ))

    return answer, all_context


def parse_score(raw: str) -> float:
    """LLM ke response se float score nikalo."""
    try:
        for token in raw.strip().split():
            val = float(token)
            if 0.0 <= val <= 1.0:
                return val
    except:
        pass
    return 0.5  # default agar parse na ho


def score_faithfulness(context, answer, llm) -> float:
    raw = llm.invoke(FAITHFULNESS_PROMPT.format(
        context=context, answer=answer
    ))
    return parse_score(raw)


def score_relevancy(question, answer, llm) -> float:
    raw = llm.invoke(RELEVANCY_PROMPT.format(
        question=question, answer=answer
    ))
    return parse_score(raw)


def score_context_recall(ground_truth, context, llm) -> float:
    raw = llm.invoke(CONTEXT_RECALL_PROMPT.format(
        ground_truth=ground_truth, context=context
    ))
    return parse_score(raw)


def run_ragas_eval(sample_size=5):
    print("=" * 60)
    print("  RAGAS-STYLE GENERATION EVALUATION")
    print(f"  Sample: {sample_size} questions | Judge: Ollama llama3")
    print("=" * 60)

    dataset              = load_golden_dataset()[:sample_size]
    kb_store, ticket_store = get_vectorstores()
    llm                  = OllamaLLM(model=OLLAMA_MODEL, temperature=0)

    all_faithfulness = []
    all_relevancy    = []
    all_recall       = []
    results          = []

    for i, item in enumerate(dataset):
        question     = item["question"]
        ground_truth = item["ground_truth"]
        query_id     = item["query_id"]

        print(f"\n[{i+1}/{sample_size}] {query_id}: {question[:55]}...")

        # Step 1 - RAG answer generate karo
        answer, context = get_rag_answer(question, kb_store, ticket_store, llm)
        print(f"  Answer: {answer[:80]}...")

        # Step 2 - Score karo
        faith   = score_faithfulness(context, answer, llm)
        relev   = score_relevancy(question, answer, llm)
        recall  = score_context_recall(ground_truth, context, llm)

        all_faithfulness.append(faith)
        all_relevancy.append(relev)
        all_recall.append(recall)

        print(f"  Faithfulness={faith:.1f} | Relevancy={relev:.1f} | Context Recall={recall:.1f}")

        results.append({
            "query_id"    : query_id,
            "question"    : question[:60],
            "faithfulness": faith,
            "relevancy"   : relev,
            "recall"      : recall,
        })

    # Average scores
    avg_faith  = sum(all_faithfulness) / len(all_faithfulness)
    avg_relev  = sum(all_relevancy)    / len(all_relevancy)
    avg_recall = sum(all_recall)       / len(all_recall)

    scores = {
        "faithfulness"    : round(avg_faith,  4),
        "answer_relevancy": round(avg_relev,  4),
        "context_recall"  : round(avg_recall, 4),
    }

    print("\n" + "=" * 60)
    print("  AVERAGE RAGAS-STYLE SCORES")
    print("=" * 60)
    for metric, score in scores.items():
        bar    = "█" * int(score * 20)
        status = "✅" if score >= 0.7 else "❌"
        print(f"  {status} {metric:<22} : {score:.4f}  {bar}")

    return scores


if __name__ == "__main__":
    run_ragas_eval(sample_size=5)