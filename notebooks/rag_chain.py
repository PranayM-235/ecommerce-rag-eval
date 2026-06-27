import sqlite3
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
CHROMA_DIR  = BASE_DIR / "data" / "chroma_db"
PRODUCTS_DB = BASE_DIR / "data" / "products.db"

# ── Config ─────────────────────────────────────────────────────
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
KB_COLLECTION      = "kb_articles_prod"
TICKETS_COLLECTION = "support_tickets_prod"
TOP_K              = 3
OLLAMA_MODEL       = "llama3"

# ── Prompt ─────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful ecommerce customer support agent.

Answer using ONLY the context below.
If answer not found, say: "I don't have enough information. Please contact support."
Do NOT make up anything. Mention source (KB ID or Ticket ID) in your answer.

--- KNOWLEDGE BASE ---
{kb_context}

--- HISTORICAL TICKETS ---
{ticket_context}

--- PRODUCT INFO ---
{product_context}

--- CUSTOMER QUESTION ---
{question}

--- YOUR ANSWER ---"""

prompt = PromptTemplate(
    input_variables=["kb_context", "ticket_context", "product_context", "question"],
    template=PROMPT_TEMPLATE,
)


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


def get_product_info(query: str) -> str:
    conn = sqlite3.connect(PRODUCTS_DB)
    cur  = conn.cursor()

    cur.execute("SELECT sku FROM products")
    all_skus    = [row[0] for row in cur.fetchall()]
    found_sku   = None
    query_upper = query.upper()

    for sku in all_skus:
        if sku.upper() in query_upper:
            found_sku = sku
            break

    if not found_sku:
        conn.close()
        return "No specific product SKU detected in query."

    cur.execute("""
        SELECT sku, name, category, price_inr, warranty_months,
               returnable, purchase_date, is_refurbished
        FROM products WHERE sku = ?
    """, (found_sku,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return "Product not found in database."

    sku, name, category, price, warranty, returnable, purchase_date, refurbished = row

    return f"""SKU           : {sku}
Product Name  : {name}
Category      : {category}
Price         : Rs {price:,}
Warranty      : {warranty} months {"(Seller warranty)" if refurbished else "(Manufacturer warranty)"}
Returnable    : {"Yes" if returnable else "No"}
Purchase Date : {purchase_date}
Refurbished   : {"Yes" if refurbished else "No"}"""


def rag_query(question, kb_store, ticket_store, llm):
    # KB se retrieve
    kb_results        = kb_store.similarity_search_with_score(question, k=TOP_K)
    kb_context        = ""
    kb_docs_retrieved = []
    for doc, score in kb_results:
        doc_id    = doc.metadata.get("doc_id", "unknown")
        filename  = doc.metadata.get("filename", "")
        kb_context += f"[Source: {doc_id} | {filename}]\n{doc.page_content}\n\n"
        kb_docs_retrieved.append({"doc_id": doc_id, "score": round(float(score), 4)})

    # Tickets se retrieve
    ticket_results        = ticket_store.similarity_search_with_score(question, k=TOP_K)
    ticket_context        = ""
    ticket_docs_retrieved = []
    for doc, score in ticket_results:
        ticket_id = doc.metadata.get("doc_id", "unknown")
        category  = doc.metadata.get("category", "")
        ticket_context += f"[Ticket: {ticket_id} | Category: {category}]\n{doc.page_content}\n\n"
        ticket_docs_retrieved.append({"doc_id": ticket_id, "score": round(float(score), 4)})

    # SQLite product lookup
    product_info = get_product_info(question)

    # Prompt banao
    final_prompt = prompt.format(
        kb_context      = kb_context     or "No KB articles retrieved.",
        ticket_context  = ticket_context or "No tickets retrieved.",
        product_context = product_info,
        question        = question,
    )

    # LLM se answer lo
    answer = llm.invoke(final_prompt)

    return {
        "question"    : question,
        "answer"      : answer,
        "kb_docs"     : kb_docs_retrieved,
        "ticket_docs" : ticket_docs_retrieved,
        "product_info": product_info,
    }


def main():
    print("=" * 60)
    print("  ECOMMERCE RAG CHATBOT")
    print("=" * 60)

    print("\n[Init]  Loading vector stores...")
    kb_store, ticket_store = get_vectorstores()
    print("[Init]  Vector stores loaded")

    print("[Init]  Connecting to Ollama llama3...")
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)
    print("[Init]  LLM ready")

    print("\nType your question. Type 'exit' to quit.\n")

    while True:
        question = input("Customer Query > ").strip()

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        print("\n[Retrieving...]\n")
        result = rag_query(question, kb_store, ticket_store, llm)

        print("=" * 60)
        print("ANSWER:")
        print(result["answer"])
        print("\nSOURCES:")
        print("  KB Docs :", [d["doc_id"] for d in result["kb_docs"]])
        print("  Tickets :", [d["doc_id"] for d in result["ticket_docs"]])
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()