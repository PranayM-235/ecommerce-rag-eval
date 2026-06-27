
import os
import json
import sqlite3
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
KB_DIR         = BASE_DIR / r"C:\Users\prana\OneDrive\Desktop\projects\ecommerce-rag-eval\ecommerce-rag-eval\data\kb_articles"
TICKETS_FILE   = BASE_DIR / r"data\tickets.json"
PRODUCTS_DB    = BASE_DIR /r"data\products.db"
CHROMA_DIR     = BASE_DIR / r"data\chroma_db"

# ── Config ─────────────────────────────────────────────────────
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
KB_COLLECTION      = "kb_articles_prod"
TICKETS_COLLECTION = "support_tickets_prod"


def load_kb_articles():
    documents = []
    for md_file in sorted(KB_DIR.glob("*.md")):
        text   = md_file.read_text(encoding="utf-8")
        doc_id = md_file.stem.split("_")[0]
        documents.append({
            "doc_id"  : doc_id,
            "text"    : text,
            "metadata": {
                "doc_id"  : doc_id,
                "source"  : "kb_articles",
                "filename": md_file.name,
            }
        })
    print(f"[KB]      Loaded {len(documents)} KB articles")
    return documents


def load_tickets():
    with open(TICKETS_FILE, encoding="utf-8") as f:
        tickets = json.load(f)
    documents = []
    for ticket in tickets:
        combined_text = (
            f"Customer Query: {ticket['customer_query']}\n"
            f"Resolution: {ticket['resolution']}"
        )
        documents.append({
            "doc_id"  : ticket["ticket_id"],
            "text"    : combined_text,
            "metadata": {
                "doc_id"  : ticket["ticket_id"],
                "source"  : "support_tickets",
                "category": ticket.get("category", ""),
                "sku"     : ticket.get("sku", "N/A"),
                "status"  : ticket.get("status", ""),
                "date"    : ticket.get("date", ""),
            }
        })
    print(f"[Tickets] Loaded {len(documents)} support tickets")
    return documents


def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    chunked = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(chunks):
            chunked.append({
                "text"    : chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index" : i,
                    "total_chunks": len(chunks),
                }
            })
    return chunked


def load_into_chroma(chunks, collection_name, embeddings):
    print(f"[Chroma]  Loading {len(chunks)} chunks into '{collection_name}'...")
    texts     = [c["text"]     for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids       = [
        f"{c['metadata']['doc_id']}_chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]
    vectorstore = Chroma(
        collection_name    = collection_name,
        embedding_function = embeddings,
        persist_directory  = str(CHROMA_DIR),
    )
    try:
        existing = vectorstore.get()
        if existing["ids"]:
            vectorstore.delete(ids=existing["ids"])
            print(f"[Chroma]  Cleared {len(existing['ids'])} old chunks")
    except Exception:
        pass
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    print(f"[Chroma]  Done — {len(chunks)} chunks stored in '{collection_name}'")
    return vectorstore


def verify_products_db():
    conn  = sqlite3.connect(PRODUCTS_DB)
    cur   = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM products")
    count = cur.fetchone()[0]
    conn.close()
    print(f"[SQLite]  products.db OK — {count} SKUs found")


def main():
    print("=" * 60)
    print("  ECOMMERCE RAG — DATA INGESTION PIPELINE")
    print("=" * 60)

    print("\n[Embed]   Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBEDDING_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )
    print("[Embed]   Model ready\n")

    print("--- Loading raw data ---")
    kb_docs     = load_kb_articles()
    ticket_docs = load_tickets()
    verify_products_db()

    print("\n--- Chunking ---")
    kb_chunks     = chunk_documents(kb_docs,     chunk_size=512, chunk_overlap=50)
    ticket_chunks = chunk_documents(ticket_docs, chunk_size=256, chunk_overlap=30)
    print(f"[Chunk]   KB       → {len(kb_chunks)} chunks")
    print(f"[Chunk]   Tickets  → {len(ticket_chunks)} chunks")

    print("\n--- Storing in ChromaDB ---")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    load_into_chroma(kb_chunks,     KB_COLLECTION,      embeddings)
    load_into_chroma(ticket_chunks, TICKETS_COLLECTION, embeddings)

    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE!")
    print(f"  ChromaDB saved at : {CHROMA_DIR}")
    print("  Next step → python src/rag_chain.py")
    print("=" * 60)


if __name__ == "__main__":
    main()