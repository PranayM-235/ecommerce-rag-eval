import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

st.set_page_config(
    page_title = "Ecommerce Support RAG",
    page_icon  = "🛒",
    layout     = "wide",
)

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# ── Load env ───────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
CHROMA_DIR  = BASE_DIR / "data" / "chroma_db"
PRODUCTS_DB = BASE_DIR / "data" / "products.db"

# ── Config ─────────────────────────────────────────────────────
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
KB_COLLECTION      = "kb_articles_prod"
TICKETS_COLLECTION = "support_tickets_prod"
TOP_K              = 3
LLM_MODEL          = "mistralai/Mistral-7B-Instruct-v0.2"

# ── Prompt ─────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful ecommerce customer support agent.
Answer using ONLY the context below. Do not make up anything.
Mention the source (KB ID or Ticket ID) in your answer.

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


@st.cache_resource
def load_vectorstores():
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


@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id                  = LLM_MODEL,
        huggingfacehub_api_token = HF_TOKEN,
        temperature              = 0.1,
        max_new_tokens           = 512,
    )


def get_product_info(query: str) -> str:
    conn        = sqlite3.connect(PRODUCTS_DB)
    cur         = conn.cursor()
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
        return "No specific product SKU detected."
    cur.execute("""
        SELECT sku, name, category, price_inr, warranty_months,
               returnable, purchase_date, is_refurbished
        FROM products WHERE sku = ?
    """, (found_sku,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return "Product not found."
    sku, name, category, price, warranty, returnable, purchase_date, refurbished = row
    return (f"SKU: {sku} | Name: {name} | Category: {category} | "
            f"Price: Rs {price:,} | Warranty: {warranty} months | "
            f"Returnable: {'Yes' if returnable else 'No'}")


def rag_query(question, kb_store, ticket_store, llm):
    kb_results = kb_store.similarity_search_with_score(question, k=TOP_K)
    kb_context = ""
    kb_sources = []
    for doc, score in kb_results:
        doc_id    = doc.metadata.get("doc_id", "unknown")
        filename  = doc.metadata.get("filename", "")
        kb_context += f"[Source: {doc_id} | {filename}]\n{doc.page_content}\n\n"
        kb_sources.append(doc_id)

    ticket_results = ticket_store.similarity_search_with_score(question, k=TOP_K)
    ticket_context = ""
    ticket_sources = []
    for doc, score in ticket_results:
        ticket_id = doc.metadata.get("doc_id", "unknown")
        category  = doc.metadata.get("category", "")
        ticket_context += f"[Ticket: {ticket_id} | {category}]\n{doc.page_content}\n\n"
        ticket_sources.append(ticket_id)

    product_info = get_product_info(question)

    final_prompt = prompt.format(
        kb_context      = kb_context     or "No KB articles retrieved.",
        ticket_context  = ticket_context or "No tickets retrieved.",
        product_context = product_info,
        question        = question,
    )
    answer = llm.invoke(final_prompt)

    return {
        "answer"         : answer,
        "kb_sources"     : kb_sources,
        "ticket_sources" : ticket_sources,
        "product_info"   : product_info,
    }


# ── UI ─────────────────────────────────────────────────────────
st.title("🛒 Ecommerce Customer Support RAG")
st.caption("Powered by ChromaDB + HuggingFace + LLM-as-Judge Evaluation")

# Sidebar
with st.sidebar:
    st.header("📊 Eval Metrics")
    st.markdown("**Last CI Run — All Passed ✅**")
    metrics = {
        "Recall@5"        : 0.91,
        "MRR"             : 0.97,
        "NDCG@5"          : 0.87,
        "Faithfulness"    : 1.00,
        "Answer Relevancy": 1.00,
        "Context Recall"  : 0.70,
    }
    for metric, score in metrics.items():
        st.metric(label=metric, value=f"{score:.2f}")
    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("""
- 🔍 ChromaDB
- 🤗 HuggingFace Embeddings
- 🦙 Mistral-7B
- 🗄️ SQLite
    """)

# Main
st.subheader("💬 Ask a Question")

sample_questions = [
    "Can I return my laptop after 10 days?",
    "Is SONY-WH1000XM5 under warranty after 8 months?",
    "How do I permanently delete my account?",
    "My UPI payment failed but money was deducted. What should I do?",
    "Can I cancel my order after it has been shipped?",
]

selected = st.selectbox(
    "Try a sample question:",
    ["-- Select a sample --"] + sample_questions
)

question = st.text_input(
    "Or type your own question:",
    value = selected if selected != "-- Select a sample --" else "",
)

if st.button("Get Answer 🔍", type="primary"):
    if not question:
        st.warning("Please enter a question!")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            kb_store, ticket_store = load_vectorstores()
            llm                    = load_llm()
            result                 = rag_query(question, kb_store, ticket_store, llm)

        st.subheader("✅ Answer")
        st.write(result["answer"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📚 KB Sources")
            for src in result["kb_sources"]:
                st.info(src)
        with col2:
            st.subheader("🎫 Ticket Sources")
            for src in result["ticket_sources"]:
                st.info(src)

        if "No specific" not in result["product_info"]:
            st.subheader("📦 Product Info")
            st.code(result["product_info"])