import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# --------------------
# CONFIG
# --------------------
load_dotenv()

CHROMA_PATH = os.path.join("data", "chroma")
# IMPORTANT: doit matcher ingest.py
COLLECTION_NAME = "knowledge_base"

# Modèle Groq (génération)
GEN_MODEL = "llama-3.1-8b-instant"

# Réglages anti "Request too large"
TOP_K = 3
MAX_CHARS_PER_DOC = 1200         # tronque chaque doc
MAX_TOTAL_CONTEXT_CHARS = 4500   # tronque le contexte total
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 400

# --------------------
# HELPERS
# --------------------
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY manquante. Mets-la dans un fichier .env à la racine:\n"
            "GROQ_API_KEY=xxxx"
        )
    return Groq(api_key=api_key)

def retrieve_from_chroma(query: str, k: int = TOP_K):
    """
    IMPORTANT: Cette fonction suppose que ta collection a été indexée AVEC embeddings.
    Comme dans ton ingest.py, tu as ajouté embeddings=... au moment du add().

    Du coup ici on peut utiliser query_texts=[query] SANS recalculer d'embedding
    (Chroma peut faire de la recherche si un embedding function est configuré,
     mais dans ton cas tu as déjà stocké les embeddings -> Chroma peut matcher
     selon sa config.
     
    Si ça ne marche pas chez toi, dis-moi, et je te donne la version
    qui réutilise exactement le même embedder que ingest.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # On tente query_texts (plus simple). Si ton Chroma exige query_embeddings,
    # il faudra fournir un embedder identique à ingest.
    res = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"]
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return list(zip(docs, metas))

def build_context(contexts):
    chunks = []
    total = 0

    for i, (doc, meta) in enumerate(contexts, start=1):
        src = meta.get("source_url", "") if isinstance(meta, dict) else ""
        doc = (doc or "").strip()

        # tronquer chaque doc
        if len(doc) > MAX_CHARS_PER_DOC:
            doc = doc[:MAX_CHARS_PER_DOC] + "..."

        piece = f"[Source {i}] {src}\n{doc}\n"
        if total + len(piece) > MAX_TOTAL_CONTEXT_CHARS:
            break
        chunks.append(piece)
        total += len(piece)

    return "\n".join(chunks)

def rag_answer(client: Groq, query: str, contexts):
    context_text = build_context(contexts)

    system = (
        "Tu es un assistant RAG. Réponds uniquement à partir du CONTEXTE. "
        "Si le CONTEXTE ne contient pas la réponse, dis 'Je ne trouve pas l'information dans le contexte.'"
    )

    user = f"CONTEXTE:\n{context_text}\n\nQUESTION: {query}\nRéponds en français."

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return resp.choices[0].message.content, context_text


# --------------------
# STREAMLIT UI
# --------------------
st.set_page_config(page_title="RAG Chatbot (Groq + Chroma)", layout="centered")
st.title("🤖 RAG Chatbot (Groq + ChromaDB)")
st.caption("UI Streamlit — avec réduction automatique du contexte pour éviter l'erreur 413.")

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Pose ta question…")

if query:
    # Affiche message user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Répondre
    with st.chat_message("assistant"):
        try:
            groq_client = get_groq_client()

            contexts = retrieve_from_chroma(query, k=TOP_K)

            answer, ctx_text = rag_answer(groq_client, query, contexts)
            st.markdown(answer)

            with st.expander("🔎 Contexte utilisé (tronqué automatiquement)"):
                st.text(ctx_text)

            with st.expander("📌 Sources"):
                for i, (_, meta) in enumerate(contexts, start=1):
                    src = meta.get("source_url", "") if isinstance(meta, dict) else ""
                    st.write(f"{i}) {src}")

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(str(e))
