import os
import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CHROMA_PATH = os.path.join("data", "chroma")
COLLECTION_NAME = "knowledge_base"  # doit matcher ingest.py

GEN_MODEL = "llama-3.1-8b-instant"

TOP_K = 2
MAX_DOC_CHARS = 1500
MAX_CONTEXT_CHARS = 6000
MAX_OUTPUT_TOKENS = 400

def get_client() -> Groq:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY manquant. Mets-le dans .env")
    return Groq(api_key=key)

def retrieve(query: str, k: int = TOP_K):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    res = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"]
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return list(zip(docs, metas))

def build_context(contexts):
    parts, total = [], 0
    for i, (doc, meta) in enumerate(contexts, start=1):
        src = (meta or {}).get("source_url", "")
        doc = (doc or "").strip()
        if len(doc) > MAX_DOC_CHARS:
            doc = doc[:MAX_DOC_CHARS] + " ..."

        block = f"[Source {i}] {src}\n{doc}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)

def answer(query: str, contexts):
    client = get_client()
    context_text = build_context(contexts)

    system = (
        "Tu es un assistant RAG. Tu réponds uniquement à partir du CONTEXTE fourni. "
        "Si le contexte ne suffit pas, dis clairement que tu ne trouves pas l'info."
    )
    user = f"CONTEXTE:\n{context_text}\n\nQUESTION: {query}\nRéponds en français."

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return resp.choices[0].message.content

def main():
    print("✅ RAG Chat (Groq) — tape 'exit' pour quitter\n")
    while True:
        q = input("You> ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        ctx = retrieve(q, k=TOP_K)
        print("\n--- CONTEXT (top matches) ---")
        for i, (_, m) in enumerate(ctx, start=1):
            print(f"{i}) {m.get('source_url','')}")
        print("----------------------------\n")

        a = answer(q, ctx)
        print("Bot>", a, "\n")

if __name__ == "__main__":
    main()
