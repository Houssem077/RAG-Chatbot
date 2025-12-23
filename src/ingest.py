import os
import pandas as pd
import chromadb
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DATASET_PATH = os.path.join("data", "dataset.csv")
CHROMA_PATH = os.path.join("data", "chroma")
COLLECTION_NAME = "knowledge_base"

# 🔹 Embedding local (rapide et fiable)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = embedding_model.encode(texts,normalize_embeddings=True)
    return embeddings.tolist()

def main():
    df = pd.read_csv(DATASET_PATH)

    if "text" not in df.columns:
        raise ValueError("Colonne 'text' introuvable")

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    batch_size = 32
    ids, docs, metas = [], [], []

    for i, row in df.iterrows():
        doc_id = str(row.get("index", i))
        text = row["text"]
        source_url = str(row.get("source_url", ""))

        ids.append(doc_id)
        docs.append(text)
        metas.append({"source_url": source_url})

        if len(ids) >= batch_size:
            embeddings = embed_texts(docs)
            collection.add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings
            )
            ids, docs, metas = [], [], []

    if ids:
        embeddings = embed_texts(docs)
        collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings
        )

    print("✅ Index terminé – Collection:", COLLECTION_NAME)
    print("📦 Docs indexés:", len(df))

if __name__ == "__main__":
    main()
