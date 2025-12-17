import argparse
import os
import json
import pickle

import numpy as np
import faiss
import networkx as nx
from community import community_louvain
from PyPDF2 import PdfReader

import spacy
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

import gradio as gr
import ollama

parser = argparse.ArgumentParser(
    description="AmbedkarGPT – Graph-based RAG system using SEMRAG ideas"
)

parser.add_argument(
    "--pdf_path",
    type=str,
    default="data/Ambedkar_book.pdf",
    help="Path to Ambedkar PDF document"
)

parser.add_argument(
    "--embedding_model",
    type=str,
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Sentence embedding model"
)

parser.add_argument(
    "--local_top_k",
    type=int,
    default=5,
    help="Number of chunks retrieved via local graph search"
)

args = parser.parse_args()

PDF_PATH = args.pdf_path
LOCAL_TOP_K = args.local_top_k

COSINE_THRESHOLD = 0.35
ENTITY_SIM_THRESHOLD = 0.4

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.json")
GRAPH_PATH = os.path.join(CACHE_DIR, "graph.pkl")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    return vectors / norms


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - float(np.dot(a, b))


def load_pdf(path: str):
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"page": i + 1, "text": text})

    return pages


pages = load_pdf(PDF_PATH)

embedder = SentenceTransformer(args.embedding_model)

def semantic_chunk(text: str):
    sentences = [s for s in sent_tokenize(text) if s.strip()]
    if not sentences:
        return []

    embeddings = embedder.encode(sentences, convert_to_numpy=True)
    chunks = []
    current = [sentences[0]]

    for i in range(len(embeddings) - 1):
        if cosine_distance(embeddings[i], embeddings[i + 1]) < COSINE_THRESHOLD:
            current.append(sentences[i + 1])
        else:
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]

    chunks.append(" ".join(current))
    return chunks


if os.path.exists(CHUNKS_PATH):
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
else:
    chunks = []
    cid = 0

    for p in pages:
        for chunk in semantic_chunk(p["text"]):
            chunks.append({
                "id": cid,
                "page": p["page"],
                "text": chunk
            })
            cid += 1

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

nlp = spacy.load("en_core_web_sm")

if os.path.exists(GRAPH_PATH):
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
        G = data["graph"]
        entity_to_chunks = data["entity_map"]
else:
    G = nx.Graph()
    entity_to_chunks = {}

    for ch in chunks:
        doc = nlp(ch["text"])
        entities = [e.text.strip() for e in doc.ents]

        for entity in entities:
            G.add_node(entity)
            entity_to_chunks.setdefault(entity, []).append(ch["id"])

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(
            {"graph": G, "entity_map": entity_to_chunks},
            f
        )


partition = community_louvain.best_partition(G)


chunk_texts = [c["text"] for c in chunks]

if os.path.exists(EMBEDDINGS_PATH):
    chunk_embeddings = np.load(EMBEDDINGS_PATH)
else:
    chunk_embeddings = embedder.encode(chunk_texts, convert_to_numpy=True)
    chunk_embeddings = normalize(chunk_embeddings)
    np.save(EMBEDDINGS_PATH, chunk_embeddings)

index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
index.add(chunk_embeddings)

entity_texts = list(G.nodes())
entity_embeddings = embedder.encode(entity_texts, convert_to_numpy=True)
entity_embeddings = normalize(entity_embeddings)

def local_search(query: str):
    query_vec = normalize(
        embedder.encode([query], convert_to_numpy=True)
    )[0]

    similarities = entity_embeddings @ query_vec
    results = []

    for i, score in enumerate(similarities):
        if score >= ENTITY_SIM_THRESHOLD:
            entity = entity_texts[i]
            for cid in entity_to_chunks.get(entity, []):
                results.append(chunks[cid])

    return results[:LOCAL_TOP_K]


def generate_answer(query: str, context: str):
    prompt = (
        "Answer the question using only the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def chat_fn(message, history):
    retrieved_chunks = local_search(message)
    context = "\n\n".join(
        f"(page {c['page']}) {c['text']}"
        for c in retrieved_chunks
    )
    return generate_answer(message, context)


gr.ChatInterface(
    fn=chat_fn,
    title="AmbedkarGPT – Graph-based RAG (Local)",
    description="Local SEMRAG-inspired system using knowledge graphs and Ollama"
).launch()

